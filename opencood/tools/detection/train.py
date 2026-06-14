# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics
import datetime
import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import common_utils
from easydict import EasyDict


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--ckpt', type=int, default=-1,
                        help='The epoch to be Continued')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt


def main():
    common_utils.seed_everything(2026)

    rank = int(os.getenv("RANK", 0))

    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    
    if opt.model_dir:
        saved_path = opt.model_dir
    else:
        saved_path = train_utils.setup_train(hypes, opt.hypes_yaml)
    
    log_file = saved_path + ('/train_%s.log' % (datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))

    # create a logger for test information
    logger = common_utils.create_logger(log_file, rank=rank)

    multi_gpu_utils.init_distributed_mode(opt, logger)
    
    logger.info("-----------------parser argument:------------------")
    for key, val in vars(opt).items():
        logger.info('{:16} {}'.format(key, val))
    
    logger.info("-----------------hypes argument:------------------")
    train_utils.print_tree(hypes, indent=0, logger=logger)

    logger.info('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset, shuffle=True)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=False,
                                  drop_last=True)

    logger.info('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if we want to train from last checkpoint.
    if opt.model_dir:
        flag, model = train_utils.load_saved_model(saved_path, model, ckpt=opt.ckpt)
        if flag == False:
            return
        init_epoch = opt.ckpt
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=False)
        model_without_ddp = model.module

    logger.info(f'----------- Model {hypes["name"]} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')
    logger.info(model)

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path + '/tensorboard') if rank == 0 else None  

    os.makedirs(saved_path + '/ckpt', exist_ok=True)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    logger.info('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            logger.info('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=(rank==0))

        for i, batch_data in enumerate(train_loader):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            
            batch_data['ego'].pop('sample_name', None)
            batch_data['ego'].pop('cav_pose', None)
            
            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            if not opt.half:
                ouput_dict = model(batch_data, dataset=opencood_train_dataset, need_box=False)
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data, dataset=opencood_train_dataset, need_box=False)
                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])

            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2, logger=logger)
            pbar2.update(1)

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)

        if rank == 0:
            if epoch % hypes['train_params']['save_freq'] == 0:
                torch.save(model_without_ddp.state_dict(), saved_path + '/ckpt/net_epoch%d.pth' % (epoch + 1))

    logger.info('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
