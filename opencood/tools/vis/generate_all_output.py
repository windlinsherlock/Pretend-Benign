import argparse
import os
import time
import datetime
from tqdm import tqdm
from pathlib import Path
import torch
import open3d as o3d
from torch.utils.data import DataLoader
import random
import copy
import math
import os
import pickle
import torch.optim as optim
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils, common_utils
from opencood.visualization import vis_utils
import matplotlib.pyplot as plt
import glob
import re
import numpy as np


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed')
    parser.add_argument('--ckpt', type=int, required=True,
                        help='The epoch to inference')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--global_sort_detections', action='store_true', default=True, 
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    parser.add_argument('--save_pcd', action='store_true', default=False, 
                        help='whether to save points.')
    parser.add_argument('--no_fuse', action='store_true', default=False, 
                        help='whether to fuse features.')
    opt = parser.parse_args()
    return opt


def inference_single_ckpt(model, data_loader, opencood_dataset, opt, logger, device, ckpt, output_path, hypes):
    logger.info('Loading Model from checkpoint_{}'.format(ckpt))
    
    saved_path = opt.model_dir
    flag, model = train_utils.load_saved_model(saved_path, model, ckpt, logger)
    if flag == False:
        return 

    model.eval()
    
    pbar = tqdm(total=len(data_loader), desc='inference', dynamic_ncols=True)
    sequence_list = hypes['attack_setting']['sequence_list']
    attack = hypes['attack_setting']['attack']

    pert_data_path = hypes['attack_setting']['pert_data_path']
    
    for i, batch_data in enumerate(data_loader):
        num_cav = batch_data['ego']['record_len'][0].item() 

        cav_pose = batch_data['ego']['cav_pose']
        cav_pose = cav_pose[0][:num_cav]
        cav_pose = cav_pose.reshape(1, *cav_pose.shape)

        sample_name = batch_data['ego']['sample_name'][0]
        sequence_name = sample_name.split('/')[1]
        sample_idx = sample_name.split('/')[-1]
        batch_data['ego'].pop('sample_name', None)
        batch_data['ego'].pop('cav_pose', None)

        if sequence_name not in sequence_list or num_cav <= 1:
            pbar.update(1)
            continue

        pert_data_filepath = os.path.join(pert_data_path, sample_name, 'pert.pkl')

        if os.path.exists(pert_data_filepath) and attack:
            with open(pert_data_filepath, 'rb') as file:
                pert_dataset = pickle.load(file)
            adv_pert = pert_dataset['pert_data']['pert']
            attacker_list = pert_dataset['pert_data']['attacker_list']
            adv_pert = torch.tensor(adv_pert)
        elif not os.path.exists(pert_data_filepath) and attack:
            pbar.update(1)
            continue
        else:
            adv_pert = torch.zeros(3, 3)
            attacker_list = []
        
        # print(attack, adv_pert.shape, attacker_list, pert_data_filepath)

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            adv_pert = adv_pert.to(device)

            box_agent_list = list(range(0, num_cav))

            output_dict = model(batch_data, adv_pert, attacker_list, 
                dataset=opencood_dataset, attack=attack, need_box=True, box_agent_list=box_agent_list)
            
                        
            if opt.save_npy:
                npy_save_path = os.path.join(output_path, 'result', sample_name)
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_all_output(output_dict,
                                                cav_pose,
                                                npy_save_path,
                                                )
        pbar.update(1)
    pbar.close()


def main():
    common_utils.seed_everything(2026)
    
    opt = test_parser()

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    
    output_path = train_utils.setup_train(hypes, opt.hypes_yaml)

    log_file = output_path + ('/generate_all_output_%s.log' % (datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))

    # create a logger for test information
    logger = common_utils.create_logger(log_file)

    logger.info("-----------------parser argument:------------------")
    for key, val in vars(opt).items():
        logger.info('{:16} {}'.format(key, val))
    
    logger.info("-----------------hypes argument:------------------")
    train_utils.print_tree(hypes, indent=0, logger=logger)

    logger.info('Dataset Building')

    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    logger.info(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=10,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    logger.info('Creating Model')

    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    
    logger.info(f'----------- Model {hypes["name"]} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')
    logger.info(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    inference_single_ckpt(model, data_loader, opencood_dataset, opt, logger, device, opt.ckpt, output_path, hypes)

    


if __name__ == '__main__':
    main()
