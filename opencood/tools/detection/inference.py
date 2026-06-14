# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
import datetime
from tqdm import tqdm
from pathlib import Path
import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils, common_utils
from opencood.visualization import vis_utils
import matplotlib.pyplot as plt
import glob
import re


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument("--hypes_yaml", type=str, default=None,
                        help='data generation yaml file needed')
    parser.add_argument('--start_epoch', type=int, default=-1,
                        help='The first epoch to inference')
    parser.add_argument('--ckpt', type=int, default=-1,
                        help='The epoch to inference')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='late, early or intermediate')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--global_sort_detections', action='store_true', default=True, 
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    opt = parser.parse_args()
    return opt


def repeat_eval_ckpt(model, data_loader, opencood_dataset, opt, logger, device, start_epoch, output_dir, hypes):
    saved_path = opt.model_dir + '/ckpt/'
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)
    file_list = glob.glob(os.path.join(saved_path, '*epoch*.pth'))
    epochs_exist = []
    for file_ in file_list:
        result = re.findall(".*epoch(.*).pth.*", file_)         
        epochs_exist.append(int(result[0]))
    max_epoch = max(epochs_exist)
    logger.info(max_epoch)

    for epoch in range(start_epoch, max_epoch+1):
        logger.info('Loading Model from checkpoint_{}'.format(epoch))
        model_file = os.path.join(saved_path, 'net_epoch%d.pth' % epoch)
        if not os.path.exists(model_file):
            logger.info("{} is not exist.".format(model_file))
            continue
        eval_output_dir = Path(output_dir) / "eval_{}".format(epoch)
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        eval_single_ckpt(model, data_loader, opencood_dataset, opt, logger, device, epoch, eval_output_dir, hypes)


def eval_single_ckpt(model, data_loader, opencood_dataset, opt, logger, device, ckpt, eval_output_dir, hypes):
    logger.info('Loading Model from checkpoint_{}'.format(ckpt))
    
    saved_path = opt.model_dir
    flag, model = train_utils.load_saved_model(saved_path, model, ckpt, logger)
    if flag == False:
        return 

    model.eval()
    
    # Create the dictionary for evaluation.
    # also store the confidence score for each prediction
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    
    pbar = tqdm(total=len(data_loader), desc='inference', dynamic_ncols=True)
    
    sequence_list = hypes.get('sequence_list', None)
    
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

        if sequence_list is not None:
            if sequence_name not in sequence_list:
                pbar.update(1)
                continue
        
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            pred_box_tensor, pred_score, gt_box_tensor = model(batch_data, dataset=opencood_dataset, need_box=True)
           
            print(f"pred_box_tensor: {pred_box_tensor.shape}, pred_score: {pred_score.shape}, gt_box_tensor: {gt_box_tensor.shape}")

            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)

            if opt.save_npy:
                npy_save_path = os.path.join(eval_output_dir, 'result', sample_name)
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   pred_score,
                                                   gt_box_tensor,
                                                   cav_pose,
                                                   batch_data['ego'][
                                                       'origin_lidar'][0],
                                                    i,
                                                   npy_save_path,
                                                   save_pcd=True)                                 
        pbar.update(1)
    pbar.close()

    eval_utils.eval_final_results(result_stat,
                                  eval_output_dir,
                                  opt.global_sort_detections,
                                  logger,
                                  ckpt)


def main():
    common_utils.seed_everything(2026)

    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.start_epoch == -1 and opt.ckpt == -1), 'start_epoch and ckpt should be setted at least one.'
    
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    if opt.hypes_yaml is not None:
        eval_output_dir = train_utils.setup_train(hypes, opt.hypes_yaml)

    # create eval dir
    if opt.ckpt != -1:
        saved_path = opt.model_dir + '/ckpt/'
        model_file = os.path.join(saved_path, 'net_epoch%d.pth' % opt.ckpt)
        assert os.path.exists(model_file), "{} is not exist.".format(model_file)
        if opt.hypes_yaml is not None:
            eval_output_dir = Path(eval_output_dir) / "eval/eval_{}".format(opt.ckpt)
        else:
            eval_output_dir = Path(opt.model_dir) / "eval/eval_{}".format(opt.ckpt)
    else:
        saved_path = opt.model_dir + '/ckpt/'
        model_file = os.path.join(saved_path, 'net_epoch%d.pth' % opt.start_epoch)
        assert os.path.exists(model_file), "{} is not exist.".format(model_file)
        if opt.hypes_yaml is not None:
            eval_output_dir = Path(eval_output_dir) / "eval/start_from_{}".format(opt.start_epoch)
        else:
            eval_output_dir = Path(opt.model_dir) / "eval/start_from_{}".format(opt.start_epoch)
    
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = eval_output_dir / ('eval_%s.log' % (datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
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
                             num_workers=20,
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
    
    if opt.ckpt != -1:
        eval_single_ckpt(model, data_loader, opencood_dataset, opt, logger, device, opt.ckpt, eval_output_dir, hypes)
    else:
        repeat_eval_ckpt(model, data_loader, opencood_dataset, opt, logger, device, opt.start_epoch, eval_output_dir, hypes)

    


if __name__ == '__main__':
    main()
