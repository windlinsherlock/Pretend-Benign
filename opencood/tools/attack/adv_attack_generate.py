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
    opt = parser.parse_args()
    return opt


def inference_single_ckpt(model, data_loader, opencood_dataset, opt, logger, device, ckpt, output_path, hypes):
    criterion = train_utils.create_loss(hypes)

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

    pbar = tqdm(total=len(data_loader), desc='adv_generate', dynamic_ncols=True)

    eps = hypes['attack_setting']['eps']
    pert_shape = hypes['attack_setting']['pert_shape']
    pert_alpha = hypes['attack_setting']['pert_alpha']
    adv_method = hypes['attack_setting']['adv_method']
    adv_iter = hypes['attack_setting']['adv_iter']
    pert_area_ratio = hypes['attack_setting']['pert_area_ratio']
    sequence_list = hypes['attack_setting']['sequence_list']
    attackers_path = hypes['attack_setting']['attackers_path']
    
    with open(attackers_path, 'rb') as file:
        attackers = pickle.load(file)

    path_list = output_path.split('/')
    data_index = path_list.index('attack') + 1
    path_list = path_list[data_index:]
    path_list.pop(-3)
    pert_data_path = '/'.join(path_list)
    current_path = os.getcwd()
    pert_data_path = os.path.join(current_path, '../../pert_data', pert_data_path)

    # print(pert_data_path)

    os.makedirs(pert_data_path, exist_ok=True)
    
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

        pert_data_filepath = os.path.join(pert_data_path, sample_name)
        os.makedirs(pert_data_filepath, exist_ok=True)
        pert_data_filepath = pert_data_filepath + '/pert.pkl'

        # print(pert_data_filepath)

        if os.path.exists(pert_data_filepath):
            with open(pert_data_filepath, 'rb') as file:
                pert_dataset = pickle.load(file)
            adv_pert = pert_dataset['pert_data']['pert']
            attacker_list = pert_dataset['pert_data']['attacker_list']
            adv_pert = torch.tensor(adv_pert)
        else:
            # get original ego agent class prediction of all anchors, without adv pert and fuse, return cls pred of all agents
            # since ground truth is unavailable during attack, use the fused result as the pseudo-GT label
            batch_data = train_utils.to_device(batch_data, device)
            with torch.no_grad():
                output_dict = model(batch_data, None, None, no_fuse=False, dataset=opencood_dataset, attack=False, need_box=False) 
                psm = output_dict['psm']                                                                        # (1, anchor_num * num_class, H, W)
                cls_preds = psm.permute(0, 2, 3, 1).contiguous()                                                # (B, H, W, anchor_num * num_class)
                cls_preds = cls_preds.view(cls_preds.shape[0], cls_preds.shape[1], cls_preds.shape[2], 2, 2)    # (B, H, W, anchor_num, num_class)

                fg_prob = torch.sigmoid(cls_preds[..., 1])
                pseudo_pos = fg_prob > hypes['postprocess']['target_args']['score_threshold'] - 0.05
                # print(hypes['postprocess']['target_args']['score_threshold'])
                target_dict = copy.deepcopy(batch_data['ego']['label_dict'])
                target_dict['pos_equal_one'] = pseudo_pos.clone().detach()
                target_dict = train_utils.to_device(target_dict, device)
                # print(target_dict['pos_equal_one'].shape)
                # print(target_dict['pos_equal_one'].sum())
                # exit()
            
            pert_dataset = {}
            pert_dataset['eps'] = eps
            pert_dataset['pert_shape'] = pert_shape
            pert_dataset['pert_alpha'] = pert_alpha
            pert_dataset['adv_method'] = adv_method
            pert_dataset['adv_iter'] = adv_iter
            pert_dataset['pert_area_ratio'] = pert_area_ratio
            pert_dataset['sequence_list'] = sequence_list

            pert_dataset['pert_data'] = {}

            attacker_list = attackers[sample_name]
            attacker_num = len(attacker_list)

            pert_dataset['pert_data'] = {
                'attacker_list': attacker_list
            }

            if adv_method == 'pgd':
                pert = torch.randn(attacker_num, pert_shape[0], pert_shape[1], pert_shape[2]) * 0.05
            elif adv_method == 'bim' or adv_method == 'cw-l2':
                pert = torch.zeros(attacker_num, pert_shape[0], pert_shape[1], pert_shape[2])
            else:
                raise NotImplementedError

            # randomly select a pert_area_ratio region for attack
            area_mask = torch.zeros(attacker_num, pert_shape[0], pert_shape[1], pert_shape[2])
            for j in range(attacker_num):
                total_pixels = pert_shape[1] * pert_shape[2]
                num_pixels_to_set = int(total_pixels * pert_area_ratio)
                indices = torch.randperm(total_pixels)[:num_pixels_to_set]
                y_indices = indices // pert_shape[2]
                x_indices = indices % pert_shape[2]
                area_mask[j, :, y_indices, x_indices] = 1
            pert = pert.to(device)
            area_mask = area_mask.to(device)
            area_mask.requires_grad = False
            pert.data = pert.data * area_mask.data

            if adv_method == 'cw-l2':
                optimizer = optim.Adam([pert], lr=pert_alpha)
            
            pert.requires_grad = True
            for j in range(adv_iter):

                if adv_method == 'pgd' or adv_method == 'bim':
                    if pert.grad is not None:
                        pert.grad.zero_()

                    output_dict = model(batch_data, pert, attacker_list, no_fuse=False, dataset=opencood_dataset, attack=True, need_box=False) 
                    cls_loss = criterion(output_dict, target_dict, adv_flag=True)
                    cls_loss.backward()

                    # print(cls_loss)

                    with torch.no_grad():
                        pert.data = pert.data + pert_alpha * pert.grad.sign()
                        pert.data = torch.clamp(pert.data, -eps, eps)
                        pert.data = pert.data * area_mask.data

                elif adv_method == 'cw-l2':
                    optimizer.zero_grad()

                    output_dict = model(batch_data, pert, attacker_list, no_fuse=False, dataset=opencood_dataset, attack=True, need_box=False) 
                    adv_loss = 0.0000 * torch.pow(pert, 2).sum() + 1 * criterion(output_dict, target_dict, adv_flag=True)
                    adv_loss.backward()

                    # print(adv_loss) 

                    optimizer.step()
                    pert.data = torch.clamp(pert.data, -eps, eps)
                    pert.data = pert.data * area_mask.data

            pert = pert.detach().clone()       

            adv_pert = pert.clone()
            
            # print(adv_pert.shape, adv_pert.max(), adv_pert.min(), adv_pert.abs().min(), adv_pert.abs().sum(), adv_pert.abs().mean())
            # exit()

            pert_dataset['pert_data']['pert'] = adv_pert.cpu().numpy()
            with open(pert_data_filepath, 'wb') as file:
                pickle.dump(pert_dataset, file)

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            adv_pert = adv_pert.to(device)

            pred_box_tensor, pred_score, gt_box_tensor, time_cost = model(batch_data, adv_pert, attacker_list, 
                no_fuse=False, dataset=opencood_dataset, attack=True, need_box=True)
            
            print(f"pred_box_tensor: {pred_box_tensor.shape}, pred_score: {pred_score.shape}, gt_box_tensor: {gt_box_tensor.shape}")
            # exit()

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
                npy_save_path = os.path.join(output_path, 'result', sample_name)
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
                                                   save_pcd=opt.save_pcd)
        pbar.update(1)
    pbar.close()

    eval_utils.eval_final_results(result_stat,
                                  output_path,
                                  opt.global_sort_detections,
                                  logger,
                                  ckpt)


def main():
    common_utils.seed_everything(2026)

    opt = test_parser()

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    
    output_path = train_utils.setup_train(hypes, opt.hypes_yaml)

    log_file = output_path + ('/attack_generate_%s.log' % (datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))

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
