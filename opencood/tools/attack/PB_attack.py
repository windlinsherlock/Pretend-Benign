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
from opencood.tools.attack.utils import PB_utils
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

    pbar = tqdm(total=len(data_loader), desc='PB_attack', dynamic_ncols=True)
    
    remove_mid_ratio = hypes['attack_setting']['remove_mid_ratio']
    pos_thresh_high = hypes['attack_setting']['pos_thresh_high']
    pos_thresh_mid = hypes['attack_setting']['pos_thresh_mid']
    pos_thresh_low = hypes['attack_setting']['pos_thresh_low']
    fusion_thresh = hypes['attack_setting']['fusion_thresh']
    iou_thresh = hypes['attack_setting']['iou_thresh']
    IF_num = hypes['attack_setting']['IF_num']
    heat_map_size = hypes['attack_setting']['heat_map_size']
    points_min = hypes['attack_setting']['points_min']
    voxel_size = hypes['attack_setting']['voxel_size']
    
    eps_list = hypes['attack_setting']['eps_list']
    pert_shape = hypes['attack_setting']['pert_shape']
    pert_alpha = hypes['attack_setting']['pert_alpha']
    adv_method = hypes['attack_setting']['adv_method']
    adv_iter = hypes['attack_setting']['adv_iter']
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
    
    points_min = torch.tensor(points_min)
    
    region_num = []
    
    for i, batch_data in enumerate(data_loader):
        num_cav = batch_data['ego']['record_len'][0].item() 

        cav_pose = batch_data['ego']['cav_pose']
        cav_pose = cav_pose[0][:num_cav]
        cav_pose = cav_pose.reshape(1, *cav_pose.shape)

        sample_name = batch_data['ego']['sample_name'][0]
        sequence_name = sample_name.split('/')[1]

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
            if 'LCF_mask' in pert_dataset['pert_data'].keys():
                LCF_mask = pert_dataset['pert_data']['LCF_mask']
                LCF_mask = torch.tensor(LCF_mask)
                LCF_mask = LCF_mask.to(device)
            else:
                LCF_mask = None
            adv_pert = torch.tensor(adv_pert)
        else:
            attacker_list = attackers[sample_name]
            attacker_num = len(attacker_list)

            # get original ego agent class prediction of all anchors, without adv pert and fuse, return cls pred of all agents
            batch_data = train_utils.to_device(batch_data, device)

            box_agent_list = attacker_list.copy()
            box_agent_list.append(0)

            # print(box_agent_list, attacker_list, num_cav)

            with torch.no_grad():
                
                original_boundingbox = model(batch_data, None, None, dataset=opencood_dataset, attack=False, 
                                             need_box=True, box_agent_list=box_agent_list)
                                             
                # mask_dict = {
                #     'HC_mask': HC_mask,
                #     'LCT_mask': LCT_mask, 
                #     'LCF_mask': LCF_mask,
                #     'IT_mask': IT_mask,
                #     'IF_mask': IF_mask
                # }
                # boundingbox_dict = {
                #     'HC_boxes': HC_boxes,
                #     'LCT_boxes': LCT_boxes,
                #     'LCF_boxes': LCF_boxes,
                #     'IT_boxes': IT_boxes
                # }

                cav_pose = cav_pose.to(device)
                points_min = points_min.to(device)

                mask_dict, boundingbox_dict = PB_utils.select_attack_region(original_boundingbox, heat_map_size, pos_thresh_high,
                        pos_thresh_mid, pos_thresh_low, fusion_thresh, iou_thresh, cav_pose, attacker_list, points_min, voxel_size, IF_num, device, remove_mid_ratio)
                # (H, W)    
                region_mask = mask_dict['HC_mask'] | mask_dict['LCT_mask'] | mask_dict['LCF_mask'] | mask_dict['IT_mask'] | mask_dict['IF_mask']
                # (1, 1, H, W)
                region_mask = region_mask.unsqueeze(0).unsqueeze(0)
                
                # print(region_mask.shape, region_mask.sum())
                region_num.append(region_mask.sum().item())
    
                # (H, W)
                clamp_map = torch.zeros((heat_map_size), dtype=boundingbox_dict['HC_boxes'].dtype, device=device)
                clamp_map[mask_dict['HC_mask']] = eps_list[0]
                clamp_map[mask_dict['LCT_mask']] = eps_list[1]
                clamp_map[mask_dict['LCF_mask']] = eps_list[2]
                clamp_map[mask_dict['IT_mask']] = eps_list[3]
                clamp_map[mask_dict['IF_mask']] = eps_list[4]
                # (1, 1, H, W)
                clamp_map = clamp_map.unsqueeze(0).unsqueeze(0)
                
            pert_dataset = {}
            pert_dataset['pert_data'] = {
                'attacker_list': attacker_list
            }

            pert = torch.randn(attacker_num, pert_shape[0], pert_shape[1], pert_shape[2]) * 0.005

            pert = pert.to(device)
            region_mask.requires_grad = False
            pert.data = pert.data * region_mask.data

            optimizer = optim.Adam([pert], lr=pert_alpha)
            
            pert.requires_grad = True
            for j in range(adv_iter):

                optimizer.zero_grad()

                torch.autograd.set_detect_anomaly(True)

                output_dict = model(batch_data, pert, attacker_list, dataset=opencood_dataset, attack=True, 
                                            need_box=False, box_agent_list=None, LCF_mask=mask_dict['LCF_mask'])

                anchor_box = batch_data['ego']['anchor_box']   
                adv_loss = criterion(mask_dict, boundingbox_dict, output_dict, anchor_box)
                adv_loss.backward()

                # print(adv_loss) 

                optimizer.step()
                pert.data = torch.clamp(pert.data, min=-clamp_map, max=clamp_map)
                pert.data = pert.data * region_mask.data

            pert = pert.detach().clone()       

            adv_pert = pert.clone()
            
            # print(adv_pert.shape, adv_pert.max(), adv_pert.min(), adv_pert.abs().min(), adv_pert.abs().sum(), adv_pert.abs().mean())
            # exit()
            LCF_mask = mask_dict['LCF_mask']
            pert_dataset['pert_data']['pert'] = adv_pert.cpu().numpy()
            pert_dataset['pert_data']['LCF_mask'] = mask_dict['LCF_mask'].cpu().numpy()
            pert_dataset['pert_data']['mask_dict'] = inference_utils.tensor_to_numpy(mask_dict)
            pert_dataset['pert_data']['boundingbox_dict'] = inference_utils.tensor_to_numpy(boundingbox_dict)
            with open(pert_data_filepath, 'wb') as file:
                pickle.dump(pert_dataset, file)

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            adv_pert = adv_pert.to(device)

            pred_box_tensor, pred_score, gt_box_tensor = model(batch_data, adv_pert, attacker_list, 
                dataset=opencood_dataset, attack=True, need_box=True, box_agent_list=None, LCF_mask=LCF_mask)

            print(f"pred_box_tensor: {pred_box_tensor.shape}, gt_box_tensor: {gt_box_tensor.shape}, {pred_box_tensor[pred_score > 0.3].shape}")
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
    
    logger.info("region_num: {}".format(sum(region_num) / len(region_num)))


def main():
    common_utils.seed_everything(2026)

    opt = test_parser()

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    
    output_path = train_utils.setup_train(hypes, opt.hypes_yaml)

    log_file = output_path + ('/PB_attack_%s.log' % (datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))

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
