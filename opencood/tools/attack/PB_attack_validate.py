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

    pbar = tqdm(total=len(data_loader), desc='PB_attack_validate', dynamic_ncols=True)

    sequence_list = hypes['attack_setting']['sequence_list']
    pert_data_path = hypes['attack_setting']['pert_data_path']

    normal_hc_score_list, normal_hc_iou_list, attack_hc_score_list, attack_hc_iou_list = [], [], [], []
    ict_remove_num_list, ict_num_list = [], []
    normal_icf_score_list, attack_icf_score_list = [], []
    it_remove_num_list , it_num_list = [], []
    
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

        pert_data_filepath = os.path.join(pert_data_path, sample_name, 'pert.pkl')

        if os.path.exists(pert_data_filepath):
            with open(pert_data_filepath, 'rb') as file:
                pert_dataset = pickle.load(file)
            adv_pert = pert_dataset['pert_data']['pert']
            attacker_list = pert_dataset['pert_data']['attacker_list']
            LCF_mask = pert_dataset['pert_data']['LCF_mask']
            LCF_mask = torch.tensor(LCF_mask)
            LCF_mask = LCF_mask.to(device)
            
            mask_dict = pert_dataset['pert_data']['mask_dict']
            boundingbox_dict = pert_dataset['pert_data']['boundingbox_dict']
            
            for key in mask_dict.keys():
                mask_dict[key] = torch.tensor(mask_dict[key])
                mask_dict[key] = mask_dict[key].to(device)
            for key in boundingbox_dict.keys():
                boundingbox_dict[key] = torch.tensor(boundingbox_dict[key])
                boundingbox_dict[key] = boundingbox_dict[key].to(device)            
                
            adv_pert = torch.tensor(adv_pert)
            adv_pert = adv_pert.to(device)

        
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            
            normal_output_dict = model(batch_data, adv_pert, attacker_list, dataset=opencood_dataset, attack=False, 
                                            need_box=False, box_agent_list=None, LCF_mask=None)
            
            attack_output_dict = model(batch_data, adv_pert, attacker_list, dataset=opencood_dataset, attack=True, 
                                            need_box=False, box_agent_list=None, LCF_mask=mask_dict['LCF_mask'])
            
            pred_box_tensor, pred_score, gt_box_tensor = model(batch_data, adv_pert, attacker_list, 
                dataset=opencood_dataset, attack=True, need_box=True, box_agent_list=None, LCF_mask=LCF_mask)
            
            print(f"pred_box_tensor: {pred_box_tensor.shape}, gt_box_tensor: {gt_box_tensor.shape}, {pred_box_tensor[pred_score > 0.3].shape}")

            anchor_box = batch_data['ego']['anchor_box']
            
            normal_hc_score, normal_hc_iou = PB_utils.get_hc_score_iou(normal_output_dict, mask_dict, boundingbox_dict, anchor_box)
            attack_hc_score, attack_hc_iou = PB_utils.get_hc_score_iou(attack_output_dict, mask_dict, boundingbox_dict, anchor_box)
            
            normal_hc_score_list.append(normal_hc_score)
            normal_hc_iou_list.append(normal_hc_iou)
            attack_hc_score_list.append(attack_hc_score)
            attack_hc_iou_list.append(attack_hc_iou)
            # print(normal_hc_score)
            # print(attack_hc_score)
            # print(normal_hc_iou)
            # print(attack_hc_iou)
            
            ict_remove_num, ict_num = PB_utils.get_ict_remove_ratio(pred_box_tensor, boundingbox_dict)
            
            ict_remove_num_list.append(ict_remove_num.item())
            ict_num_list.append(ict_num)
            # print(ict_remove_num.item(), ict_num)
            
            normal_icf_score = PB_utils.get_icf_score(normal_output_dict, mask_dict, boundingbox_dict)
            attack_icf_score = PB_utils.get_icf_score(attack_output_dict, mask_dict, boundingbox_dict)
            
            normal_icf_score_list.append(normal_icf_score)
            attack_icf_score_list.append(attack_icf_score)
            # print(normal_icf_score)
            # print(attack_icf_score)
            
            it_remove_num, it_num = PB_utils.get_it_remove_ratio(pred_box_tensor, boundingbox_dict)
            
            it_remove_num_list.append(it_remove_num.item())   
            it_num_list.append(it_num)
            # print(it_remove_num.item(), it_num)
                        
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
                     
        pbar.update(1)
    pbar.close()

    eval_utils.eval_final_results(result_stat,
                                  output_path,
                                  opt.global_sort_detections,
                                  logger,
                                  ckpt)
        
    normal_hc_score = torch.cat(normal_hc_score_list)
    normal_hc_iou = torch.cat(normal_hc_iou_list)
    attack_hc_score = torch.cat(attack_hc_score_list)
    attack_hc_iou = torch.cat(attack_hc_iou_list)
    
    logger.info('avg_normal_hc_score: {}'.format(normal_hc_score.mean().item()))
    logger.info('avg_normal_hc_iou: {}'.format(normal_hc_iou.mean().item()))
    logger.info('avg_attack_hc_score: {}'.format(attack_hc_score.mean().item()))
    logger.info('avg_attack_hc_iou: {}'.format(attack_hc_iou.mean().item()))
    
    ict_remove_num = sum(ict_remove_num_list)
    ict_num = sum(ict_num_list)
    logger.info('ict_remove_ratio: {}'.format(ict_remove_num / float(ict_num)))
    
    normal_icf_score = torch.cat(normal_icf_score_list)
    attack_icf_score = torch.cat(attack_icf_score_list)
    logger.info('avg_normal_icf_score: {}'.format(normal_icf_score.mean().item()))
    logger.info('avg_attack_icf_score: {}'.format(attack_icf_score.mean().item()))
    
    it_remove_num = sum(it_remove_num_list)
    it_num = sum(it_num_list)
    logger.info('it_remove_ratio: {}'.format(it_remove_num / float(it_num)))
    
    PB_validate_path = os.path.join(output_path, 'PB_validate.pkl')
    PB_validate = {
        'normal_hc_score': normal_hc_score,
        'normal_hc_iou': normal_hc_iou,
        'attack_hc_score': attack_hc_score,
        'attack_hc_iou': attack_hc_iou,
        'normal_icf_score': normal_icf_score,
        'attack_icf_score': attack_icf_score        
    }
    PB_validate = inference_utils.tensor_to_numpy(PB_validate)
    with open(PB_validate_path, 'wb') as file:
        pickle.dump(PB_validate, file)
    

def main():
    common_utils.seed_everything(2026)

    opt = test_parser()

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    
    output_path = train_utils.setup_train(hypes, opt.hypes_yaml)

    log_file = output_path + ('/PB_attack_validate_%s.log' % (datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))

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
