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
    parser.add_argument('--statistic', action='store_true', default=False, 
                        help='')
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
    
    time_cost_list = []
    collab_sim_list = []
    attacker_sim_list = []
    collab_match_list = []
    attacker_match_list = []

    tp_collab_list = []
    fp_collab_list = []
    tp_attacker_list = []
    fp_attacker_list = []
    all_collab_list = []
    all_attacker_list = []
    all_agent_list = []
    all_estimated_collab_list = []
    all_estimated_attacker_list = []

    pbar = tqdm(total=len(data_loader), desc='FLD', dynamic_ncols=True)

    sequence_list = hypes['attack_setting']['sequence_list']
    salient_threshold_I = hypes['FLD_setting']['salient_threshold_I']
    salient_threshold_U = hypes['FLD_setting']['salient_threshold_U']
    score_thresh = hypes['FLD_setting']['score_thresh']
    logger.info(f"{salient_threshold_I}, {salient_threshold_U}, {score_thresh}")

    pert_data_path = hypes['attack_setting']['pert_data_path']
    GPS_Attack = hypes['attack_setting'].get('GPS_Attack', False)
    
    scene_precision = {
        '0.3': [],
        '0.5': [],
        '0.7': []
    }
    
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
            pbar.update(1)
            continue

        # print(adv_pert.shape, attacker_list, pert_data_filepath)
        # exit()

        all_agent_list.append(num_cav-1)
        collab_list = [i for i in range(1, num_cav) if i not in attacker_list]
        all_collab_list.append(len(collab_list))
        all_attacker_list.append(len(attacker_list))

        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            adv_pert = adv_pert.to(device)
            
            estimated_collab_list, result, time_cost, collab_sim, attacker_sim, collab_match, attacker_match = model(batch_data, adv_pert, attacker_list, no_fuse=False, 
                dataset=opencood_dataset, salient_threshold_I=salient_threshold_I, salient_threshold_U=salient_threshold_U, score_thresh=score_thresh, LCF_mask=LCF_mask, GPS_Attack=GPS_Attack)

            pred_box_tensor = result['pred_box']
            pred_score = result['pred_score']
            gt_box_tensor = result['gt_box']

            collab_sim_list.extend(collab_sim)
            attacker_sim_list.extend(attacker_sim)
            collab_match_list.extend(collab_match)
            attacker_match_list.extend(attacker_match)

            print(f"pred_box_tensor: {pred_box_tensor.shape}, pred_score: {pred_score.shape}, gt_box_tensor: {gt_box_tensor.shape}")
            
            if opt.statistic:
                eval_utils.get_scene_precision(pred_box_tensor, gt_box_tensor, scene_precision)

            estimated_attacker_list = [i for i in range(1, num_cav) if i not in estimated_collab_list]

            all_estimated_attacker_list.append(len(estimated_attacker_list))
            all_estimated_collab_list.append(len(estimated_collab_list))

            tp_collab = [i for i in estimated_collab_list if i in collab_list]
            fp_collab = [i for i in estimated_collab_list if i not in collab_list]
            tp_attacker = [i for i in estimated_attacker_list if i in attacker_list]
            fp_attacker = [i for i in estimated_attacker_list if i not in attacker_list]

            tp_collab_list.append(len(tp_collab))
            fp_collab_list.append(len(fp_collab))
            tp_attacker_list.append(len(tp_attacker))
            fp_attacker_list.append(len(fp_attacker))

            time_cost_list.append(time_cost)
            
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
                                                   save_pcd=False)
        pbar.update(1)
    pbar.close()

    eval_utils.eval_final_results(result_stat,
                                  output_path,
                                  opt.global_sort_detections,
                                  logger,
                                  ckpt)
    
    sample_num = float(len(time_cost_list))
    avg_time_cost = sum(time_cost_list) / sample_num
    
    if sum(all_collab_list) == 0:
        tp_collab_recall, fp_collab_recall = 0., 0.
    else:
        tp_collab_recall = sum(tp_collab_list) / float(sum(all_collab_list))
        fp_collab_recall = sum(fp_collab_list) / float(sum(all_collab_list))
        
    if sum(all_attacker_list) == 0:
        tp_attacker_recall, fp_attacker_recall = 0., 0.
    else:
        tp_attacker_recall = sum(tp_attacker_list) / float(sum(all_attacker_list))
        fp_attacker_recall = sum(fp_attacker_list) / float(sum(all_attacker_list))

    if sum(all_estimated_collab_list) == 0:
        tp_collab_precision, fp_collab_precision = 0., 0.
    else:
        tp_collab_precision = sum(tp_collab_list) / float(sum(all_estimated_collab_list))
        fp_collab_precision = sum(fp_collab_list) / float(sum(all_estimated_collab_list))
    
    if sum(all_estimated_attacker_list) == 0:
        tp_attacker_precision, fp_attacker_precision = 0., 0.
    else:
        tp_attacker_precision = sum(tp_attacker_list) / float(sum(all_estimated_attacker_list))
        fp_attacker_precision = sum(fp_attacker_list) / float(sum(all_estimated_attacker_list))

    if sum(all_attacker_list) == 0:
        attack_succ_ratio = 0.
    else:
        attack_succ_ratio = sum(fp_collab_list) / float(sum(all_attacker_list))
    
    logger.info('avg_time_cost: {}'.format(avg_time_cost))
    logger.info('tp_collab_recall: {}, fp_collab_recall: {}'.format(tp_collab_recall, fp_collab_recall))
    logger.info('tp_attacker_recall: {}, fp_attacker_recall: {}'.format(tp_attacker_recall, fp_attacker_recall))
    logger.info('tp_collab_precision: {}, fp_collab_precision: {}'.format(tp_collab_precision, fp_collab_precision))
    logger.info('tp_attacker_precision: {}, fp_attacker_precision: {}'.format(tp_attacker_precision, fp_attacker_precision))
    logger.info('attack_succ_ratio: {}'.format(attack_succ_ratio))
    
    sim_match_path = os.path.join(output_path, 'sim_match.pkl')
    sim_match = {
        'collab_sim': collab_sim_list,
        'attacker_sim': attacker_sim_list,
        'collab_match': collab_match_list,
        'attacker_match': attacker_match_list
    }
    with open(sim_match_path, 'wb') as file:
        pickle.dump(sim_match, file)
    
    collab_sim = torch.tensor(collab_sim_list)
    attacker_sim = torch.tensor(attacker_sim_list)
    collab_match = torch.tensor(collab_match_list)
    attacker_match = torch.tensor(attacker_match_list)
    
    logger.info("----collab_sim-----")
    logger.info("min: {}, max: {}, avg: {}".format(collab_sim.min().item(), collab_sim.max().item(), collab_sim.mean().item()))
    logger.info("----Attacker_sim-----")
    logger.info("min: {}, max: {}, avg: {}".format(attacker_sim.min().item(), attacker_sim.max().item(), attacker_sim.mean().item()))

    logger.info("----collab_match-----")
    logger.info("min: {}, max: {}, avg: {}".format(collab_match.min().item(), collab_match.max().item(), collab_match.mean().item()))
    logger.info("----Attacker_match-----")
    logger.info("min: {}, max: {}, avg: {}".format(attacker_match.min().item(), attacker_match.max().item(), attacker_match.mean().item()))
    
    if opt.statistic:
        scene_p_filepath = os.path.join(output_path, 'scene_precision.pkl')
        logger.info(f"{len(scene_precision['0.3'])}, {len(scene_precision['0.5'])}, {len(scene_precision['0.7'])}")
        logger.info(f"{type(scene_precision['0.3'][0])}")
        # save as a pkl file
        with open(scene_p_filepath, 'wb') as file:
            
            pickle.dump(scene_precision, file)


def main():
    common_utils.seed_everything(2026)

    opt = test_parser()

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    
    output_path = train_utils.setup_train(hypes, opt.hypes_yaml)

    log_file = output_path + ('/FLD_%s.log' % (datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))

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
