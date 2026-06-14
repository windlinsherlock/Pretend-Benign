import argparse
import os
import time
import datetime
from tqdm import tqdm
from pathlib import Path
import torch
import open3d as o3d
from collections import OrderedDict
from torch.utils.data import DataLoader
import torch.nn as nn
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
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Continued training path')
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed')
    opt = parser.parse_args()
    return opt


def main():
    common_utils.seed_everything(2026)

    opt = test_parser()

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=10,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    group_path = '/'.join(opt.hypes_yaml.split('/')[:-1])
    current_dir = os.getcwd()
    output_path = os.path.abspath(current_dir + '/' + group_path)
    if not os.path.exists(output_path):
        exit()
    save_path = os.path.join(output_path, 'attackers.pkl')

    print(save_path)

    pbar = tqdm(total=len(data_loader), desc='attackers_generate', dynamic_ncols=True)

    attacker_ratios = hypes['attack_setting']['attacker_ratios']
    sequence_list = hypes['attack_setting']['sequence_list']
    attackers = {}
    
    num = 0
    
    print(attacker_ratios, sequence_list)
    
    for i, batch_data in enumerate(data_loader):
        num_cav = batch_data['ego']['record_len'][0].item() 

        sample_name = batch_data['ego']['sample_name'][0]
        sequence_name = sample_name.split('/')[1]
        sample_idx = sample_name.split('/')[-1]
        
        if sequence_name not in sequence_list or num_cav <= 1:
            pbar.update(1)
            continue
        
        # generate attackers
        if isinstance(attacker_ratios, list):
            attacker_ratio = random.uniform(attacker_ratios[0], attacker_ratios[1])
        else:
            attacker_ratio = attacker_ratios

        attacker_num = math.ceil((num_cav-1) * attacker_ratio)
        cav_list = range(1, num_cav)
        attacker_list = random.sample(cav_list, attacker_num)
        attacker_list = sorted(attacker_list)

        attackers[sample_name] = attacker_list

        # print(cav_list, attacker_list)
        # exit()

        num += 1
        pbar.update(1)
    
    print(attackers)

    with open(save_path, 'wb') as file:
        pickle.dump(attackers, file)
    
    print(num)


if __name__ == '__main__':
    main()
