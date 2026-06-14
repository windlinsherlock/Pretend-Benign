import torch
import torch.nn as nn
import random
import torchvision.transforms.functional as TF
import numpy as np

def random_flip_and_rotate(feature):
    C, H, W = feature.shape
    
    # Step 1: Flip along the H axis (horizontal flip)
    feature_flipped_h = feature.flip(1)
    
    # Step 2: Randomly rotate by -30° or 30° along the H axis
    rotation_angle = random.choice([-30, 30])
    feature_rotated = TF.rotate(feature_flipped_h.unsqueeze(0), rotation_angle, fill=0)  # Using 0 for padding
    feature_rotated = feature_rotated.squeeze(0)
    
    # Step 3: Random flip along W axis
    if random.random() > 0.5:
        feature_rotated = feature_rotated.flip(2)
    
    return feature_rotated


class GPSAttack(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, features_2d, attacker_list):
        """
        features_2d:  (num_cav, C, H, W)
        """
        assert 0 not in attacker_list, "ego cav must not be in attacker_list"
        num_cav = features_2d.shape[0]
        num = 0
        for i in range(num_cav):
            if i in attacker_list:
                features_2d[i] = random_flip_and_rotate(features_2d[i])

        return features_2d



class CoAttackModule(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, features_2d, pert, attacker_list):
        """
        features_2d:  (num_cav, C, H, W)
        """
        assert pert.shape[0] == len(attacker_list), "pert' shape must be equal to attacker_list' length"
        assert 0 not in attacker_list, "ego cav must not be in attacker_list"
        num_cav = features_2d.shape[0]
        num = 0
        for i in range(num_cav):
            if i in attacker_list:
                features_2d[i] = features_2d[i] + pert[num]
                num += 1

        return features_2d