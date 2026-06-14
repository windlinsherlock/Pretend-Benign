import torch
import torch.nn as nn


class MeanFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        
    def forward(self, features_2d):
        '''
        features_2d: (num_cav,C,H,W)
        '''
        features_2d = torch.mean(features_2d, dim=0, keepdim=True)  # (1,C,H,W)

        return features_2d