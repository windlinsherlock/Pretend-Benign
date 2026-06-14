from common_utils import feature_wraps
import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F


if __name__ == "__main__":
    feature_sequence = torch.ones((4, 3, 1, 100, 352))
    history_pose_list = torch.tensor([[[ 48.8823, 139.5461,   1.3534],
         [ 49.1939, 139.5584,   1.6149],
         [ 49.5436, 139.5712,   1.7501]],

        [[ 48.8823, 139.5461,   1.3534],
         [ 49.1939, 139.5584,   1.6149],
         [ 49.5436, 139.5712,   1.7501]],

        [[ 50.8253, 139.6082,   1.6048],
         [ 51.3294, 139.6180,   1.3125],
         [ 51.8710, 139.6254,   1.0274]],

        [[ 50.8253, 139.6082,   1.6048],
         [ 51.3294, 139.6180,   1.3125],
         [ 51.8710, 139.6254,   1.0274]]])
    cur_pose_list = torch.tensor([[ 49.9321, 139.5841,   1.7835],
    
        [ 49.9321, 139.5841,   1.7835],

        [ 52.4258, 139.6302,   0.7217],

        [ 52.4258, 139.6302,   0.7217]])
    points_min = torch.tensor([-140.8, -40])
    voxel_size = 0.8
    print(feature_sequence.shape, history_pose_list.shape, cur_pose_list.shape)
    print(points_min)
    feature_sequence = feature_wraps(feature_sequence, history_pose_list, cur_pose_list, points_min, voxel_size)
    print(feature_sequence.shape)