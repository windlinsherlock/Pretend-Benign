# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
from collections import OrderedDict
import pickle

import numpy as np
import torch

from opencood.utils.common_utils import torch_tensor_to_numpy


def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']

    output_dict['ego'] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_intermediate_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return inference_early_fusion(batch_data, model, dataset)


def save_prediction_gt(pred_tensor, pred_score, gt_tensor, cav_pose, pcd, timestamp, save_path, save_pcd):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    pred_score = torch_tensor_to_numpy(pred_score)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)
    cav_pose = torch_tensor_to_numpy(cav_pose)

    if save_pcd:
        np.save(os.path.join(save_path, 'pcd.npy'), pcd_np)
        
    np.save(os.path.join(save_path, 'pred.npy'), pred_np)
    np.save(os.path.join(save_path, 'pred_score.npy'), pred_score)
    np.save(os.path.join(save_path, 'cav_pose.npy'), cav_pose)
    np.save(os.path.join(save_path, 'gt.npy'), gt_np)


def save_heat_maps(heat_maps, cav_pose, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    heat_maps = torch_tensor_to_numpy(heat_maps)
    cav_pose = torch_tensor_to_numpy(cav_pose)
        
    np.save(os.path.join(save_path, 'heat_maps.npy'), heat_maps)
    np.save(os.path.join(save_path, 'cav_pose.npy'), cav_pose)


def tensor_to_numpy(d):
    """
    Convert all leaf nodes of type torch.Tensor in the dictionary to NumPy arrays (on the CPU).
    """
    if isinstance(d, dict):
        return {k: tensor_to_numpy(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [tensor_to_numpy(v) for v in d]
    elif isinstance(d, tuple):
        return tuple(tensor_to_numpy(v) for v in d)
    elif isinstance(d, torch.Tensor):
        return d.cpu().detach().numpy()
    else:
        return d


def save_all_output(output_dict, cav_pose, save_path):

    cav_pose = torch_tensor_to_numpy(cav_pose)

    np.save(os.path.join(save_path, 'cav_pose.npy'), cav_pose)

    output_dict = tensor_to_numpy(output_dict)
    
    # save as a pkl file
    with open(os.path.join(save_path, 'all_output.pkl'), 'wb') as file:
    
        pickle.dump(output_dict, file)