import numpy as np
import torch
from opencood.pcdet_utils.iou3d_nms import iou3d_nms_utils
import math


def tensor_to_numpy(d):
    """
    Convert all torch.Tensor leaf nodes in a dictionary to NumPy arrays on CPU.
    """
    if isinstance(d, dict):
        return {k: tensor_to_numpy(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [tensor_to_numpy(v) for v in d]
    elif isinstance(d, tuple):
        return tuple(tensor_to_numpy(v) for v in d)
    elif isinstance(d, torch.Tensor):
        return d.cpu().detach().numpy()  # convert Tensor to a NumPy array
    else:
        return d  # return the original value if it is not a Tensor


def get_IF_mask(heat_map_size, ready_mask, IF_num, device):
    IF_mask = torch.zeros(heat_map_size, dtype=torch.bool, device=device)
    valid_positions = (ready_mask == False)
    valid_indices = torch.nonzero(valid_positions, as_tuple=True)
    random_indices = torch.randint(0, valid_indices[0].size(0), (IF_num,))
    IF_mask[valid_indices[0][random_indices], valid_indices[1][random_indices]] = True
    return IF_mask

    

def select_attack_region(all_output, heat_map_size, pos_thresh_high, pos_thresh_mid, pos_thresh_low, fusion_thresh,
                                    iou_thresh, cav_pose, attacker_list, points_min, voxel_size, IF_num, device, remove_mid_ratio):
    cav_pose = cav_pose[0]
    
    fusion_boxes = all_output['fusion']['pred_box']
    fusion_score = all_output['fusion']['pred_score']
    mask = fusion_score > pos_thresh_low
    fusion_boxes = fusion_boxes[mask]
    fusion_score = fusion_score[mask]
    
    ego_boxes = all_output['0']['pred_box']
    ego_score = all_output['0']['pred_score']
    mask = ego_score > pos_thresh_low
    ego_boxes = ego_boxes[mask]
    ego_score = ego_score[mask]

    attackers_boxes = []
    attackers_score = []
    
    for attacker_idx in attacker_list:
        attacker_boxes = all_output[str(attacker_idx)]['pred_box']
        attacker_score = all_output[str(attacker_idx)]['pred_score']
        mask = attacker_score > pos_thresh_low
        attacker_boxes = attacker_boxes[mask]
        attacker_score = attacker_score[mask]
        attackers_boxes.append(attacker_boxes)
        attackers_score.append(attacker_score)
    

    HC_mask, HC_boxes = select_HC_region(ego_boxes, ego_score, heat_map_size, pos_thresh_high, 
                                                        points_min, voxel_size, device)

    LCT_mask, LCT_boxes, LCF_mask, LCF_boxes = select_LC_region(ego_boxes, ego_score, fusion_boxes, fusion_score, heat_map_size, 
                        pos_thresh_high, pos_thresh_mid, pos_thresh_low, fusion_thresh, iou_thresh, points_min, voxel_size, device, remove_mid_ratio)
    
    IT_mask, IT_boxes = select_IT_region(ego_boxes, ego_score, fusion_boxes, fusion_score, attackers_boxes, attackers_score, 
                                         heat_map_size, pos_thresh_high, pos_thresh_mid, pos_thresh_low, iou_thresh, points_min, voxel_size, device)

    ready_mask = HC_mask | LCT_mask | LCF_mask | IT_mask
    
    # IF_mask = get_IF_mask(heat_map_size, ready_mask, IF_num, device)
    
    IF_mask = select_IF_region(ready_mask, heat_map_size, cav_pose, attacker_list, points_min, voxel_size, IF_num, device)
    
    mask_dict = {
        'HC_mask': HC_mask,
        'LCT_mask': LCT_mask, 
        'LCF_mask': LCF_mask,
        'IT_mask': IT_mask,
        'IF_mask': IF_mask
    }
    boundingbox_dict = {
        'HC_boxes': HC_boxes,
        'LCT_boxes': LCT_boxes,
        'LCF_boxes': LCF_boxes,
        'IT_boxes': IT_boxes
    }
    
    return mask_dict, boundingbox_dict


def painting_boxes(boxes, heat_map_size, points_min, voxel_size, device, large=False):
    center = boxes[:, :2]                                   # (x, y)
    center = center - points_min
    center = (center / voxel_size).to(torch.long)

    w = boxes[:, 3].clone()
    h = boxes[:, 4].clone()
    r = boxes[:, 6]

    mask = torch.abs(r) > 1
    w[mask], h[mask] = h[mask], w[mask]
    w = (w / voxel_size).to(torch.long)
    h = (h / voxel_size).to(torch.long)

    Heat_mask = torch.zeros(heat_map_size, dtype=torch.bool, device=device)
    Heat_mask[center[:, 1], center[:, 0]] = True

    # determine box positions
    for i, pos in enumerate(center):
        x, y = pos                                              # note that y is the row and x is the column
        width, height = w[i], h[i]
        if large:
            width = torch.ceil((width - 1) / 2).to(torch.long)
            height = torch.ceil((height - 1) / 2).to(torch.long)         
        else:
            width = torch.div(width-1, 2, rounding_mode='trunc')
            height = torch.div(height-1, 2, rounding_mode='trunc')
        for dy in range(y-height, y+height+1):
            for dx in range(x-width, x+width+1):
                Heat_mask[dy, dx] = True
    
    boxes = torch.cat((boxes, center.to(boxes.dtype)), dim=-1)         # （num_boxes, 9) -> (x,y,z,l,w,h, x_index, y_index)

    return Heat_mask, boxes


def select_HC_region(ego_boxes, ego_score, heat_map_size, pos_thresh_high, points_min, voxel_size, device):
    
    mask = ego_score > pos_thresh_high
    ego_boxes = ego_boxes[mask]
    ego_score = ego_score[mask]
            
    HC_mask, HC_boxes = painting_boxes(ego_boxes, heat_map_size, points_min, voxel_size, device)

    return HC_mask, HC_boxes

def select_LC_region(ego_boxes, ego_score, fusion_boxes, fusion_score, heat_map_size, 
        pos_thresh_high, pos_thresh_mid, pos_thresh_low, fusion_thresh, iou_thresh, points_min, voxel_size, device, remove_mid_ratio):
    
    mask = fusion_score > fusion_thresh
    fusion_boxes = fusion_boxes[mask]
    fusion_score = fusion_score[mask]
    
    mask = (ego_score > pos_thresh_low) & (ego_score < pos_thresh_high)
    ego_boxes = ego_boxes[mask]
    ego_score = ego_score[mask]
    
    ious = iou3d_nms_utils.boxes_iou3d_gpu(ego_boxes, fusion_boxes)

    ious, _ = torch.max(ious, dim=-1)
    mask = ious < iou_thresh
    LCF_boxes = ego_boxes[mask]
    LCF_score = ego_score[mask]

    mask = ~mask
    low_mask = mask & (ego_score < pos_thresh_mid)
    mid_mask = mask & (ego_score > pos_thresh_mid) & (ego_score < pos_thresh_high)
    
    low_boxes = ego_boxes[low_mask]
    low_score = ego_score[low_mask]
    mid_boxes = ego_boxes[mid_mask]
    mid_score = ego_score[mid_mask]
        
    remove_mid_num = math.ceil(remove_mid_ratio * mid_boxes.shape[0])
    random_indices = torch.randperm(mid_boxes.shape[0])
    selected_indices = random_indices[:remove_mid_num]  
    mid_boxes = mid_boxes[selected_indices]        
    mid_score = mid_score[selected_indices]  
    
    LCT_boxes = torch.cat((low_boxes, mid_boxes), axis=0)
    LCT_score = torch.cat((low_score, mid_score), axis=0)     
         
    LCT_mask, LCT_boxes = painting_boxes(LCT_boxes, heat_map_size, points_min, voxel_size, device, large=True)
    
    LCF_mask, LCF_boxes = painting_boxes(LCF_boxes, heat_map_size, points_min, voxel_size, device)

    return LCT_mask, LCT_boxes, LCF_mask, LCF_boxes

def select_IT_region(ego_boxes, ego_score, fusion_boxes, fusion_score, attackers_boxes, attackers_score, 
                    heat_map_size, pos_thresh_high, pos_thresh_mid, pos_thresh_low, iou_thresh, points_min, voxel_size, device):
    ious = iou3d_nms_utils.boxes_iou3d_gpu(fusion_boxes, ego_boxes)
    ious, _ = torch.max(ious, dim=-1)
    mask = (ious < iou_thresh) & (fusion_score > pos_thresh_mid)
    fusion_boxes = fusion_boxes[mask]
    fusion_score = fusion_score[mask]
    
    for attacker_boxes, attacker_score in zip(attackers_boxes, attackers_score):
        mask = attacker_score > pos_thresh_mid
        attacker_boxes = attacker_boxes[mask]
        attacker_score = attacker_score[mask]

        if attacker_boxes.shape[0] != 0 and ego_boxes.shape[0] != 0:
            ious = iou3d_nms_utils.boxes_iou3d_gpu(attacker_boxes, ego_boxes)
            ious, _ = torch.max(ious, dim=-1)
            mask = ious < iou_thresh
            attacker_boxes = attacker_boxes[mask]
            attacker_score = attacker_score[mask]

        if attacker_boxes.shape[0] != 0 and fusion_boxes.shape[0] != 0:
            ious = iou3d_nms_utils.boxes_iou3d_gpu(attacker_boxes, fusion_boxes)
            ious, _ = torch.max(ious, dim=-1)
            mask = ious < iou_thresh
            attacker_boxes = attacker_boxes[mask]
            attacker_score = attacker_score[mask]

        fusion_boxes = torch.cat((fusion_boxes, attacker_boxes), dim=0)
        fusion_score = torch.cat((fusion_score, attacker_score), dim=0)
    
    IT_mask, IT_boxes = painting_boxes(fusion_boxes, heat_map_size, points_min, voxel_size, device, large=True)

    return IT_mask, IT_boxes

def select_IF_region(ready_mask, heat_map_size, cav_pose, attacker_list, points_min, voxel_size, IF_num, device):

    cav_num = cav_pose.shape[0]
    cav_pose = cav_pose[:, [1, 0]]
    cav_pose[:, 1] = cav_pose[:, 1] * -1                    # (x,y)
    cav_pose = cav_pose - points_min
    cav_pose = (cav_pose / voxel_size).to(torch.long)

    ego_pose = cav_pose[0]
    attackers_pose = cav_pose[attacker_list]

    far_mask = attackers_pose[:, 0] > ego_pose[0] + 5
    far_x = attackers_pose[far_mask, 0]
    near_mask = attackers_pose[:, 0] < ego_pose[0] - 5
    near_x = attackers_pose[near_mask, 0]

    if far_x.shape[0] > 0:
        min_far_x = min(torch.min(far_x) + 30, torch.max(far_x))
    else:
        min_far_x = None

    if near_x.shape[0] > 0:
        max_near_x = max(torch.max(near_x) - 30, torch.min(near_x))
    else:
        max_near_x = None

    y_indices, x_indices = torch.meshgrid(
        torch.arange(heat_map_size[0], dtype=torch.long),
        torch.arange(heat_map_size[1], dtype=torch.long),
        indexing="ij",
    )

    index_array = torch.stack((y_indices, x_indices), dim=-1)
    index_array = index_array.to(device)
    mask = torch.full(heat_map_size, False, dtype=torch.bool, device=device)

    if min_far_x is not None:
        mask = mask | ((index_array[:, :, 1] >= min_far_x) & (index_array[:, :, 1] <= min_far_x + 50) & ~ready_mask)

    if max_near_x is not None:
        mask = mask | ((index_array[:, :, 1] <= max_near_x) & (index_array[:, :, 1] >= max_near_x - 50) & ~ready_mask)

    index_array = index_array[mask]
    selected_indices = index_array[torch.randperm(len(index_array))[:IF_num]]
    
    heat_maps = torch.zeros(heat_map_size, dtype=torch.bool, device=device)
    heat_maps[selected_indices[:, 0], selected_indices[:, 1]] = True

    IF_mask = heat_maps
    return IF_mask


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
          
              
def get_hc_score_iou(output_dict, mask_dict, boundingbox_dict, anchor_box):
    
    HC_mask = mask_dict['HC_mask']
    HC_boxes = boundingbox_dict['HC_boxes']
    
    rm = output_dict['rm']                                                                              # (1, anchor_num * 7, H, W)
    psm = output_dict['psm']                                                                            # (1, anchor_num * num_class, H, W) 
    
    cls_preds = psm.permute(0, 2, 3, 1).contiguous()                                                    # (1, H, W, anchor_num * num_class)
    cls_preds = cls_preds.view(cls_preds.shape[0], cls_preds.shape[1], cls_preds.shape[2], 2, 2)        # (1, H, W, anchor_num, num_class)
    cls_preds = torch.sigmoid(cls_preds)
    pred_scores = cls_preds[0, ..., 1]                                                                  # (H, W, anchor_num)
    
    pred_boxes = delta_to_boxes3d(rm, anchor_box)                                                       # (1, H*W*2, 7)
    pred_boxes = pred_boxes[:, :, [0, 1, 2, 5, 4, 3, 6]]             # hwl -> lwh
    pred_boxes = pred_boxes.reshape(pred_scores.shape[0], pred_scores.shape[1], 2, 7)                   # (H, W, anchor_num, 7)
        
    pred_scores, max_indices = torch.max(pred_scores, dim=-1)                                           # pred_scores: (H, W), max_indices: (H, W)
    # index the corresponding pred_boxes with max_indices
    selected_boxes = torch.gather(pred_boxes, dim=2, index=max_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 7))
    pred_boxes = selected_boxes.squeeze(2)                                                         # (H, W, 7)
    
    centers = HC_boxes[:, 7:]
    centers = centers.to(dtype=torch.long)
    pred_scores = pred_scores[centers[:, 1], centers[:, 0]]                                             # (num_boxes, )
    pred_boxes = pred_boxes[centers[:, 1], centers[:, 0]]                                               # (num_boxes, 7)
    
    boxes_iou = torch.zeros(                                                                            # (num_boxes, )
                (pred_boxes.shape[0]),
                dtype=pred_boxes.dtype,  
                device=pred_boxes.device  
            )

    for i in range(pred_boxes.shape[0]):
        boxes_a = pred_boxes[i].reshape(1, -1)
        boxes_b = HC_boxes[i, :7].reshape(1, -1)
        ious = iou3d_nms_utils.boxes_iou3d_gpu(boxes_a, boxes_b)
        boxes_iou[i] = ious[0, 0]
    
    return pred_scores, boxes_iou


def get_ict_remove_ratio(pred_boxes, boundingbox_dict):
    LCT_boxes = boundingbox_dict['LCT_boxes']
    
    ious = iou3d_nms_utils.boxes_iou3d_gpu(LCT_boxes[:, :7], pred_boxes)
    ious, _ = torch.max(ious, dim=-1)
    remaining_num = (ious > 0.2).sum()
    remove_num = LCT_boxes.shape[0] - remaining_num
    
    return remove_num, LCT_boxes.shape[0]


def get_icf_score(output_dict, mask_dict, boundingbox_dict):
    LCF_mask = mask_dict['LCF_mask']
    LCF_boxes = boundingbox_dict['LCF_boxes']
    
    psm = output_dict['psm']                                                                            # (1, anchor_num * num_class, H, W) 
    
    cls_preds = psm.permute(0, 2, 3, 1).contiguous()                                                    # (1, H, W, anchor_num * num_class)
    cls_preds = cls_preds.view(cls_preds.shape[0], cls_preds.shape[1], cls_preds.shape[2], 2, 2)        # (1, H, W, anchor_num, num_class)
    cls_preds = torch.sigmoid(cls_preds)
    pred_scores = cls_preds[0, ..., 1]                                                                  # (H, W, anchor_num)
        
    pred_scores, _ = torch.max(pred_scores, dim=-1)                                                     # pred_scores: (H, W)
    
    centers = LCF_boxes[:, 7:]
    centers = centers.to(dtype=torch.long)
    pred_scores = pred_scores[centers[:, 1], centers[:, 0]]                                             # (num_boxes, )

    return pred_scores
    

def get_it_remove_ratio(pred_boxes, boundingbox_dict):
    IT_boxes = boundingbox_dict['IT_boxes']
    
    ious = iou3d_nms_utils.boxes_iou3d_gpu(IT_boxes[:, :7], pred_boxes)
    ious, _ = torch.max(ious, dim=-1)
    remaining_num = (ious > 0.2).sum()
    remove_num = IT_boxes.shape[0] - remaining_num
    
    return remove_num, IT_boxes.shape[0]
    
    
def delta_to_boxes3d(deltas, anchors, channel_swap=True):
    """
    Convert the output delta to 3d bbx.

    Parameters
    ----------
    deltas : torch.Tensor
        (N, 14, H, W)
    anchors : torch.Tensor
        (H, W, 2, 7) -> xyzhwlr
    channel_swap : bool
        Whether to swap the channel of deltas. It is only false when using
        FPV-RCNN

    Returns
    -------
    box3d : torch.Tensor
        (N, H*W*2, 7)
    """
    # batch size
    N = deltas.shape[0]
    if channel_swap:
        deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)             # (N, H*W*2, 7)
    else:
        deltas = deltas.contiguous().view(N, -1, 7)

    boxes3d = torch.zeros_like(deltas)
    if deltas.is_cuda:
        anchors = anchors.cuda()
        boxes3d = boxes3d.cuda()

    # (W*L*2, 7)
    anchors_reshaped = anchors.view(-1, 7).float()
    # the diagonal of the anchor 2d box, (W*L*2)
    anchors_d = torch.sqrt(
        anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
    anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
    anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

    # Inv-normalize to get xyz
    boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + \
                            anchors_reshaped[..., [0, 1]]
    boxes3d[..., [2]] = torch.mul(deltas[..., [2]],
                                    anchors_reshaped[..., [3]]) + \
                        anchors_reshaped[..., [2]]
                        
    input_values = deltas[..., [3, 4, 5]]

    # hwl
    boxes3d[..., [3, 4, 5]] = torch.exp(
        input_values) * anchors_reshaped[..., [3, 4, 5]]
    # yaw angle
    boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

    return boxes3d
