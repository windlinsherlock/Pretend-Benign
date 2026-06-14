import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.pcdet_utils.iou3d_nms import iou3d_nms_utils


class PBAdvLoss(nn.Module):
    def __init__(self, args):
        super(PBAdvLoss, self).__init__()
        self.alpha = args.get('alpha', 1)
        self.thresh_score = args.get('thresh_score', 0.32)
        self.thresh_iou = args.get('thresh_iou', 0.5)

        self.num_class = args.get('num_class', 2)
    
    def forward(self, mask_dict, boundingbox_dict, output_dict, anchor_box):
        """
        Parameters
        ----------
        mask_dict : {
            'HC_mask': HC_mask,             (H, W) bool
            'LCT_mask': LCT_mask, 
            'LCF_mask': LCF_mask,
            'IT_mask': IT_mask,
            'IF_mask': IF_mask
        }
        boundingbox_dict : {
            'HC_boxes': HC_boxes,          (num_boxes, 9) (x,y,z,l,w,h,r,index_x,index_y)
            'LCT_boxes': LCT_boxes,
            'LCF_boxes': LCF_boxes,
            'IT_boxes': IT_boxes
        }
        output_dict: {
            'psm': psm,                    (1, anchor_num * num_class, H, W)
            'rm': rm                       (1, anchor_num*7 , H, W)
        }
        anchor_box: (H,W,2,7)  
        """
        
        rm = output_dict['rm']                                                                          # (1, anchor_num * 7, H, W)
        psm = output_dict['psm']                                                                        # (1, anchor_num * num_class, H, W) 
        # features = output_dict['features']                                                              # (num_attacker+1, C, H, W)
        # features = features.permute(0, 2, 3, 1).contiguous()                                            # (num_attacker+1, H, W, C)
        
        cls_preds = psm.permute(0, 2, 3, 1).contiguous()                                                   # (1, H, W, anchor_num * num_class)
        cls_preds = cls_preds.view(cls_preds.shape[0], cls_preds.shape[1], cls_preds.shape[2], 2, 2)       # (1, H, W, anchor_num, num_class)
        cls_preds = torch.sigmoid(cls_preds)
        pred_scores = cls_preds[0, ..., 1]                                                                 # (H, W, anchor_num)

        # (1, H*W*anchor_num, 7)    (x,y,z,h,w,l,r) convert to l, w, h for visualization
        pred_boxes = self.delta_to_boxes3d(rm, anchor_box)                                             # (1, H*W*2, 7)
        pred_boxes = pred_boxes[:, :, [0, 1, 2, 5, 4, 3, 6]]             # hwl -> lwh
        pred_boxes = pred_boxes.reshape(pred_scores.shape[0], pred_scores.shape[1], 2, 7)              # (H, W, anchor_num, 7)

        pred_scores, max_indices = torch.max(pred_scores, dim=-1)                                       # pred_scores: (H, W), max_indices: (H, W)
        # index the corresponding pred_boxes with max_indices
        selected_boxes = torch.gather(pred_boxes, dim=2, index=max_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 7))
        pred_boxes = selected_boxes.squeeze(2)                                                         # (H, W, 7)
        
        HC_mask = mask_dict['HC_mask']                                                                 # (H, W)
        HC_boxes = boundingbox_dict['HC_boxes']                                                        # (num_boxes, 9) (x,y,z,l,w,h,r,index_x,index_y)
        # H_indices, W_indices = HC_mask.nonzero(as_tuple=True)
        # HC_features = features[:, H_indices, W_indices]                                                # (num_attacker+1, N, C)
        # print(HC_features.shape)
        HC_loss = self.remove_adv_loss(HC_mask, HC_boxes, pred_scores, pred_boxes, penalty=True, region_all=False)
        # HC_loss = self.remove_adv_loss(HC_mask, HC_boxes, pred_scores, pred_boxes, penalty=False, region_all=True)
        
        LCT_mask = mask_dict['LCT_mask']                                                               # (H, W)
        LCT_boxes = boundingbox_dict['LCT_boxes']                                                      # (num_boxes, 9) (x,y,z,l,w,h,r,index_x,index_y)
        LCT_loss = self.remove_adv_loss(LCT_mask, LCT_boxes, pred_scores, pred_boxes, penalty=False, region_all=True)

        LCF_mask = mask_dict['LCF_mask']                                                               # (H, W)
        LCF_boxes = boundingbox_dict['LCF_boxes']                                                      # (num_boxes, 9) (x,y,z,l,w,h,r,index_x,index_y)
        # H_indices, W_indices = LCF_mask.nonzero(as_tuple=True)
        # LCF_features = features[:, H_indices, W_indices]                                                # (num_attacker+1, N, C)
        # print(LCF_features.shape)
        LCF_loss = self.add_adv_loss(LCF_mask, LCF_boxes, pred_scores, no_boxes=False)

        IT_mask = mask_dict['IT_mask']                                                               # (H, W)
        IT_boxes = boundingbox_dict['IT_boxes']                                                      # (num_boxes, 9) (x,y,z,l,w,h,r,index_x,index_y)
        IT_loss = self.remove_adv_loss(IT_mask, IT_boxes, pred_scores, pred_boxes, penalty=False, region_all=True)

        IF_mask = mask_dict['IF_mask']                                                               # (H, W)
        IF_loss = self.add_adv_loss(IF_mask, None, pred_scores, no_boxes=True)

        PB_adv_loss = HC_loss + LCT_loss + LCF_loss + IT_loss + IF_loss
        
        return PB_adv_loss


    def add_adv_loss(self, mask, boxes, pred_scores, no_boxes):
        # add_adv_loss = −log(z′_score)
        epsilon = 1e-8  
        if no_boxes:
            pred_scores = pred_scores[mask]                                                            # (num_heat, )
            # print(pred_scores)
            adv_loss = -torch.log(pred_scores + epsilon)                                               # (num_heat, )
            return adv_loss.sum()

        centers = boxes[:, 7:]
        centers = centers.to(dtype=torch.long)
        pred_scores = pred_scores[centers[:, 1], centers[:, 0]]                                             # (num_boxes, )
        # print(pred_scores)
        adv_loss = -torch.log(pred_scores + epsilon)                                                        # (num_boxes, )

        return adv_loss.sum()

    def remove_adv_loss(self, mask, boxes, pred_scores, pred_boxes, penalty, region_all):
        # prevent numerical errors
        epsilon = 1e-8  

        if region_all:
            pred_scores_heat = pred_scores[mask]                                                            # (num_heat,)
            # print(pred_scores_heat)
            heat_adv_loss = -torch.log(1 - pred_scores_heat + epsilon)                                      # (num_heat,)
        
        centers = boxes[:, 7:]
        centers = centers.to(dtype=torch.long)
        pred_scores = pred_scores[centers[:, 1], centers[:, 0]]                                                                     # (num_boxes, )
        pred_boxes = pred_boxes[centers[:, 1], centers[:, 0]]                                                                       # (num_boxes, 7)

        boxes_iou = torch.zeros(                                                                            # (num_boxes, )
                        (pred_boxes.shape[0]),
                        dtype=pred_boxes.dtype,  
                        device=pred_boxes.device  
                    )

        for i in range(pred_boxes.shape[0]):
            boxes_a = pred_boxes[i].reshape(1, -1)
            boxes_b = boxes[i, :7].reshape(1, -1)
            ious = iou3d_nms_utils.boxes_iou3d_gpu(boxes_a, boxes_b)
            boxes_iou[i] = ious[0, 0]

        # print(pred_scores)
        # print(boxes_iou)
        # remove_adv_loss = −log(1 − z′_score) · IoU(z′, z) + λ * penalty(z',z)
        '''
        penalty(z',z) = {
            -log(1 - (thresh_score-z'_score)^(1/2)), if z'_score < threshold 
            +
            -log(1 - (thresh_score-z'_score)^(1/2)), if IoU(z',z) < thresh_iou
        }
        '''
        boxes_adv_loss = -torch.log(1 - pred_scores + epsilon) * boxes_iou                                  # (num_boxes, )

        if penalty:
            penalty_score_mask = pred_scores < self.thresh_score
            penalty_score = pred_scores[penalty_score_mask]
            penalty_score_loss = -torch.log(1 - torch.pow(self.thresh_score - penalty_score, 1/2) + epsilon)

            penalty_iou_mask = boxes_iou < self.thresh_iou
            penalty_iou = boxes_iou[penalty_iou_mask]
            penalty_iou_loss = -torch.log(1 - torch.pow(self.thresh_iou - penalty_iou, 1/2) + epsilon)
        
        adv_loss = boxes_adv_loss.sum()

        if region_all:
            # print(adv_loss, heat_adv_loss.sum())
            adv_loss = adv_loss + heat_adv_loss.sum()

        if penalty:
            # print(adv_loss, self.alpha * (penalty_score_loss.sum() + penalty_iou_loss.sum()))
            adv_loss = adv_loss +  self.alpha * (penalty_score_loss.sum() + penalty_iou_loss.sum())

        return adv_loss


    @staticmethod
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
        if torch.max(input_values) > 20:
            input_values = torch.clamp(input_values, max=20)

        # hwl
        boxes3d[..., [3, 4, 5]] = torch.exp(
            input_values) * anchors_reshaped[..., [3, 4, 5]]
        # yaw angle
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d
