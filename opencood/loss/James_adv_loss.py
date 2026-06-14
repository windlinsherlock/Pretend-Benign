import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.pcdet_utils.iou3d_nms import iou3d_nms_utils


class JamesAdvLoss(nn.Module):
    def __init__(self, args):
        super(JamesAdvLoss, self).__init__()
        self.alpha = 0.2
        self.gamma = 1

        self.neg_threshold = 0.7
        self.pos_threshold = 0.3

        self.num_class = args.get('num_class', 2)
    
    def forward(self, origin_output, output_dict, target_dict, anchor_box):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        origin_rm = origin_output['rm']                                                                 # (1, anchor_num * 7, H, W)
        origin_psm = origin_output['psm']                                                               # (1, anchor_num * num_class, H, W)  

        rm = output_dict['rm']                                                                          # (1, anchor_num * 7, H, W)
        psm = output_dict['psm']                                                                        # (1, anchor_num * num_class, H, W)  

        pos_equal_one = target_dict['pos_equal_one']                                                    # (1, H, W, anchor_num) 
        neg_equal_one = target_dict['neg_equal_one']                                                    # (1, H, W, anchor_num) 
        # print(pos_equal_one.shape, neg_equal_one.shape)
        # print(pos_equal_one.sum(), neg_equal_one.sum(), pos_equal_one.numel())

        origin_cls = origin_psm.permute(0, 2, 3, 1).contiguous()                                                        # (1, H, W, anchor_num * num_class)
        origin_cls = origin_cls.view(origin_cls.shape[0], origin_cls.shape[1], origin_cls.shape[2], 2, 2)               # (1, H, W, anchor_num, num_class)
        origin_cls = torch.sigmoid(origin_cls)
        cls_class_neg = origin_cls[..., 0]                                                                                # (1, H, W, anchor_num)
        cls_class_pos = origin_cls[..., 1]                                                                                # (1, H, W, anchor_num)

        pos_mask = cls_class_pos > self.pos_threshold
        neg_mask = cls_class_neg > self.neg_threshold
        # print(pos_mask.shape, neg_mask.shape)
        # print(pos_mask.sum(), neg_mask.sum(), pos_mask.numel())

        pos_mask = pos_mask * pos_equal_one
        neg_mask = neg_mask * neg_equal_one
        pos_mask = pos_mask.to(torch.bool)
        neg_mask = neg_mask.to(torch.bool)
        # print(pos_mask.shape, neg_mask.shape)
        # print(pos_mask.sum(), neg_mask.sum(), pos_mask.numel())
        # origin_cls_pos = origin_cls[pos_mask, 1]
        # origin_cls_neg = origin_cls[neg_mask, 0]

        cls_preds = psm.permute(0, 2, 3, 1).contiguous()                                                                # (1, H, W, anchor_num * num_class)
        cls_preds = cls_preds.view(cls_preds.shape[0], cls_preds.shape[1], cls_preds.shape[2], 2, 2)                    # (1, H, W, anchor_num, num_class)
        cls_preds = torch.sigmoid(cls_preds)
        cls_preds_pos = cls_preds[..., 1][pos_mask]  
        cls_preds_neg = cls_preds[..., 1][neg_mask] 
        # print(cls_preds_pos.shape, cls_preds_neg.shape)

        # (1, H*W*anchor_num, 7)    (x,y,z,h,w,l,r) convert to l, w, h for visualization 
        box3d_origin = self.delta_to_boxes3d(origin_rm, anchor_box)
        box3d_origin = box3d_origin[:, :, [0, 1, 2, 5, 4, 3, 6]]             # hwl -> lwh
        box3d_origin = box3d_origin.reshape(origin_cls.shape[0], origin_cls.shape[1], origin_cls.shape[2], 2, 7)        # (1, H, W, anchor_num, 7)
        box3d_origin_pos = box3d_origin[pos_mask]

        # (1, H*W*anchor_num, 7)    (x,y,z,h,w,l,r) convert to l, w, h for visualization
        box3d_new = self.delta_to_boxes3d(rm, anchor_box)
        box3d_new = box3d_new[:, :, [0, 1, 2, 5, 4, 3, 6]]             # hwl -> lwh
        box3d_new = box3d_new.reshape(origin_cls.shape[0], origin_cls.shape[1], origin_cls.shape[2], 2, 7)              # (1, H, W, anchor_num, 7)
        box3d_new_pos = box3d_new[pos_mask]
        
        box3d_iou = torch.zeros(
                        (box3d_origin_pos.shape[0]),
                        dtype=torch.float32,  
                        device=box3d_origin_pos.device  
                    )

        for i in range(box3d_origin_pos.shape[0]):
            boxes_a = box3d_origin_pos[i].reshape(1, -1)
            boxes_b = box3d_new_pos[i].reshape(1, -1)
            ious = iou3d_nms_utils.boxes_iou3d_gpu(boxes_a, boxes_b)
            box3d_iou[i] = ious[0, 0]
        # print(box3d_iou)

        epsilon = 1e-8  # prevent numerical errors
        # − log(1 − z′σu) · IoU(z′, z)
        adv_loss_pos = - torch.log(1 - cls_preds_pos + epsilon) * box3d_iou                                              # (N,)
        # − λ * (1 - z′σv)^γ * log(z′σv)
        adv_loss_neg = - torch.log(cls_preds_neg + epsilon) * self.alpha * torch.pow((1 - cls_preds_neg), self.gamma)
        
        James_adv_loss = adv_loss_pos.sum() + adv_loss_neg.sum()
        
        return James_adv_loss
    
    @staticmethod
    def delta_to_boxes3d(deltas, anchors, channel_swap=True):
        """
        Convert the output delta to 3d bbx.

        Parameters
        ----------
        deltas : torch.Tensor
            (N, W, L, 14)
        anchors : torch.Tensor
            (W, L, 2, 7) -> xyzhwlr
        channel_swap : bool
            Whether to swap the channel of deltas. It is only false when using
            FPV-RCNN

        Returns
        -------
        box3d : torch.Tensor
            (N, W*L*2, 7)
        """
        # batch size
        N = deltas.shape[0]
        if channel_swap:
            deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
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
        # hwl
        boxes3d[..., [3, 4, 5]] = torch.exp(
            deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        # yaw angle
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d
