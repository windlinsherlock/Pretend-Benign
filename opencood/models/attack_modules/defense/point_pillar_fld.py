import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.attack_modules.base_module.co_attack_module import CoAttackModule, GPSAttack
from opencood.models.attack_modules import fusion_module
import time


class Predict(nn.Module):
    def __init__(self):
        super(Predict, self).__init__()
        self.cls_head = None
        self.reg_head = None
    
    def forward(self, dataset, fused_features, batch_data):
        psm = self.cls_head(fused_features)
        rm = self.reg_head(fused_features)

        output_dict = {}

        output_dict['ego'] = {  'psm': psm,                          # (B, anchor_num*num_class, H, W)
                                'rm': rm                             # (B, anchor_num*7 , H, W)
                            }
        
        pred_box, pred_score, gt_box = dataset.post_process(batch_data, output_dict)

        return pred_box, pred_score, gt_box


class FLD(nn.Module):
    def __init__(self):
        super(FLD, self).__init__()
        
        self.predict = None
        self.cls_head = None
        self.fusion_module = None

    def forward(self, spatial_features_2d, dataset, num_cav, batch_data, attacker_list, salient_threshold_I, salient_threshold_U, score_thresh):
        # ego_features = spatial_features_2d[0].unsqueeze(0)
        # ego_box, ego_score, gt_box = self.predict(dataset, ego_features, batch_data)
        # print(gt_box.shape)

        # print(attacker_list)

        all_agent_list = [i for i in range(num_cav)]
        collab_list = [i for i in range(1, num_cav) if i not in attacker_list]

        # spatial_features_2d: (num_cav, C, H, W)
        psm = self.cls_head(spatial_features_2d)
        cls_preds = psm.permute(0, 2, 3, 1).contiguous()                                                # (num_cav, H, W, anchor_num * num_class)
        cls_preds = cls_preds.view(cls_preds.shape[0], cls_preds.shape[1], cls_preds.shape[2], 2, 2)    # (num_cav, H, W, anchor_num, num_class)
        heat_maps = cls_preds[:,:,:,:,1]                                                                # (num_cav, H, W, anchor_num)
        heat_maps = torch.max(heat_maps, dim=-1)[0]
        heat_maps = torch.sigmoid(heat_maps)
        masked_maps = heat_maps > score_thresh                                                                  # (num_cav, H, W)
        
        masked_maps_ego = masked_maps[0].unsqueeze(0).repeat(num_cav, 1, 1)

        masked_maps_I = masked_maps * masked_maps_ego
        masked_maps_U = masked_maps | masked_maps_ego

        masked_num_I = torch.sum(masked_maps_I, dim=(1, 2))                                         # (num_cav)
        masked_num_U = torch.sum(masked_maps_U, dim=(1, 2))
        
        # print(masked_maps.shape)
        # print(masked_num_I)
        # print(masked_num_U)
        # exit()

        mask_features_2d = spatial_features_2d.permute(0, 2, 3, 1).contiguous()                  # (num_cav, H, W, C)

        masked_maps_I = masked_maps_I.unsqueeze(-1)                                                 # (num_cav, H, W, 1)

        masked_features_I = mask_features_2d * masked_maps_I                                         # (num_cav, H, W, C)

        masked_features_I_ego = masked_features_I[0].unsqueeze(0).repeat(num_cav, 1, 1, 1)              # (num_cav, H, W, C)

        # compute numerator: dot product
        dot_product = torch.sum(masked_features_I * masked_features_I_ego, dim=-1)                      # (num_cav, H, W)
        # compute denominator: L2 norm
        norm_features = torch.norm(masked_features_I, dim=-1)                                           # (num_cav, H, W)
        norm_features_ego = torch.norm(masked_features_I_ego, dim=-1)
        # avoid division by zero
        epsilon = 1e-8
        cosine_similarity = dot_product / (norm_features * norm_features_ego + epsilon)                 # (num_cav, H, W)

        similarity_I = torch.sum(cosine_similarity, dim=(1, 2))                                         # (num_cav,)
        similarity_I = similarity_I / masked_num_I

        match_IoU = masked_num_I / masked_num_U.float()                                                 # (num_cav,)

        if torch.isnan(similarity_I).any():
            similarity_I = torch.nan_to_num(similarity_I, nan=0.0)
        # print(similarity_I , match_IoU)

        collab_mask = (similarity_I > salient_threshold_I) & (match_IoU > salient_threshold_U)
        attacker_mask = ~collab_mask
        
        # print(collab_mask, attacker_mask)

        estimated_collab = torch.where(collab_mask)[0]
        estimated_attacker = torch.where(attacker_mask)[0]
        
        # print(collab_list, attacker_list)
        # print(estimated_collab, estimated_attacker)
        
        collab_features = spatial_features_2d[estimated_collab]
        fused_features = self.fusion_module(collab_features)

        collab_box, collab_score, gt_box = self.predict(dataset, fused_features, batch_data)

        succ_result = {
            'pred_box': collab_box,
            'pred_score': collab_score,
            'gt_box': gt_box
        }
        
        estimated_collab = estimated_collab[1:]

        collab_sim = similarity_I[collab_list].cpu().tolist()
        attacker_sim = similarity_I[attacker_list].cpu().tolist()

        collab_match = match_IoU[collab_list].cpu().tolist()
        attacker_match = match_IoU[attacker_list].cpu().tolist()
        
        # print(estimated_collab, collab_sim, attacker_sim, collab_match, attacker_match)

        return estimated_collab, succ_result, collab_sim, attacker_sim, collab_match, attacker_match
        




class PointPillarFLD(nn.Module):
    def __init__(self, args):
        super(PointPillarFLD, self).__init__()
        
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])

        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], self.scatter.num_bev_features)

        self.batch_norm = nn.BatchNorm2d(self.backbone.num_bev_features)

        self.co_attack = CoAttackModule()
        self.GPSAttack = GPSAttack()

        self.fusion_module = fusion_module.__all__[args['fusion_module']](
            feature_dim = self.backbone.num_bev_features
        )

        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'] * args['num_class'],
                                  kernel_size=1)

        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_num'],
                                  kernel_size=1)
        
        self.predict = Predict()
        self.FLD = FLD()

    
    def forward(self, batch_data, pert, attacker_list, no_fuse, dataset, salient_threshold_I, salient_threshold_U, score_thresh, LCF_mask=None, GPS_Attack=False):
        
        data_dict = batch_data['ego']

        voxel_features = data_dict['processed_lidar']['voxel_features']         # (sum(num_cav) * num_voxels, max_points_per_voxel, 4)
        voxel_coords = data_dict['processed_lidar']['voxel_coords']             # (sum(num_cav) * num_voxels, 4)
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']     # (sum(num_cav) * num_voxels, )
        record_len = data_dict['record_len']                                    # (B, )
        batch_size = record_len.shape[0]
        assert batch_size == 1, "batch_size must be 1 in attack inference"

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)        
        
        batch_dict = self.scatter(batch_dict)

        batch_dict = self.backbone(batch_dict)                                  

        spatial_features_2d = batch_dict['spatial_features_2d']                 # (num_cav, C, H, W)
        
        spatial_features_2d = self.batch_norm(spatial_features_2d)

        if LCF_mask is not None:
            H_indices, W_indices = LCF_mask.nonzero(as_tuple=True)              # H_indices and W_indices are one-dimensional indices
            spatial_features_2d_new = spatial_features_2d.clone()
            for attacker_index in attacker_list:
                spatial_features_2d_new[attacker_index, :, H_indices, W_indices] = spatial_features_2d[0, :, H_indices, W_indices]
            spatial_features_2d = spatial_features_2d_new
        
        if GPS_Attack:
            spatial_features_2d = self.GPSAttack(spatial_features_2d, attacker_list)
        else:
            spatial_features_2d = self.co_attack(spatial_features_2d, pert, attacker_list)

        num_cav = spatial_features_2d.shape[0]

        self.predict.cls_head = self.cls_head
        self.predict.reg_head = self.reg_head
        self.FLD.predict = self.predict
        self.FLD.cls_head = self.cls_head
        self.FLD.fusion_module = self.fusion_module
        
        # start time
        start_time = time.perf_counter()
        
        estimated_collab, succ_result, collab_sim, attacker_sim, collab_match, attacker_match = self.FLD(spatial_features_2d, dataset, num_cav, 
                                                                batch_data, attacker_list, salient_threshold_I, salient_threshold_U, score_thresh)
        
        torch.cuda.synchronize()  # ensure all GPU tasks are complete
        end_time = time.perf_counter()
        time_cost = end_time - start_time
        print(f"Total inference time (CPU + GPU): {time_cost:.6f} seconds")

        return estimated_collab, succ_result, time_cost, collab_sim, attacker_sim, collab_match, attacker_match