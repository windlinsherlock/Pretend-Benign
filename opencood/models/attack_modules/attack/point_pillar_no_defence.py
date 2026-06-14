import torch
import torch.nn as nn


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.attack_modules.base_module.co_attack_module import CoAttackModule
from opencood.models.attack_modules import fusion_module
import time


class Predict(nn.Module):
    def __init__(self):
        super(Predict, self).__init__()
        self.cls_head = None
        self.reg_head = None
        self.fusion_module = None
    
    def forward(self, dataset, spatial_features_2d, batch_data, no_fuse):
        if no_fuse:
            fused_features = spatial_features_2d[0].unsqueeze(0)                # (1, C, H, W)
        else:
            fused_features = self.fusion_module(spatial_features_2d)

        psm = self.cls_head(fused_features)
        rm = self.reg_head(fused_features)

        output_dict = {}

        output_dict['ego'] = {  'psm': psm,                          # (B, anchor_num*2 , H, W)
                                'rm': rm                             # (B, anchor_num*7 , H, W)
                            }
        
        pred_box, pred_score, gt_box = dataset.post_process(batch_data, output_dict)

        return pred_box, pred_score, gt_box


class PointPillarNoDefence(nn.Module):
    def __init__(self, args):
        super(PointPillarNoDefence, self).__init__()
        
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])

        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], self.scatter.num_bev_features)

        self.batch_norm = nn.BatchNorm2d(self.backbone.num_bev_features)

        self.co_attack = CoAttackModule()

        self.fusion_module = fusion_module.__all__[args['fusion_module']](
            feature_dim = self.backbone.num_bev_features
        )

        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'] * args['num_class'],
                                  kernel_size=1)

        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_num'],
                                  kernel_size=1)
        self.predict = Predict()


    def forward(self, batch_data, pert, attacker_list, no_fuse, dataset, attack, need_box, need_heat_map=False, upper_bound=False):
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

        spatial_features_2d = batch_dict['spatial_features_2d']                 # (sum(num_cav), C, H, W)
        
        spatial_features_2d = self.batch_norm(spatial_features_2d)
        
        # print(spatial_features_2d.shape, spatial_features_2d.max().item(), spatial_features_2d.min().item(), spatial_features_2d.mean().item(), 
        #                 spatial_features_2d.abs().mean().item(), spatial_features_2d.var().item())
        # exit()

        if attack:
            spatial_features_2d = self.co_attack(spatial_features_2d, pert, attacker_list)         # (sum(num_cav), C, H, W)
            if upper_bound:
                num_cav = spatial_features_2d.shape[0]
                collab_list = [i for i in range(0, num_cav) if i not in attacker_list]
                spatial_features_2d = spatial_features_2d[collab_list]
        
        if need_box:
            self.predict.cls_head = self.cls_head
            self.predict.reg_head = self.reg_head
            self.predict.fusion_module = self.fusion_module

            # start time
            start_time = time.perf_counter()

            pred_box, pred_score, gt_box = self.predict(dataset, spatial_features_2d, batch_data, no_fuse)

            torch.cuda.synchronize()  # ensure all GPU tasks are complete
            end_time = time.perf_counter()
            time_cost = end_time - start_time
            print(f"Total inference time (CPU + GPU): {time_cost:.6f} seconds")
            
            return pred_box, pred_score, gt_box, time_cost
        
        elif need_heat_map:
            single_heat_map = self.cls_head(spatial_features_2d)                    # (num_cav, anchor_num * num_class, H, W)

            fused_features = self.fusion_module(spatial_features_2d)
            fusion_heat_map = self.cls_head(fused_features)                         # (1, anchor_num * num_class, H, W)

            heat_maps = torch.cat([single_heat_map, fusion_heat_map], dim=0)        # (num_cav+1, anchor_num * num_class, H, W)

            cls_preds = heat_maps.permute(0, 2, 3, 1).contiguous()                                          # (num_cav+1, H, W, anchor_num * num_class)
            cls_preds = cls_preds.view(cls_preds.shape[0], cls_preds.shape[1], cls_preds.shape[2], 2, 2)    # (num_cav+1, H, W, anchor_num, num_class)
            heat_maps = cls_preds[:,:,:,:,1]                                                                # (num_cav+1, H, W, anchor_num)
            heat_maps = torch.max(heat_maps, dim=-1)[0]
            heat_maps = torch.sigmoid(heat_maps)

            return heat_maps
        
        else:
            if no_fuse:
                fused_features = spatial_features_2d[0].unsqueeze(0)                # (1, C, H, W)
            else:
                fused_features = self.fusion_module(spatial_features_2d)

            psm = self.cls_head(fused_features)
            rm = self.reg_head(fused_features)

            output_dict = {'psm': psm,                          # (1, anchor_num * num_class, H, W)
                        'rm': rm                                # (1, anchor_num*7 , H, W)
                        }

            return output_dict