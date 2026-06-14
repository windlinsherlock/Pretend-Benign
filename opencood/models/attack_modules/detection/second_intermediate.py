import torch
import torch.nn as nn


from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.attack_modules import fusion_module


class SecondIntermediate(nn.Module):
    def __init__(self, args):
        super(SecondIntermediate, self).__init__()

        self.batch_size = args['batch_size']
        # mean_vfe
        self.mean_vfe = MeanVFE(args['mean_vfe'], 4)
        # sparse 3d backbone
        self.backbone_3d = VoxelBackBone8x(args['backbone_3d'],
                                           4, args['grid_size'])
        # height compression
        self.height_compression = HeightCompression(args['height_compression'])
        # base ben backbone
        self.backbone_2d = BaseBEVBackbone(args['base_bev_backbone'], 256)

        self.batch_norm = nn.BatchNorm2d(self.backbone_2d.num_bev_features)

        self.fusion_module = fusion_module.__all__[args['fusion_module']](
            feature_dim = self.backbone_2d.num_bev_features
        )
        # head
        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'] * args['num_class'],
                                  kernel_size=1)

        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_num'],
                                  kernel_size=1)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, batch_data, dataset, need_box):
        data_dict = batch_data['ego']
        
        voxel_features = data_dict['processed_lidar']['voxel_features']         # (sum(num_cav) * num_voxels, max_points_per_voxel, 4)
        voxel_coords = data_dict['processed_lidar']['voxel_coords']             # (sum(num_cav) * num_voxels, 4)
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']     # (sum(num_cav) * num_voxels, )
        record_len = data_dict['record_len']                                    # (B, )

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'batch_size': torch.sum(record_len).cpu().numpy(),
                      'record_len': record_len}

        batch_dict = self.mean_vfe(batch_dict)              # 'voxel_features': (sum(num_cav) * num_voxels, 4)
                
        batch_dict = self.backbone_3d(batch_dict)           # 'encoded_spconv_tensor': (sum(num_cav), C, nz, ny, nx)
        
        batch_dict = self.height_compression(batch_dict)    # 'spatial_features': (sum(num_cav), C*nz, ny, nx)
                
        batch_dict = self.backbone_2d(batch_dict)           # spatial_features_2d:(sum(num_cav), C , H, W)
                
        spatial_features_2d = batch_dict['spatial_features_2d']

        spatial_features_2d = self.batch_norm(spatial_features_2d)

        # split_x: [(num_cav1,C,H,W), (num_cav2,C,H,W), ...,(num_cav n,C,H,W)]
        split_x = self.regroup(spatial_features_2d, batch_dict['record_len'])
        out = []

        for xx in split_x:
            h = self.fusion_module(xx)
            out.append(h)

        spatial_features_2d = torch.cat(out, dim=0)             # (B,C,W,H)

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)
        
        if need_box:
            output_dict = {}
            output_dict['ego'] = {  'psm': psm,                          # (B, anchor_num * num_class , H, W)
                                'rm': rm                             # (B, anchor_num*7 , H, W)
                            }
            pred_box, pred_score, gt_box = dataset.post_process(batch_data, output_dict)
            
            return pred_box, pred_score, gt_box
        else:
            output_dict = {'psm': psm,                          # (B, anchor_num * num_class , H, W)
                        'rm': rm                             # (B, anchor_num*7 , H, W)
                        }

        return output_dict