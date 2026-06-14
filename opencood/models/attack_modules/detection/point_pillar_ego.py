import torch
import torch.nn as nn


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone


class PointPillarEgo(nn.Module):
    def __init__(self, args):
        super(PointPillarEgo, self).__init__()
        
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])

        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], self.scatter.num_bev_features)

        self.batch_norm = nn.BatchNorm2d(self.backbone.num_bev_features)

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
                      'record_len': record_len}

    
        batch_dict = self.pillar_vfe(batch_dict)        # pillar_features: (sum(num_cav) * num_voxels, C)
        
        batch_dict = self.scatter(batch_dict)           # spatial_features: (sum(num_cav), C * self.nz, self.ny, self.nx)
       
        batch_dict = self.backbone(batch_dict)          # spatial_features_2d:(sum(num_cav), C , H, W)

        spatial_features_2d = batch_dict['spatial_features_2d']

        spatial_features_2d = self.batch_norm(spatial_features_2d)               # (sum(num_cav), C , H, W)

        # split_x: [(num_cav1,C,H,W), (num_cav2,C,H,W), ...,(num_cav n,C,H,W)]
        split_x = self.regroup(spatial_features_2d, batch_dict['record_len'])
        out = []
        
        for xx in split_x:
            # xx (num_cav, C, H, W)
            h = xx[0].unsqueeze(0)                                              # (1, C, H, W)
            out.append(h)
        
        spatial_features_2d = torch.cat(out, dim=0)                             # (B,C,W,H)

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        if need_box:
            output_dict = {}
            output_dict['ego'] = {  'psm': psm,                          # (B, anchor_num , H, W)
                                'rm': rm                             # (B, anchor_num*7 , H, W)
                            }
            pred_box, pred_score, gt_box = dataset.post_process(batch_data, output_dict)
            
            return pred_box, pred_score, gt_box
        else:
            output_dict = {'psm': psm,                          # (B, anchor_num * num_class , H, W)
                        'rm': rm                             # (B, anchor_num*7 , H, W)
                        }

            return output_dict