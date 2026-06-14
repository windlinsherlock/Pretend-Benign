import torch
import torch.nn as nn
import random

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.attack_modules.base_module.co_attack_module import CoAttackModule, GPSAttack
from opencood.models.attack_modules.utils.robosac_utils import cal_robosac_steps, get_jaccard_score
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

        output_dict['ego'] = {  'psm': psm,                          # (B, anchor_num , H, W)
                                'rm': rm                             # (B, anchor_num*7 , H, W)
                            }
        
        pred_box, pred_score, gt_box = dataset.post_process(batch_data, output_dict)

        return pred_box, pred_score, gt_box


class Robosac(nn.Module):
    def __init__(self):
        super(Robosac, self).__init__()
        
        self.predict = None
        self.fusion_module = None

    def forward(self, spatial_features_2d, dataset, step_budget, num_cav, batch_data, box_matching_thresh):
        estimate_attacker_nums = [i for i in range(0, num_cav)]
        estimated_attacker_nums = num_cav-1
        
        NMax = []                # probing_step_limit_by_attacker_ratio
        for temp_num_attackers in estimate_attacker_nums:
            temp_num_consensus = num_cav - temp_num_attackers - 1
            NMax.append(cal_robosac_steps(num_cav, temp_num_consensus, temp_num_attackers))

        NMax[0] = 1
        NTry = [0] * len(estimate_attacker_nums)
        
        # print(NMax, num_cav)

        step = 0
        all_agent_list = [i for i in range(1, num_cav)]

        ego_features = spatial_features_2d[0].unsqueeze(0)

        ego_box, ego_score, gt_box = self.predict(dataset, ego_features, batch_data)

        jac_score = get_jaccard_score(ego_box, gt_box)

        # print(jac_score)
        
        assert step_budget >= num_cav, "step_budget must be larger than num_cav"
        
        succ_result = {
            'pred_box': ego_box,
            'pred_score': ego_score,
            'gt_box': gt_box
        }
        estimated_collab_agent_list = []

        while step < step_budget and NTry < NMax:

            for i in range(len(estimate_attacker_nums)):
                consensus_set_size = num_cav - 1 - estimate_attacker_nums[i]
                if NTry[i] < NMax[i]:
                    # print("Probing {} agents for consensus".format(consensus_set_size))
                    step += 1
                    collab_agent_list = random.sample(
                        all_agent_list, k=consensus_set_size
                    )
                    collab_agent_list.append(0)
                    collab_features = spatial_features_2d[collab_agent_list]
                    fused_features = self.fusion_module(collab_features)
                    collab_agent_list = collab_agent_list[:-1]

                    collab_box, collab_score, _ = self.predict(dataset, fused_features, batch_data)

                    jac_score = get_jaccard_score(ego_box, collab_box)
                    # print("Jaccard Coefficient: {}".format(jac_score))

                    if jac_score < box_matching_thresh:
                        # print('No consensus reached when probing {} consensus agents. Current step is {}'.format(consensus_set_size, step))
                        # print('Attacker(s) is(are) among {}'.format(collab_agent_list))

                        NTry[i] += 1
                        # if NTry[i] == NMax[i]:
                        #     print("Probing of {} agents for consensus has reached its sampling limit {} with assumed attacker num {} and consensus set size {}." \
                        #           .format(consensus_set_size, NMax[i], estimate_attacker_nums[i], consensus_set_size))
                        #     print("From now on we won't try to probe {} agents consensus since it seems unlikely to reach that.".format(consensus_set_size))
                    else:
                        sus_agent_list = [i for i in all_agent_list if i not in collab_agent_list]
                        # print('Achieved consensus at step {}, with {} agents: {}. Using the result as temporal final output of this frame, \
                        #             and skipping smaller consensus set tries. \n Attacker(s) is(are) among {}, excluded.'.format(
                        #             step, consensus_set_size, collab_agent_list, sus_agent_list))
                        
                        succ_result['pred_box'] = collab_box
                        succ_result['pred_score'] = collab_score

                        if estimate_attacker_nums[i] < estimated_attacker_nums:
                            # print('Larger consensus set ({} agents) probed. We will skip all the smaller consensus set tries. \
                            # Update attacker num estimation to {}'.format(
                            #     consensus_set_size, estimate_attacker_nums[i]))
                            
                            estimated_attacker_nums = estimate_attacker_nums[i]
                            estimated_collab_agent_list = collab_agent_list

                            for j in range(i, len(estimate_attacker_nums)):
                                NTry[j] = NMax[j]
        
        step += 1
        
        jac_score = get_jaccard_score(succ_result['pred_box'], succ_result['gt_box'])
        # print(estimated_collab_agent_list, step, succ_result['pred_box'].shape, succ_result['pred_score'].shape, succ_result['gt_box'].shape, jac_score)

        return estimated_collab_agent_list, succ_result, step


class PointPillarRobosac(nn.Module):
    def __init__(self, args):
        super(PointPillarRobosac, self).__init__()
        
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
        self.robosac = Robosac()

    
    def forward(self, batch_data, pert, attacker_list, no_fuse, dataset, step_budget, box_matching_thresh, LCF_mask=None, GPS_Attack=False):
        
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
        self.robosac.predict = self.predict
        self.robosac.fusion_module = self.fusion_module
        
        # start time
        start_time = time.perf_counter()

        # print(attacker_list, num_cav)
        
        collab_agent_list, succ_result, step = self.robosac(spatial_features_2d, dataset, step_budget, num_cav, batch_data, box_matching_thresh)
        
        torch.cuda.synchronize()  # ensure all GPU tasks are complete
        end_time = time.perf_counter()
        time_cost = end_time - start_time
        print(f"Total inference time (CPU + GPU): {time_cost:.6f} seconds")

        return collab_agent_list, succ_result, step, time_cost