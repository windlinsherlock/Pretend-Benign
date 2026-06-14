import numpy as np
import torch
import opencood.utils.common_utils as common_utils
from scipy.optimize import linear_sum_assignment
from opencood.pcdet_utils.iou3d_nms import iou3d_nms_utils
import time



# compute max_step from attacker_ratio and num_consensus
def cal_robosac_steps(num_agent, num_consensus, num_attackers):
    # exclude ego agent
    num_agent = float(num_agent - 1)
    eta = num_attackers / num_agent

    epsilon = 1e-10  # small value to prevent division by zero
    eta = np.clip(eta, 0, 1 - epsilon)  # ensure eta does not equal 1
    # print(num_attackers, num_agent, eta, num_consensus)
    N = np.ceil(np.log(1 - 0.99) / np.log(1 - np.power(1 - eta, num_consensus))).astype(int)
    return N


# compute consensus_size from step_budget and attacker_ratio
def cal_robosac_consensus(num_agent, step_budget, num_attackers):
    num_agent = num_agent - 1
    eta = num_attackers / num_agent
    s = np.floor(np.log(1-np.power(1-0.99, 1/step_budget)) / np.log(1-eta)).astype(int)
    return s

def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return torch.tensor(list(zip(x, y)), dtype=torch.long)

def get_jaccard_score(ego_boxes, collab_boxes, iou_threshold=0.5):
    '''
    ego_boxes:           (N, 7)
    collab_boxes:        (M, 7)
    '''
    N = ego_boxes.shape[0]
    M = collab_boxes.shape[0]
    if N == 0 or M == 0:
        return 0.0

    ious = iou3d_nms_utils.boxes_iou3d_gpu(ego_boxes, collab_boxes)
       
    a = (ious > iou_threshold).int() 
    # [[1 0 0 0]
    #  [0 1 0 0]
    #  [0 0 1 0]]
    # print(a.sum(1)): [1 1 1]
    # print(a.sum(0)): [1 1 1 0]
    # check the unique-match case
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = torch.stack(torch.where(a), dim=1).cpu()
        # [[0 0]
        #  [1 1]
        #  [2 2]]
    else:
        ious_cpu = ious.cpu().numpy()
        matched_indices = linear_assignment(-ious_cpu)
        
    unmatched_ego_boxes = []
    for d in range(N):
        if d not in matched_indices[:, 0]:
            unmatched_ego_boxes.append(d)
    
    unmatched_collab_boxes = []
    for t in range(M):
        if t not in matched_indices[:, 1]:
            unmatched_collab_boxes.append(t)
    
    matches = []
    for m in matched_indices:
        if ious[m[0], m[1]] < iou_threshold:
            unmatched_ego_boxes.append(m[0].item())
            unmatched_collab_boxes.append(m[1].item())
        else:
            matches.append(m.unsqueeze(0))

    intersect = len(matches)
    union = float(N + M - intersect)

    jaccard_score = intersect / union if union > 0 else 0.0
        
    return jaccard_score

