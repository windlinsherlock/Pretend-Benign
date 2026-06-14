import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F


def x_to_bev(pose):
    """
    Transformation matrix from x-coordinate system to BEV space.

    Parameters
    ----------
    pose : array
        [x, y, yaw]

    Returns
    -------
    matrix : np.ndarray
        The 3x3 transformation matrix in BEV space.
    """
    x, y, yaw = pose[:]

    # Compute rotation matrix components
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))

    # Initialize transformation matrix
    matrix = np.identity(3)
    # Translation part
    matrix[0, 2] = x
    matrix[1, 2] = y

    # Rotation part
    matrix[0, 0] = c_y
    matrix[0, 1] = -s_y
    matrix[1, 0] = s_y
    matrix[1, 1] = c_y

    return matrix


def bev1_to_bev2(bev1, bev2):
    """
    Transformation matrix from bev1 to bev2.

    Parameters
    ----------
    bev1 : array
        The pose of bev1 under world coordinates [x, y, yaw].
    bev2 : array
        The pose of bev2 under world coordinates [x, y, yaw].

    Returns
    -------
    transformation_matrix : np.ndarray
        The 3x3 transformation matrix in BEV space.
    """
    bev1 = bev1.detach().cpu().numpy()
    bev2 = bev2.detach().cpu().numpy()
    
    bev1_to_world = x_to_bev(bev1)
    bev2_to_world = x_to_bev(bev2)
    world_to_bev2 = np.linalg.inv(bev2_to_world)

    transformation_matrix = np.dot(world_to_bev2, bev1_to_world)
    return transformation_matrix


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def project_points_by_bev_matrix_torch(points, transformation_matrix, device):
    """
    Project the points to another coordinate system in BEV space 
    based on the transformation matrix.

    Parameters
    ----------
    points : torch.Tensor
        2D points in BEV space, (N, 2)
    transformation_matrix : torch.Tensor
        Transformation matrix in BEV space, (3, 3)

    Returns
    -------
    projected_points : torch.Tensor
        The projected points, (N, 2)
    """
    # Ensure points are in torch.Tensor format
    points, _ = check_numpy_to_torch(points)
    transformation_matrix, _ = check_numpy_to_torch(transformation_matrix)
    
    points = points.to(device)
    transformation_matrix = transformation_matrix.to(device)

    # Convert points to homogeneous coordinates: (N, 3)
    points_homogeneous = F.pad(points, (0, 1), mode="constant", value=1)

    # Perform matrix multiplication: (N, 3) @ (3, 3) -> (N, 3)
    projected_points = torch.mm(points_homogeneous, transformation_matrix.T)
    
    # Return the first two columns (x, y) after projection
    return projected_points[:, :2]


def feature_wraps(feature_sequence, history_pose_list, cur_pose_list, points_min, voxel_size):
    '''
    feature_sequence: (B, N, C, H, W)
    history_pose_list: (B, N, 3)
    cur_pose_list: (B, 3)
    points_min: (2)
    voxel_size: float
    '''
    device = feature_sequence.device
    batch_size = feature_sequence.shape[0]
    sequence_len = feature_sequence.shape[1]
    H, W = feature_sequence.shape[-2], feature_sequence.shape[-1]
    
    cur_feature_sequence = feature_sequence.clone()
    
    y_indices, x_indices = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'                           # (H, W)
    )
    
    y_centers = points_min[1] + (y_indices + 0.5) * voxel_size
    x_centers = points_min[0] + (x_indices + 0.5) * voxel_size
    grid_centers = torch.stack((x_centers, y_centers), dim=-1).reshape(H*W, 2)              # (H*W, 2)
    
    for i in range(batch_size):
        history_pose = history_pose_list[i]                     # (N,3)                                  
        cur_pose = cur_pose_list[i]                             # (3)
        
        for j in range(sequence_len):
            transformation_matrix = bev1_to_bev2(history_pose[j], cur_pose)         # (3,3)
            points = grid_centers.clone()                       # (H*W, 2)
            trans_points = project_points_by_bev_matrix_torch(points, transformation_matrix, device)
            
            center = trans_points - points_min
            center = (center / voxel_size).to(torch.long)
            x, y = center[:, 0], center[:, 1]                               # (H*W)
            # prevent out-of-bounds access
            x = torch.clamp(x, 0, W-1)                              
            y = torch.clamp(y, 0, H-1)
            x_index, y_index = x_indices.reshape(-1).to(torch.long), y_indices.reshape(-1).to(torch.long)      # (H*W)
            
            # mask = (x == x_index) & (y == y_index)
            # print(mask.sum())
            # print((x-x_index).abs().max(), (y-y_index).abs().max())
            
            cur_feature_sequence[i, j, :, y, x] = feature_sequence[i, j, :, y_index, x_index]
            
    return cur_feature_sequence


