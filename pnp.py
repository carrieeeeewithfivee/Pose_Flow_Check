import numpy as np
import cv2
#from https://github.com/lliuz/UnRigidFlow/tree/master/utils

def get_valid_depth(gt, crop=True):
    valid = (gt > 0) & (gt < 80)
    if crop:
        h, w = gt.shape[:2]
        crop_mask = gt != gt
        y1, y2 = int(0.40810811 * h), int(0.99189189 * h)
        x1, x2 = int(0.03594771 * w), int(0.96405229 * w)
        crop_mask[y1:y2, x1:x2] = 1
        valid = valid & crop_mask
    return valid

def depth_flow2pose(depth, flow, K, K_inv, gs=16, th=1., method='AP3P', depth2=None):
    """
    :param depth:       H x W
    :param flow:        h x w x2
    :param K:           3 x 3
    :param K_inv:       3 x 3
    :param gs:          grad size for sampling
    :param th:          threshold for RANSAC
    :param method:      PnP method
    :return:
    """
    if method == 'PnP':
        PnP_method = cv2.SOLVEPNP_ITERATIVE
    elif method == 'AP3P':
        PnP_method = cv2.SOLVEPNP_AP3P
    elif method == 'EPnP':
        PnP_method = cv2.SOLVEPNP_EPNP
    else:
        raise ValueError('PnP method ' + method)

    H, W = depth.shape[:2]
    valid_mask = get_valid_depth(depth)
    sample_mask = np.zeros_like(valid_mask)
    sample_mask[::gs, ::gs] = 1
    valid_mask &= sample_mask == 1

    h, w = flow.shape[:2]
    flow[:, :, 0] = flow[:, :, 0] / w * W
    flow[:, :, 1] = flow[:, :, 1] / h * H
    flow = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR)

    grid = np.stack(np.meshgrid(range(W), range(H)), 2).astype(
        np.float32)  # HxWx2
    one = np.expand_dims(np.ones_like(grid[:, :, 0]), 2)
    homogeneous_2d = np.concatenate([grid, one], 2)
    d = np.expand_dims(depth, 2)
    points_3d = d * (K_inv @ homogeneous_2d.reshape(-1, 3).T).T.reshape(H, W, 3)

    points_2d = grid + flow
    valid_mask &= (points_2d[:, :, 0] < W) & (points_2d[:, :, 0] >= 0) & \
                  (points_2d[:, :, 1] < H) & (points_2d[:, :, 1] >= 0)

    ret, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d[valid_mask],
                                                  points_2d[valid_mask],
                                                  K, np.zeros([4, 1]),
                                                  reprojectionError=th,
                                                  flags=PnP_method)
    if not ret:
        inlier_ratio = 0.
    else:
        inlier_ratio = len(inliers) / np.sum(valid_mask)
    pose_mat = np.eye(4, dtype=np.float32)
    pose_mat[:3, :] = cv2.hconcat([cv2.Rodrigues(rvec)[0], tvec])

    return pose_mat, np.concatenate([rvec, tvec]), inlier_ratio #pose_mat, pose_vec, inlier_ratio

def mesh_grid(H, W):
    # mesh grid
    xv, yv = np.meshgrid(range(H), range(W), indexing='ij')
    base_grid = np.stack([xv, yv], axis=0)
    return base_grid

def depth_pose2flow_pt(depth, pose, K, K_inv):
    """ The intrinsic K and K_inv should match with the size of depth.
    :param depth:   B x H x W
    :param pose:    B x 4 x 4
    :param K:       B x 3 x 3
    :param K_inv:   3 x 3
    :return:        B x 2 x H x W
    """
    # depth to camera coordinates
    H, W = depth.shape

    grid = mesh_grid(H, W)
    ones = np.ones((1, H, W))
    homogeneous_2d = np.concatenate([grid, ones], axis=0).reshape(3, -1)
    d = depth[np.newaxis, :, :]
    points_3d = d * (K_inv @ homogeneous_2d).reshape(3, H, W)

    # camera coordinates to pixel coordinates
    homogeneous_3d = np.concatenate([points_3d, ones], axis=0).reshape(4, -1)
    points_2d = K @ (pose @ homogeneous_3d)[:3]  # [B, 3, H*W]
    points_2d = points_2d.reshape(3, H, W)
    temp = np.clip(points_2d[2], 1e-3, float('inf'))[np.newaxis, :, :]
    points_2d = points_2d[:2] / temp
    flow = points_2d - grid

    return flow