import numpy as np

def get_focal_length_baseline(cam2cam, cam=2):
    P2_rect = cam2cam['P2'].reshape(3, 4)
    P3_rect = cam2cam['P3'].reshape(3, 4)

    # cam 2 is left of camera 0  -6cm
    # cam 3 is to the right  +54cm
    b2 = P2_rect[0, 3] / -P2_rect[0, 0]
    b3 = P3_rect[0, 3] / -P3_rect[0, 0]
    baseline = b3 - b2

    focal_length = None
    if cam == 2:
        focal_length = P2_rect[0, 0]
    elif cam == 3:
        focal_length = P3_rect[0, 0]

    return focal_length, baseline

def deg2mat_xyz(deg, left_hand=True):
    if left_hand:
        deg = -deg
    
    rad = np.deg2rad(deg)
    
    x, y, z = np.split(rad, 3, axis=0)
    
    sz = np.sin(z[0])
    cz = np.cos(z[0])
    sy = np.sin(y[0])
    cy = np.cos(y[0])
    sx = np.sin(x[0])
    cx = np.cos(x[0])

    #return np.array([sz, cz, sy, cy, sx, cx])
    
    rot_x = np.array([
        1.0, 0.0, 0.0,
        0.0, cx, -sx,
        0.0, sx, cx
    ]).reshape(3, 3)
    
    rot_y = np.array([
        cy, 0.0, sy,
        0.0, 1.0, 0.0,
        -sy, 0.0, cy
    ]).reshape(3, 3)
    
    rot_z = np.array([
        cz, -sz, 0.0,
        sz, cz, 0.0,
        0.0, 0.0, 1.0
    ]).reshape(3, 3)
    
    R = np.dot(np.dot(rot_x, rot_y), rot_z)
    #return np.array([rot_x, rot_y, rot_z])
    
    return R

def pixel_coord_generation(depth, is_homogeneous=True):
    h, w = depth.shape
    # Np array and Camera image plane are transpose (x <-> y)
    n_x = np.linspace(-1.0, 1.0, w)[..., np.newaxis]
    n_y = np.linspace(-1.0, 1.0, h)[..., np.newaxis]
    
    x_t = np.dot(np.ones((h, 1)), n_x.T)
    y_t = np.dot(n_y, np.ones((1, w)))
    
    x_t = (x_t + 1.0) * 0.5 * (w - 1.0)
    y_t = (y_t + 1.0) * 0.5 * (h - 1.0)
    
    xy_coord = np.concatenate([x_t[..., np.newaxis], y_t[..., np.newaxis]], axis=-1)
    
    if is_homogeneous:
        xy_coord = np.concatenate([xy_coord, np.ones((h, w, 1))], axis=-1)
    
    return xy_coord

def create_motion(K_t0, K_t1, R, T, p_coord_t0, d_t0, gen_depth=False):
    p_shape = p_coord_t0.shape
    
    flat_p_coord_t0 = np.reshape(np.transpose(p_coord_t0, (2, 0, 1)), (3, -1))
    flat_d_t0 = np.reshape(d_t0, (-1))
    
    M = np.vstack((np.hstack((R, T[:, None])), [0, 0, 0 ,1]))
    
    K_t0_inv = np.linalg.inv(K_t0)
    K_t1_pad = np.vstack((np.hstack((K_t1, np.zeros([3, 1]))), [0, 0, 0 ,1]))
    
    filler = np.ones((1, p_shape[0] * p_shape[1]))
    
    c_coord_t0 = np.matmul(K_t0_inv, flat_p_coord_t0)
    
    #print(c_coord_t0.shape, flat_d_t0.shape)
    c_coord_t0 = c_coord_t0 * flat_d_t0
    
    c_coord_t0 = np.vstack((c_coord_t0, filler))
    c_coord_t1 = np.matmul(M, c_coord_t0)
    
    flat_scene_motion = c_coord_t1[:3, :] - c_coord_t0[:3, :]
    
    unnormal_p_coord_t1 = np.matmul(K_t1_pad, c_coord_t1)
    # Absolute for avioding reflection
    p_coord_t1 = unnormal_p_coord_t1 / (np.abs(unnormal_p_coord_t1[2, :]) + 1e-12)
    
    flat_f = p_coord_t1[:2, :] - flat_p_coord_t0[:2, :]
        
    scene_motion = np.transpose(np.reshape(flat_scene_motion, (3, p_shape[0], p_shape[1])), (1, 2, 0))
    f = np.transpose(np.reshape(flat_f, (2, p_shape[0], p_shape[1])), (1, 2, 0))
    
    return scene_motion, f