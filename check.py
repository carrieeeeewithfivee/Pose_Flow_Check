import numpy as np 
import cv2
from gt_process import *
from pnp import *
from flow_vis import *
import matplotlib.pyplot as plt

#get data
flow_path = 'Data/flow/000000.npy'
depth_path = 'Data/depth/000000.npy'
K_path = 'Data/K/000000.txt'
pose_path = 'Data/pose/000000.npy'

#load data
gt_pose = np.load(pose_path)
cam2cam = {}
with open(K_path, 'r') as f:
    for line in f.readlines():
        key, value = line.split(':', 1)
        cam2cam[key] = np.array([float(x) for x in value.split()])

with open(flow_path, 'rb') as f:
    flow_tensor = np.load(f, allow_pickle=True).astype(np.float32) # (2, 384, 1280)
    flow_tensor = cv2.resize(flow_tensor.transpose(1,2,0), (256,128), interpolation=cv2.INTER_LINEAR)
    flow_tensor[:,:,0] = flow_tensor[:,:,0] / 1280. * 256.
    flow_tensor[:,:,1] = flow_tensor[:,:,1] / 384. * 128.
        
with open(depth_path, 'rb') as f: 
    depth_tensor = np.load(f, allow_pickle=True).astype(np.float32) # (352, 1216)
    depth_tensor = cv2.resize(depth_tensor, (256,128), interpolation=cv2.INTER_LINEAR)

#resize K
K = cam2cam['P2'].reshape(3, 4)[:, :3]
K_scaled = K.copy() #1280, 384 -> 256, 128
K_scaled[0] = K[0] * 256 / 1280
K_scaled[1] = K[1] * 128 / 384
K = K_scaled
K_inverse = np.linalg.inv(K)

#get focal lenth and baseline
fl, bl = get_focal_length_baseline(cam2cam, cam=2) #718.856 0.5323318578407914

#calculate depth
depth = fl*bl / np.clip(depth_tensor, 1e-3, float('inf'))

#rigid flow from gt
relative_camera_translation = gt_pose[0][:3]
relative_camera_rotation = deg2mat_xyz(gt_pose[0][3:])
pixel_coord_t0 = pixel_coord_generation(depth)
_, rigid_flow = create_motion(K, K, relative_camera_rotation, relative_camera_translation, pixel_coord_t0, depth, False)
object_flow = flow_tensor - rigid_flow

#rigid flow from pnp
pose_mat, _, inlier_ratio = depth_flow2pose(depth, flow_tensor, K, K_inverse)
pnp_rigid_flow = depth_pose2flow_pt(depth, pose_mat, K, K_inverse)
pnp_rigid_flow = np.moveaxis(pnp_rigid_flow,0,-1)
object_pnp_flow = flow_tensor - pnp_rigid_flow

#vis
rad_max, _, _ = find_rad_minmax(flow_tensor)
color_flow = flow_to_color(flow_tensor, rad_max=rad_max)
color_rigid = flow_to_color(rigid_flow, rad_max=rad_max)
color_pnp_rigid = flow_to_color(pnp_rigid_flow, rad_max=rad_max)
color_object = flow_to_color(object_flow, rad_max=rad_max)
color_pnp_object = flow_to_color(object_pnp_flow, rad_max=rad_max)
#plot
fig = plt.figure()
ax1 = fig.add_subplot(321)
ax1.title.set_text('Total Flow')
ax1.imshow(color_flow)
ax2 = fig.add_subplot(322)
ax2.title.set_text('Depth')
ax2.imshow(depth)
ax3 = fig.add_subplot(323)
ax3.title.set_text('Rigid GT Flow')
ax3.imshow(color_rigid)
ax3 = fig.add_subplot(324)
ax3.title.set_text('Rigid pnp Flow')
ax3.imshow(color_pnp_rigid)
ax3 = fig.add_subplot(325)
ax3.title.set_text('Object GT Flow')
ax3.imshow(color_object)
ax3 = fig.add_subplot(326)
ax3.title.set_text('Object pnp Flow')
ax3.imshow(color_pnp_object)
plt.tight_layout()
plt.show()