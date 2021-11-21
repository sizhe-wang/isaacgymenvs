"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

yumi Cube Pick
----------------
Use Jacobian matrix and inverse kinematics control of yumi robot to pick up a cube.
Damped Least Squares method from: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
"""
import os
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from examples.myutils import load_assets
from examples.myutils import create_assets
from examples.myutils import compute
from examples.myutils import log
from examples.myutils import save
from examples.myutils.print_info import printInfo

import math
import numpy as np
import torch
import random
import time
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models
from PIL import Image

# parameters ==================================================
# camera
location = gymapi.Vec3(0.2, 0.1, 0.8)
lookat = gymapi.Vec3(0.4, 0, 0.1)
width = 256
height = 256
# table table上方为workspace，方块掉出会被reset
table_dims = gymapi.Vec3(0.5, 0.5, 0.1)
table_pos_p = gymapi.Vec3(0.4, 0, 0.5 * table_dims.z)
# cube
cube_size = 0.035    # 0.035
# 方块可放位置的中心
cube_middle = gymapi.Transform()
cube_middle.p = gymapi.Vec3(table_pos_p.x, table_pos_p.y, table_dims.z + 0.5 * cube_size)   # x, y和table的pos一样
# 每多少帧检测一次是否有env需要reset方块
reset_frames = 50
# ================================================================

# set random seed
np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="yumi Jacobian Inverse Kinematics Example")

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    # sim_params.physx.max_gpu_contact_pairs = 2097152
    # 增大max_gpu_contact_pairs来解决场景中碰撞较多时GPU显存溢出的问题
else:
    raise Exception("This example can only be used with PhysX")

# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# create sim
# sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    raise Exception("Failed to create sim")

headless = True
# # create viewer
if not headless:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")

# load asset
asset_root = "../../assets"
# create table asset
table_asset = load_assets.load_table(gym, sim, table_dims)
# create cube asset
cube_asset = load_assets.load_cube(gym, sim, cube_size)
# load yumi asset
asset_file = "urdf/yumi_description/urdf/yumi.urdf"
# asset_file = "urdf/franka_description/robots/franka_panda.urdf"


yumi_asset = load_assets.load_yumi(gym, sim, asset_root, asset_file)

# configure yumi dofs
yumi_dof_props, default_dof_pos, default_dof_state = compute.compute_yumi_default_dof(gym, yumi_asset)

# get link index of panda hand, which we will use as end effector
yumi_link_dict = gym.get_asset_rigid_body_dict(yumi_asset)  # return 包含主体名称和资产相关索引之间映射的字典
yumi_hand_index = yumi_link_dict["gripper_r_base"]
# yumi_hand_index = yumi_link_dict["panda_hand"]


# set light parameters
intensity = gymapi.Vec3(0.21, 0.21, 0.21)
ambient = gymapi.Vec3(0.4, 0.4, 0.4)
light_height = 1.5
direction1 = gymapi.Vec3(1.0, 1.0, light_height)    # 光源的位置（每个env相同，无法每个env不同)
direction2 = gymapi.Vec3(1.0, -1.0, light_height)
direction3 = gymapi.Vec3(-0.1, 1.0, light_height)
direction4 = gymapi.Vec3(-0.1, -1.0, light_height)
same = True     # 四盏灯位置是否相同
if same:
    gym.set_light_parameters(sim, 0, intensity, ambient, direction1)
    gym.set_light_parameters(sim, 1, intensity, ambient, direction1)
    gym.set_light_parameters(sim, 2, intensity, ambient, direction1)
    gym.set_light_parameters(sim, 3, intensity, ambient, direction1)
else:
    gym.set_light_parameters(sim, 0, intensity, ambient, direction1)
    gym.set_light_parameters(sim, 1, intensity, ambient, direction2)
    gym.set_light_parameters(sim, 2, intensity, ambient, direction3)
    gym.set_light_parameters(sim, 3, intensity, ambient, direction4)


# configure env grid
num_envs = 32

num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
gym.add_ground(sim, plane_params)

# prepare some list
envs = []
cube_idxs = []
hand_idxs = []
init_pos_list = []
init_rot_list = []
cube_handle63 = []
camera_handles = []
default_cube_states = []
cube_id_envs = []

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add table
    table_handle = create_assets.create_table(gym, env, table_dims, table_pos_p, table_asset, i)

    # add cube
    cube_handle, cube_pose = create_assets.create_cube(gym, env, cube_middle, cube_size, cube_asset, table_dims, i)
    cube_handle63.append(cube_handle)

    # get cube's global index of cube in rigid body state tensor
    cube_idx = gym.get_actor_rigid_body_index(env, cube_handle, 0, gymapi.DOMAIN_SIM)
    cube_idxs.append(cube_idx)
    cube_id_env = gym.get_actor_rigid_body_index(env, cube_handle, 0, gymapi.DOMAIN_ENV)
    cube_id_envs.append(cube_id_env)
    default_cube_states.append([cube_pose.p.x, cube_pose.p.y, cube_pose.p.z,
                                cube_pose.r.x, cube_pose.r.y, cube_pose.r.z, cube_pose.r.w,
                                0, 0, 0, 0, 0, 0])
    # rigid body state:
    # position([0:3]),
    # rotation([3:7]),
    # linear velocity([7:10]),
    # angular velocity([10:13])

    # add yumi
    yumi_handle, yumi_pose = create_assets.create_yumi(gym, env, yumi_asset, i)
    num_dofs = gym.get_actor_dof_count(env, yumi_handle)
    props = gym.get_actor_dof_properties(env, yumi_handle)

    # add camera
    camera_handle = create_assets.create_camera(gym, env, location, lookat, width, height)
    camera_handles.append(camera_handle)

    # 打印

    print("yumi dof properties:")
    print(yumi_dof_props)
    if printInfo.print_num_dofs:
        print("yumi dof:")
        print(num_dofs)
        printInfo.print_num_dofs = False
        # yumi dof:
        # 9
        # yumi dof properties
        # [( True, -2.9408798, 2.9408798, 0, 3.1415927, 300., 0., 0.5, 0., 0.)
        #  ( True, -2.5045474, 0.7592182, 0, 3.1415927, 300., 0., 0.5, 0., 0.)
        #  ( True, -2.9408798, 2.9408798, 0, 3.1415927, 300., 0., 0.5, 0., 0.)
        #  ( True, -2.1554816, 1.3962634, 0, 3.1415927, 300., 0., 0.5, 0., 0.)
        #  ( True, -5.061455 , 5.061455 , 0, 6.981317 , 300., 0., 0.5, 0., 0.)
        #  ( True, -1.5358897, 2.4085543, 0, 6.981317 , 300., 0., 0.5, 0., 0.)
        #  ( True, -3.996804 , 3.996804 , 0, 6.981317 , 300., 0., 0.5, 0., 0.)
        #  ( True,  0.       , 0.025    , 0, 2.       ,  20., 0., 1. , 0., 0.)
        #  ( True,  0.       , 0.025    , 0, 2.       ,  20., 0., 1. , 0., 0.)]

    # set dof properties
    gym.set_actor_dof_properties(env, yumi_handle, yumi_dof_props)
    # set initial dof states
    gym.set_actor_dof_states(env, yumi_handle, default_dof_state, gymapi.STATE_ALL)
    # set initial position targets
    gym.set_actor_dof_position_targets(env, yumi_handle, default_dof_pos)

    # get inital hand pose
    hand_handle = gym.find_actor_rigid_body_handle(env, yumi_handle, "gripper_r_base")
    # hand_handle = gym.find_actor_rigid_body_handle(env, yumi_handle, "panda_hand")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    # get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env, yumi_handle, "gripper_r_base", gymapi.DOMAIN_SIM)
    # hand_idx = gym.find_actor_rigid_body_index(env, yumi_handle, "panda_hand", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

default_cube_states = to_torch(default_cube_states, device=device, dtype=torch.float).view(num_envs, 13)  # view功能相当于reshape

_cube_idx = cube_idxs

# point camera at middle env
if not headless:
    cam_pos = gymapi.Vec3(4, 3, 2)
    cam_target = gymapi.Vec3(-4, -3, 0)
    middle_env = envs[num_envs // 2 + num_per_row // 2]
    gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# cube_pose_init = np.copy(gym.get_actor_rigid_body_states(env, cube_idxs[0], gymapi.STATE_POS))
cube_pose_init = gym.get_actor_rigid_body_states(env, cube_idxs[0], gymapi.STATE_POS)

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# initial hand position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)

# hand orientation for grasping
down_q = torch.stack(num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view((num_envs, 4))

# cube corner coords, used to determine grasping yaw
cube_half_size = 0.5 * cube_size
corner_coord = torch.Tensor([cube_half_size, cube_half_size, cube_half_size])
corners = torch.stack(num_envs * [corner_coord]).to(device)

# downard axis
down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

# get jacobian tensor
# for fixed-base yumi, tensor has shape (num envs, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "yumi")
jacobian = gymtorch.wrap_tensor(_jacobian)

# jacobian entries corresponding to yumi hand
j_eef = jacobian[:, yumi_hand_index - 1, :]

# get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, 9, 1)

# get root tensor
_root_tensor = gym.acquire_actor_root_state_tensor(sim)
root_tensor = gymtorch.wrap_tensor(_root_tensor)

# Create a tensor noting whether the hand should return to the initial position
hand_restart = torch.full([num_envs], False, dtype=torch.bool).to(device)

# simulation loop
count = 0

global_indices = torch.arange(num_envs * 3, dtype=torch.int32, device=device).view(num_envs, -1)

if printInfo.print_cube_idxs:
    print("cube_idxs:")
    print(cube_idxs[63])
    print(cube_idxs)

if printInfo.print_start:
    print("====================================== Simulation start ======================================")
model = models.resnet34(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))

model.to(device)
model.eval()
preprocess = transforms.Compose([            #[1]
     # transforms.Resize(472),                    #[2]
     # transforms.CenterCrop(472),                #[3]
     # transforms.ToTensor(),                     #[4]
     transforms.Normalize(                      #[5]
     mean=[0.485, 0.456, 0.406],                #[6]
     std=[0.229, 0.224, 0.225]                  #[7]
     )])
# while not gym.query_viewer_has_closed(viewer):
render_img = True

while True:

    time_start = time.time()
    count += 1
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # communicate physics to graphics system
    gym.step_graphics(sim)



    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)

    cube_pos = rb_states[cube_idxs, :3]
    cube_rot = rb_states[cube_idxs, 3:7]

    hand_pos = rb_states[hand_idxs, :3]
    hand_rot = rb_states[hand_idxs, 3:7]

    to_cube = cube_pos - hand_pos
    cube_dist = torch.norm(to_cube, dim=-1).unsqueeze(-1)
    cube_dir = to_cube / cube_dist
    cube_dot = cube_dir @ down_dir.view(3, 1)

    # how far the hand should be from cube for grasping
    grasp_offset = 0.12  # 0.12

    # determine if we're holding the cube (grippers are closed and cube is near)
    gripper_sep = dof_pos[:, 7] + dof_pos[:, 8]
    gripped = (gripper_sep < cube_size) & (cube_dist < grasp_offset + 0.5 * cube_size)

    yaw_q = compute.cube_grasping_yaw(cube_rot, corners)
    cube_yaw_dir = compute.quat_axis(yaw_q, 0)
    hand_yaw_dir = compute.quat_axis(hand_rot, 0)
    yaw_dot = torch.bmm(cube_yaw_dir.view(num_envs, 1, 3), hand_yaw_dir.view(num_envs, 3, 1)).squeeze(-1)

    # determine if we have reached the initial position; if so allow the hand to start moving to the cube
    to_init = init_pos - hand_pos
    init_dist = torch.norm(to_init, dim=-1)
    hand_restart = (hand_restart & (init_dist > 0.02)).squeeze(-1)
    return_to_start = (hand_restart | gripped.squeeze(-1)).unsqueeze(-1)

    # if hand is above cube, descend to grasp offset
    # otherwise, seek a position above the cube
    above_cube = ((cube_dot >= 0.99) & (yaw_dot >= 0.95) & (cube_dist < grasp_offset * 3)).squeeze(-1)
    grasp_pos = cube_pos.clone()
    grasp_pos[:, 2] = torch.where(above_cube, cube_pos[:, 2] + grasp_offset, cube_pos[:, 2] + grasp_offset * 2.5)

    # compute goal position and orientation
    goal_pos = torch.where(return_to_start, init_pos, grasp_pos)
    goal_rot = torch.where(return_to_start, init_rot, quat_mul(down_q, quat_conjugate(yaw_q)))

    # compute position and orientation error
    pos_err = goal_pos - hand_pos
    orn_err = compute.orientation_error(goal_rot, hand_rot)
    dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    d = 0.05  # damping term
    lmbda = torch.eye(6).to(device) * (d ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 9, 1)

    # update position targets
    pos_target = dof_pos + u

    # gripper actions depend on distance between hand and cube
    close_gripper = (cube_dist < grasp_offset + 0.02) | gripped
    # always open the gripper above a certain height, dropping the cube and restarting from the beginning
    hand_restart = hand_restart | (cube_pos[:, 2] > table_dims.z + 0.1)
    keep_going = torch.logical_not(hand_restart)
    close_gripper = close_gripper & keep_going.unsqueeze(-1)
    grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * num_envs).to(device), torch.Tensor([[0.04, 0.04]] * num_envs).to(device))

    pos_target[:, 7:9] = grip_acts.unsqueeze(-1)
    if printInfo.print_hand_pose:
        print("hand pos:")
        print(hand_pose.p)
        print(hand_pose.r)
        printInfo.print_hand_pose = False

    # set new position targets
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_target))

    # 保存相机图片
    if count % 1 == 0:
        if render_img:
            # render the camera sensors
            gym.render_all_camera_sensors(sim)
            # save.save_images(gym, sim, num_envs, envs, camera_handles, "yumi_image")
            gym.start_access_image_tensors(sim)
            #
            # User code to digest tensors
            #
            # get image tensor
            image_tensors = []

            for j in range(num_envs):
                _image_tensor = gym.get_camera_image_gpu_tensor(sim, envs[j], camera_handles[j], gymapi.IMAGE_COLOR)
                # H * W * 3
                image_tensor = gymtorch.wrap_tensor(_image_tensor)[:, :, :3].permute(2, 0, 1).contiguous()
                image_tensors.append(image_tensor)

                show_image = False
                if show_image:
                    image_array = image_tensor.permute(1, 2, 0).cpu().numpy()
                    image = Image.fromarray(image_array).convert("RGB")
                    image.show()
                    # exit()
            gym.end_access_image_tensors(sim)

            image_tensors = torch.stack(image_tensors)
            # Normalize
            image_tensors = image_tensors / 255.
            image_tensors = preprocess(image_tensors)
            # print(image_tensors.shape)
            output = model(image_tensors.view(-1, 3, 256, 256)).squeeze()
            # print(output)
            # print(output.size())
            # exit()

            # save.save_images(gym, sim, num_envs, envs, camera_handles, "yumi_image")
            # save a copy of the original root states
            # saved_root_tensor = root_tensor.clone()

    # reset
    # ========================================================================================
    # yumi夹具活动范围：x[x_low, x_up],y[y_low, y_up]（活动范围即桌面上）
    # reset cube when unreachable
    # ========================================================================================
    if count % reset_frames == 0:
        x_low = table_pos_p.x - table_dims.x / 2.
        x_up = table_pos_p.x + table_dims.x / 2.
        y_low = table_pos_p.y - table_dims.y / 2.
        y_up = table_pos_p.y + table_dims.y / 2.
        reset_buf = (cube_pos[:, 0] > x_up) | (cube_pos[:, 0] < x_low) | (cube_pos[:, 1] < y_low) | (cube_pos[:, 1] > y_up)
        reset_int = torch.where(reset_buf, torch.ones_like(reset_buf), torch.zeros_like(reset_buf))
        reset_env_ids = reset_int.nonzero(as_tuple=False).squeeze(-1)
        cube_indices = global_indices[reset_env_ids, 1].flatten()
        resize_root_tensor_cube = torch.tensor([cube_middle.p.x, cube_middle.p.y, cube_middle.p.z, 1, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0]).to(device).repeat(num_envs * 3).view(num_envs, 3, 13)
        resize_root_tensor_cube[:, 1, :2] += 0.2 * torch.rand((num_envs, 2), device=device) - 0.1
        resize_root_tensor_cube[:, 1, 3: 7] += 0.2 * (torch.rand((num_envs, 4), device=device) - 0.1)
        root_tensor_cube = resize_root_tensor_cube.view(num_envs * 3, 13)

        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root_tensor_cube),
                                                gymtorch.unwrap_tensor(cube_indices), len(cube_indices))

        # print
        if printInfo.print_reset:
            print("reset%d" % (count//reset_frames))
    print("run%d"%count)
    print('time cost is :', time.time() - time_start)

    # gym.step_graphics(sim)

    if not headless:
        # render the viewer
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time to sync viewer with
        # simulation rate. Not necessary in headless.
        gym.sync_frame_time(sim)

        # Check for exit condition - user closed the viewer window
        if gym.query_viewer_has_closed(viewer):
            break


# cleanup
if not headless:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
