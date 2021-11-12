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


# ========================================================
# 根据末端执行器的微分运动计算机器人关节的微分运动
# dpose: 末端执行器的微分运动 delta的[p.x, p.y, p.z, r.x, r.y, r.z, r.w]
# u: 关节的微分运动 （只考虑arm，不考虑finger，所以是7个dof）
# j_eef：arm的雅可比矩阵
# 没有阻尼的话 u = torch.inverse(j_eef) @ dpose
# 但是yumi设置的是DOF_MODE_POS（其他选项是DOF_MODE_NONE，DOF_MODE_EFFORT，DOF_MODE_VEL），阻尼必须非零
# ========================================================
def control_ik(dpose, device, j_eef, num_envs, damping=0.05):
    print("j_eef")
    print(j_eef.size())
    print("dpose")
    print(dpose.size())
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)    # torch.eye(6) 6维单位阵
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)    # @ 矩阵-向量乘法
    return u


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def pre_physics_step(self, actions):
    # do actions
    self.actions = actions.clone().to(self.device)  # 没找到action在哪里设置的
    targets = self.yumi_dof_targets[:, :self.num_yumi_dofs] + self.yumi_dof_speed_scales * self.dt * self.actions * self.action_scale
    # 上下限截断
    self.yumi_dof_targets[:, :self.num_yumi_dofs] = tensor_clamp(
        targets, self.yumi_dof_lower_limits, self.yumi_dof_upper_limits)
    self.gym.set_dof_position_target_tensor(self.sim,
                                            gymtorch.unwrap_tensor(self.yumi_dof_targets))


# parameters ==================================================
# camera
location = gymapi.Vec3(0.2, 0.1, 0.8)
lookat = gymapi.Vec3(0.4, 0, 0.1)
width = 472
height = 472
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
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
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
yumi_asset = load_assets.load_yumi(gym, sim, asset_root, asset_file)

# configure yumi dofs
yumi_dof_props, default_dof_pos, default_dof_state = compute.compute_yumi_default_dof(gym, yumi_asset)

# get link index of panda hand, which we will use as end effector
yumi_link_dict = gym.get_asset_rigid_body_dict(yumi_asset)  # return 包含主体名称和资产相关索引之间映射的字典
yumi_hand_index = yumi_link_dict["gripper_r_base"]

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
num_envs = 64
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
yumi_handles = []

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
    yumi_handles.append(yumi_handle)
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
    # down_states = [28.67, -67.79, -19.64, -79.44, 117.26, -19.22, -1.01, 0.025, 0.025]
    down_states = [0.5003859, -1.1831587, -0.01762783, -0.34278267, -1.38648956, 2.04657308, -0.33545228, 0.025, 0.025]
    gym.set_actor_dof_states(env, yumi_handle, down_states, gymapi.STATE_ALL)
    # set initial position targets
    gym.set_actor_dof_position_targets(env, yumi_handle, down_states)

    # get inital hand pose
    hand_handle = gym.find_actor_rigid_body_handle(env, yumi_handle, "gripper_r_base")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    # get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env, yumi_handle, "gripper_r_base", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

default_cube_states = to_torch(default_cube_states, device=device, dtype=torch.float).view(num_envs, 13)  # view功能相当于reshape

_cube_idx = cube_idxs

# point camera at middle env
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
j_eef = jacobian[:, yumi_hand_index - 1, :, :7]

# get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, 9, 1)
print(type(dof_pos))
# exit()

# get root tensor
_root_tensor = gym.acquire_actor_root_state_tensor(sim)
root_tensor = gymtorch.wrap_tensor(_root_tensor)

# Create a tensor noting whether the hand should return to the initial position
hand_restart = torch.full([num_envs], False, dtype=torch.bool).to(device)

# simulation loop
time = 0

global_indices = torch.arange(num_envs * 3, dtype=torch.int32, device=device).view(num_envs, -1)

if printInfo.print_cube_idxs:
    print("cube_idxs:")
    print(cube_idxs[63])
    print(cube_idxs)

if printInfo.print_start:
    print("====================================== Simulation start ======================================")

while not gym.query_viewer_has_closed(viewer):
    time += 1
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # communicate physics to graphics system
    gym.step_graphics(sim)

    # render the camera sensors
    gym.render_all_camera_sensors(sim)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    # if time == 1:
    #     for handle in range(len(yumi_handles)):
    #         down_states = [0.5003859, -1.1831587, -0.01762783, -0.34278267, -1.38648956, 2.04657308, -0.33545228, 0., 0.]
    #         gym.set_actor_dof_states(env, handle, down_states, gymapi.STATE_ALL)


    # test actions
    # ================================================================================================
    # ===================================================================
    # Maximum velocity: [0:7]4.e+02  [7:9]1.e+06
    # actions:
    # 1 其他不变，z下降1cm 到一定高度reset
    # 2 其他不变 x, y 随机【-0.01，0.01】
    # 3 其他不变，yaw随机旋转
    # 4 其他不变，夹爪随机开合
    # ===================================================================
    # Set action tensors
    # action = torch.zeros_like(dof_pos, device=device)
    action = dof_pos.contiguous()
    action = action.squeeze(-1)
    # initial hand position and orientation tensors
    init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
    init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)
    # # hand orientation for grasping
    # down_q = torch.stack(num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view(
    #     (num_envs, 4))
    # ==========================================================================================
    # action1: 其他不变，z下降1cm TODO:到一定高度reset
    pos_err1 = torch.Tensor([0, 0, -0.01]).to(device).repeat(num_envs).view(num_envs, 3)  # shape [num_envs, 3]
    orn = torch.Tensor([1, 0, 0, 0]).to(device).repeat(num_envs).view(num_envs, 4)
    orn_err1 = orientation_error(orn, orn)
    dpose1 = torch.cat([pos_err1, orn_err1], -1).unsqueeze(-1)
    # ==========================================================================================
    # action2: 其他不变 x, y 随机【-0.01，0.01】
    # pos_err2_xy = 0.2 * (torch.rand((num_envs, 2), device=device) - 0.5)
    pos_err2_xy = 0.1 * (torch.rand((num_envs, 2), device=device))

    pos_err2_z = torch.Tensor([0.]).to(device).repeat(num_envs).view(num_envs, 1)
    pos_err2 = torch.cat([pos_err2_xy, pos_err2_z], -1).view(num_envs, 3)
    orn = torch.Tensor([1, 0, 0, 0]).to(device).repeat(num_envs).view(num_envs, 4)
    orn_err2 = orientation_error(orn, orn)
    dpose2 = torch.cat([pos_err2, orn_err2], -1).unsqueeze(-1)
    # ==========================================================================================
    # action3: 其他不变，yaw旋转
    # 最大角度: max_angle（弧度）
    # 四元数[cos(angle/2), 0, 0, sin(angle/2)]
    # cos(angle/2)  [0, max_cos]
    # sin(angle/2)  [-max_sin, max_sin]
    max_angle = np.pi/12.0
    max_cos = np.cos(max_angle/2)
    max_sin = np.sin(max_angle/2)
    rand_cos_tensor = max_cos * torch.rand(num_envs, device=device).view(num_envs, 1)
    # rand_sin_tensor = 2 * max_sin * torch.rand(num_envs, device=device).view(num_envs, 1) - max_sin
    rand_sin_tensor = max_sin * torch.rand(num_envs, device=device).view(num_envs, 1)
    tensor0 = torch.Tensor([0.]).to(device).repeat(num_envs).view(num_envs, 1)
    pos_err3 = torch.Tensor([0, 0, 0.]).to(device).repeat(num_envs).view(num_envs, 3)  # shape [num_envs, 3]
    orn0 = torch.Tensor([1, 0, 0, 0]).to(device).repeat(num_envs).view(num_envs, 4)
    goal_orn = torch.cat([rand_cos_tensor, rand_sin_tensor, tensor0, tensor0], -1)
    orn_err3 = orientation_error(goal_orn, orn0)
    dpose3 = torch.cat([pos_err3, orn_err3], -1).unsqueeze(-1)
    # ==========================================================================================
    # action4: 其他不变，夹爪随机开合
    gripper_rand = torch.rand(num_envs, device=device).view(num_envs, 1)
    gripper_action = torch.where(gripper_rand > 0.5,
                                 torch.Tensor([[0., 0.]] * num_envs).to(device),
                                 torch.Tensor([[0.025, 0.025]] * num_envs).to(device))
    gripper_action = gripper_action.view(num_envs, 2)

    # choose action
    # ==========================================================================================
    # action1-3
    dpose = dpose3  # TODO:choose action
    action[:, :7] = dof_pos.squeeze(-1)[:, :7] + control_ik(dpose, device, j_eef, num_envs)
    # # action4
    # action[:, 7:9] = gripper_action
    # Deploy actions
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(action))

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
