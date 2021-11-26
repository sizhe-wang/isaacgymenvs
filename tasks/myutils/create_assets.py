import os
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.myutils.print_info import printInfo

import math
import numpy as np
import torch


# ============================================================================================
# create_camera
# 创建一个相机并返回其handle
# 参数：
#       env: 来自gym.create_env(sim, env_lower, env_upper, num_per_row)
#       location:  the coordinates of the camera in the environment’s local coordinate frame
#                  格式：gymapi.Vec3()
#       lookat: the coordinates of the point the camera is looking at,
#               again in the environments local coordinates.
#               格式：gymapi.Vec3()
#       width, height: 像素
#
# return：创建相机的 handle
# ============================================================================================
def create_camera(gym, env, location, lookat, width, height):
    camera_props = gymapi.CameraProperties()
    camera_props.width = width
    camera_props.height = height
    camera_props.enable_tensors = True
    camera_props.horizontal_fov = 50.0
    camera_handle = gym.create_camera_sensor(env, camera_props)
    gym.set_camera_location(camera_handle, env, location, lookat)
    if printInfo.print_create_success:
        print("Successfully created camera")
    return camera_handle


def create_camera_attach(gym, env, width, height, body_handle):
    camera_props = gymapi.CameraProperties()
    camera_props.width = width
    camera_props.height = height
    camera_props.enable_tensors = True
    camera_props.horizontal_fov = 75.0
    local_transform = gymapi.Transform()
    local_transform.p.x = 0
    local_transform.p.y = -0.04
    local_transform.p.z = 0.05
    local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(-90.0))
    camera_handle = gym.create_camera_sensor(env, camera_props)
    gym.attach_camera_to_body(camera_handle, env, body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

    if printInfo.print_create_success:
        print("Successfully created camera")
    return camera_handle

# ============================================================================================
# create_yumi
# 创建一个yumi并返回其handle
# 参数：
#       gym:
#       env:
#       yumi_asset:
#       i: 创建env时的计数器，即创建env循环的i
# return：创建相机的 handle
# ============================================================================================
def create_yumi(gym, env, yumi_asset, i):
    yumi_pose = gymapi.Transform()
    yumi_pose.p = gymapi.Vec3(0.35, 0, 0.33)
    yumi_handle = gym.create_actor(env, yumi_asset, yumi_pose, "yumi", i, 2)
    if printInfo.print_create_success:
        print("Successfully created yumi")
    return yumi_handle, yumi_pose


# ============================================================================================
# create_cube
# 创建一个方块并返回其handle
# 参数：
#       gym:
#       env:
#       cube_middle:方块可初始位置的中心，真实初始位置为cube_middle在x,y方向上各加一随机数，z方向上刚好和桌面接触
#                   (gymapi.Vec3())
#       cube_size:  方块的棱长   (float)
#       cube_asset:
#       table_dims: 桌子的尺寸   (gymapi.Vec3())
#       i: 创建env时的计数器，即创建env循环的i
# return：创建相机的 handle
# ============================================================================================
def create_cube(gym, env, cube_middle, cube_size, cube_asset, table_dims, i):
    cube_pose = gymapi.Transform()
    # cube_pose.p.x = cube_middle.p.x + np.random.uniform(-0.1, 0.1)
    # cube_pose.p.y = cube_middle.p.y + np.random.uniform(-0.1, 0.1)
    cube_pose.p.x = cube_middle.p.x
    cube_pose.p.y = cube_middle.p.y
    cube_pose.p.z = table_dims.z + 0.5 * cube_size
    cube_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    cube_handle = gym.create_actor(env, cube_asset, cube_pose, "cube", i, 0)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
    cube_properties = gymapi.RigidBodyProperties()
    cube_properties.mass = 0.25  # kg
    gym.set_actor_rigid_body_properties(env, cube_handle, [cube_properties], False)
    if printInfo.print_create_success:
        print("Successfully created cube")
    return cube_handle, cube_pose


# ============================================================================================
# create_table
# 创建一个桌子并返回其handle
# 参数：
#       gym:
#       env:
#       table_dims: 桌子的尺寸   (gymapi.Vec3())
#       table_asset:
#       i: 创建env时的计数器，即创建env循环的i
# return：创建相机的 handle
# ============================================================================================
def create_table(gym, env, table_dims, table_asset, i):
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.35, 0, 0.5 * table_dims.z)
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)
    if printInfo.print_create_success:
        print("Successfully created table")
    return table_handle, table_pose
