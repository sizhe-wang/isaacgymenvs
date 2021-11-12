import os
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from myutils.print_info import printInfo

import math
import numpy as np
import torch


def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats


def compute_yumi_default_dof(gym, yumi_asset):
    # 获取dof限制，默认状态夹爪为最大值（张开），其余为中值
    yumi_dof_props = gym.get_asset_dof_properties(yumi_asset)
    # printInfo
    if printInfo.print_yumi_dof_props:
        print("yumi_dof_props:")
        print(yumi_dof_props)
        printInfo.print_yumi_dof_props = False
    yumi_lower_limits = yumi_dof_props["lower"]
    yumi_upper_limits = yumi_dof_props["upper"]
    yumi_ranges = yumi_upper_limits - yumi_lower_limits
    # yumi_mids = 0.3 * (yumi_upper_limits + yumi_lower_limits)
    yumi_mids = yumi_lower_limits + 0.5 * yumi_ranges

    # 设置驱动模式为position control
    yumi_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)

    # franka的设置，不这样设置机器人就一直不动
    yumi_dof_props["stiffness"][:7].fill(400.0)
    yumi_dof_props["damping"][:7].fill(40.0)
    # grippers
    yumi_dof_props["stiffness"][7:].fill(800.0)
    yumi_dof_props["damping"][7:].fill(40.0)

    # 默认dof
    yumi_num_dofs = gym.get_asset_dof_count(yumi_asset)
    default_dof_pos = np.zeros(yumi_num_dofs, dtype=np.float32)
    default_dof_pos[:7] = yumi_mids[:7]
    # grippers open
    default_dof_pos[7:] = yumi_upper_limits[7:]
    # 前7个自由度取中值，后两个为夹爪，默认最大值，即为open

    default_dof_state = np.zeros(yumi_num_dofs, gymapi.DofState.dtype)
    default_dof_state["pos"] = default_dof_pos
    # printInfo
    if printInfo.print_default_dof_state:
        print("default_dof_state")
        print(default_dof_state)
        printInfo.print_default_dof_state = False
    if printInfo.print_default_dof_pos:
        print("default_dof_pos")
        print(default_dof_pos)
        printInfo.print_default_dof_pos = False
    return yumi_dof_props, default_dof_pos, default_dof_state
