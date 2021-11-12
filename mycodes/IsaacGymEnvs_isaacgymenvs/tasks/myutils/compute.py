import os
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.myutils.print_info import printInfo

import math
import numpy as np
import torch


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
    yumi_dof_props["damping"][:7].fill(80.0)
    # grippers
    yumi_dof_props["stiffness"][7:].fill(1.0e6)
    yumi_dof_props["damping"][7:].fill(1.0e2)

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
    return default_dof_pos, default_dof_state
