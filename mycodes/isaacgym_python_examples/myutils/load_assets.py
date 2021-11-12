import os
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from myutils.print_info import printInfo

import math
import numpy as np
import torch


def load_from_asset(gym, sim, asset_root, asset_file):
    asset_options = gymapi.AssetOptions()
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    if printInfo.print_load_success:
        print("Successfully loaded asset")
    return asset


def load_yumi(gym, sim, asset_root, asset_file):
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.01  # 添加到所有资产刚体/链接的惯性张量的对角线元素的值。可以提高模拟稳定性
    asset_options.fix_base_link = True  # 导入时将资产基础设置为固定位置
    asset_options.disable_gravity = True
    asset_options.flip_visual_attachments = False  # 将网格从 Z-up 左手坐标系切换到 Y-up 右手坐标系。

    # V-HACD
    asset_options.vhacd_enabled = True  # 默认情况下，禁用凸分解
    vhacd_params = gymapi.VhacdParams()
    vhacd_params.resolution = 1000000  # 体素化阶段生成的最大体素数。10,000-64,000,000。默认 100,000。
    asset_options.vhacd_params = vhacd_params
    # 单击查看器 GUI 中的“查看器”选项卡并启用“渲染碰撞网格”复选框来查看凸分解的结果。
    # 如果网格具有表示凸分解的子网格，Gym 可以将子网格加载为资产中的单独形状。要启用此功能，请使用
    # asset_options.convex_decomposition_from_submeshes = True。
    yumi_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    if printInfo.print_load_success:
        print("Successfully loaded yumi")
    return yumi_asset


def load_cube(gym, sim, cube_size):
    asset_options = gymapi.AssetOptions()
    cube_asset = gym.create_box(sim, cube_size, cube_size, cube_size, asset_options)
    if printInfo.print_load_success:
        print("Successfully loaded cube")
    return cube_asset

def load_table(gym, sim, table_dims):
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
    if printInfo.print_load_success:
        print("Successfully loaded table")
    return table_asset
