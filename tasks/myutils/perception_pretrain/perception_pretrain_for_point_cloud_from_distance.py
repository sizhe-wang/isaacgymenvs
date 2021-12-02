# main perception pretrain module
# train with point cloud
import os
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.tasks.myutils import load_assets
from isaacgymenvs.tasks.myutils import create_assets
from isaacgymenvs.tasks.myutils import compute
from isaacgymenvs.tasks.myutils import save
from isaacgymenvs.tasks.myutils import log
from isaacgymenvs.tasks.myutils.print_info import printInfo
from scipy.spatial.transform import Rotation as R
from PIL import Image
import yaml
import hydra

import math
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision import models
import time
from torchgeometry.core import conversions
from isaacgymenvs.tasks.myutils.perception_pretrain.resent_for_perception_pretrain import ResNet18, ResNet34
from autolab_core import RigidTransform
from autolab_core import DepthImage, CameraIntrinsics
from torch.utils.tensorboard import SummaryWriter
import trimesh
from skimage.util import random_noise


# ========================================================
# 根据末端执行器的微分运动计算机器人关节的微分运动
# dpose: 末端执行器的微分运动 delta的[p.x, p.y, p.z, rx, ry, rz]
# u: 关节的微分运动 （只考虑arm，不考虑finger，所以是7个dof）
# j_eef：arm的雅可比矩阵
# ========================================================
def control_ik(dpose, device, j_eef, num_envs, damping=0.05):
    # solve damped least squares
    # print("dpose", dpose)
    # print("j_eef", j_eef)
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)    # torch.eye(6) 6维单位阵
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)    # @ 矩阵-向量乘法
    return u


class YumiCube(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        # 一些基础配置
        self.cfg = cfg
        self.headless = headless
        self.env_spacing = self.cfg["env"]['envSpacing']
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.num_cubes = self.cfg["env"]["numCubes"]
        self.height_reward_scale = self.cfg["env"]["heightRewardScale"]
        self.lift_reward_scale = self.cfg["env"]["liftRewardScale"]

        self.lift_height = self.cfg["env"]["liftHeight"]

        self.xyz_scale = self.cfg["env"]["xyzScale"]
        self.rz_scale = self.cfg["env"]["rzScale"]
        self.pretrained = self.cfg["env"]["pretrained"]
        self.cube_random = self.cfg["env"]["cubeRandom"]
        self.gripper_random = self.cfg["env"]["gripperRandom"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.real_feature_input = self.cfg["env"]["realFeatureInput"]
        self.have_gravity = self.cfg["env"]["haveGravity"]
        self.image_mode = self.cfg["env"]["imageMode"]  # 0: depth repeat to 3 channels    1: organized point cloud
        self.add_noise_to_image = self.cfg["env"]["addNoiseToImage"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.dt = 1 / 60.

        self._num_obs = 11
        self._num_acts = 5

        self.cfg["env"]["numObservations"] = self._num_obs
        self.cfg["env"]["numActions"] = self._num_acts
        # add ===================================================================
        # table
        self.table_dims = gymapi.Vec3(0.7, 0.7, 0.1)
        # 关于相机
        self.camera_same = self.cfg["env"]["camera"]["same"]
        self.camera_width = self.cfg["env"]["camera"]["width"]
        self.camera_height = self.cfg["env"]["camera"]["height"]
        self.camera_horizontal_fov = self.cfg["env"]["camera"]["horizontal_fov"]
        self.camera_vertical_fov = self.camera_height / self.camera_width * self.camera_horizontal_fov
        self.camera_location = gymapi.Vec3(0.35, 0.65, 0.75)
        self.camera_lookat = gymapi.Vec3(0.35, 0., self.table_dims.z)

        # cube
        self.cube_size = 0.035
        self.cube_spacing = self.cube_size + 0.005
        self.cube_middle = gymapi.Transform()
        self.cube_middle.p = gymapi.Vec3(0.35, 0., self.table_dims.z + self.cube_size / 2. + 0.0025)

        # object reset space
        self.object_space_lower = gymapi.Vec3(0.05, -0.25, 0.)
        self.object_space_upper = gymapi.Vec3(0.65, 0.25, self.table_dims.z + self.cube_size / 2. + 0.0025)

        # prepare some lists
        self.envs = []
        self.cube_idxs = []

        # rot:"intrinsic rotations" or "extrinsic rotations"
        self.extrinsic_rotations = self.cfg["env"]["extrinsicRotations"]

        # about control_ik
        self.damping = 0.0

        # /add ===================================================================
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.yumi_default_dof_pos = to_torch([0, 0, 0, 0, 0.025, 0.025], device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # 2: 0:position, 1:velocity
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.yumi_dof_pos = self.dof_state[:, 0].view(self.num_envs, self.num_dofs)
        self.yumi_dof_vel = self.dof_state[:, 1].view(self.num_envs, self.num_dofs)

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)

        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)
        # 3: 0:yumi, 1:table, 2:cube

        # add ===================================================================
        self.cube_states = self.root_state_tensor[:, 2].view(self.num_envs, 1, 13)  # TODO: 确定cube在env中的序号是不是2: 是
        self.yumi_states = self.root_state_tensor[:, 0].view(self.num_envs, 1, 13)
        # /add ===================================================================

        self.yumi_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        # image
        self.net = ResNet34(num_classes=6).to(self.device)
        self.net.create_optimzer()
        # self.net.create_scheduler(milestones=[100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000], gamma=0.1)
        self.net.create_scheduler(milestones=[50, 100, 300], gamma=0.1)
        self.net.train()

        # self.preprocess = transforms.Compose([  # [1]
        #     # transforms.Resize(472),                    #[2]
        #     # transforms.CenterCrop(472),                #[3]
        #     # transforms.ToTensor(),                     #[4]
        #     transforms.Normalize(  # [5]
        #         mean=[0.485, 0.456, 0.406],  # [6]
        #         std=[0.229, 0.224, 0.225]  # [7]
        #     )])
        self.render_img = True


        # /add ===================================================================
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.success_ids = []

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81) if self.have_gravity else gymapi.Vec3(0.0, 0.0, 0.0)
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.env_spacing, int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../yumi_description")
        yumi_asset_file = "/urdf/yumi.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            yumi_asset_file = self.cfg["env"]["asset"].get("assetFileNameYumi", yumi_asset_file)

        # load yumi asset
        yumi_asset = load_assets.load_yumi(self.gym, self.sim, asset_root, yumi_asset_file)
        # self.yumi_asset = yumi_asset

        # load table asset
        table_asset = load_assets.load_table(self.gym, self.sim, self.table_dims)

        # load cube asset
        cube_asset = load_assets.load_cube(self.gym, self.sim, self.cube_size)

        yumi_dof_stiffness = to_torch([1.0e6, 1.0e6, 1.0e6, 20000, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
        yumi_dof_damping = to_torch([100, 100, 100, 100, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        self.num_yumi_bodies = self.gym.get_asset_rigid_body_count(yumi_asset)
        self.num_yumi_dofs = self.gym.get_asset_dof_count(yumi_asset)

        if printInfo.print_num_yumi_bodies_dofs:
            print("num yumi bodies: ", self.num_yumi_bodies)
            print("num yumi dofs: ", self.num_yumi_dofs)

        # set yumi dof properties
        yumi_dof_props = self.gym.get_asset_dof_properties(yumi_asset)
        self.yumi_dof_lower_limits = []
        self.yumi_dof_upper_limits = []
        for i in range(self.num_yumi_dofs):
            yumi_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                yumi_dof_props['stiffness'][i] = yumi_dof_stiffness[i]
                yumi_dof_props['damping'][i] = yumi_dof_damping[i]
            else:
                yumi_dof_props['stiffness'][i] = 7000.0
                yumi_dof_props['damping'][i] = 50.0

            self.yumi_dof_lower_limits.append(yumi_dof_props['lower'][i])
            self.yumi_dof_upper_limits.append(yumi_dof_props['upper'][i])

        self.yumi_dof_lower_limits = to_torch(self.yumi_dof_lower_limits, device=self.device)
        self.yumi_dof_upper_limits = to_torch(self.yumi_dof_upper_limits, device=self.device)
        self.yumi_dof_speed_scales = torch.ones_like(self.yumi_dof_lower_limits)

        self.yumi_dof_speed_scales[[4, 5]] = 0.1
        yumi_dof_props['effort'][4] = 50
        yumi_dof_props['effort'][5] = 50


        # compute aggregate size
        num_yumi_bodies = self.gym.get_asset_rigid_body_count(yumi_asset)
        num_yumi_shapes = self.gym.get_asset_rigid_shape_count(yumi_asset)
        num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)
        num_cube_bodies = self.gym.get_asset_rigid_body_count(cube_asset)
        num_cube_shapes = self.gym.get_asset_rigid_shape_count(cube_asset)
        max_agg_bodies = num_yumi_bodies + num_table_bodies + self.num_cubes * num_cube_bodies + 10
        max_agg_shapes = num_yumi_shapes + num_table_shapes + self.num_cubes * num_cube_shapes + 10

        self.yumis = []
        self.tables = []
        self.default_cube_states = []
        self.default_yumi_states = []
        self.cube_start = []
        self.envs = []
        self.cubes = []
        self.cameras = []
        self.init_pos_list = []
        self.init_rot_list = []
        self.hand_idxs = []
        self.yumi_indices = []
        self.cube_indices =[]

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            yumi_actor, yumi_start_pose = create_assets.create_yumi(self.gym, env_ptr, yumi_asset, i, self.gripper_random)
            self.gym.set_actor_dof_properties(env_ptr, yumi_actor, yumi_dof_props)
            yumi_idx = self.gym.get_actor_index(env_ptr, yumi_actor, gymapi.DOMAIN_SIM)
            self.yumi_indices.append(yumi_idx)
            # get inital hand pose
            hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, yumi_actor, "gripper_r_base")
            hand_pose = self.gym.get_rigid_transform(env_ptr, hand_handle)
            self.default_yumi_states.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z,
                                             hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w,
                                             0, 0, 0, 0, 0, 0])

            self.init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            self.init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])
            # get global index of hand in rigid body state tensor
            hand_idx = self.gym.find_actor_rigid_body_index(env_ptr, yumi_actor, "gripper_r_base", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)

            table_actor, table_start_pose = create_assets.create_table(self.gym, env_ptr, self.table_dims, table_asset, i)

            if self.num_cubes > 0:
                self.cube_start.append(self.gym.get_sim_actor_count(self.sim))
                cube_actor, cube_pose = create_assets.create_cube(self.gym, env_ptr, self.cube_middle,
                                                                   self.cube_size, cube_asset, self.table_dims, i, self.cube_random)
                self.default_cube_states.append([cube_pose.p.x, cube_pose.p.y, cube_pose.p.z,
                                                 cube_pose.r.x, cube_pose.r.y, cube_pose.r.z, cube_pose.r.w,
                                                 0, 0, 0, 0, 0, 0])
                self.cubes.append(cube_actor)
                cube_idx = self.gym.get_actor_index(env_ptr, cube_actor, gymapi.DOMAIN_SIM)
                self.cube_indices.append(cube_idx)

            # camera_actor = create_assets.create_camera(self.gym, env_ptr, self.camera_location, self.camera_lookat,
            #                                            self.camera_width, self.camera_height)

            camera_actor = create_assets.create_camera(self.gym, env_ptr, self.camera_location, self.camera_lookat, self.camera_width, self.camera_height, self.camera_horizontal_fov)

            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)
            self.yumis.append(yumi_actor)
            self.tables.append(table_actor)
            self.cameras.append(camera_actor)
            print("create%d"%i)

        self.yumi_indices = to_torch(self.yumi_indices, dtype=torch.long, device=self.device)
        self.cube_indices = to_torch(self.cube_indices, dtype=torch.long, device=self.device)
        # 无所谓找的是哪个环境的，所有环境里都一样，指的是某个关节在actor里的编号
        # print("create end")
        # self.hand_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.yumis[0], "yumi_link_7_r")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.yumis[0], "gripper_r_finger_l")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.yumis[0], "gripper_r_finger_r")
        self.default_cube_states = to_torch(self.default_cube_states, device=self.device, dtype=torch.float
                                            ).view(self.num_envs, self.num_cubes, 13)
        self.default_yumi_states = to_torch(self.default_yumi_states, device=self.device, dtype=torch.float
                                            ).view(self.num_envs, 1, 13)
        # print("made default cube states")
        # self.init_data()
        # print("init data finished")

    def compute_reward(self):

        cube_index = self.gym.find_actor_rigid_body_index(self.envs[0], self.cubes[0], "box", gymapi.DOMAIN_ENV)

        hand_index = self.hand_idxs[0]

        object_height = self.rigid_body_states[:, cube_index, 2].view(self.num_envs, 1)
        object_x = self.rigid_body_states[:, cube_index, 0].view(self.num_envs, 1)
        object_y = self.rigid_body_states[:, cube_index, 1].view(self.num_envs, 1)
        gripper_x = self.rigid_body_states[:, hand_index, 0].view(self.num_envs, 1)
        gripper_y = self.rigid_body_states[:, hand_index, 1].view(self.num_envs, 1)
        gripper_height = self.rigid_body_states[:, hand_index, 2].view(self.num_envs, 1)

        rewards = torch.zeros(self.num_envs, device=self.device)

        offset_height = 0.14
        diff_height = offset_height - self.cube_size / 2.
        offset_gripper_pos = torch.cat([self.rigid_body_states[:, hand_index, :2], (self.rigid_body_states[:, hand_index, 2].unsqueeze(-1) - diff_height)], dim=-1)

        # rewards for distance
        d = torch.norm(self.rigid_body_states[:, cube_index, :3] - offset_gripper_pos, dim=-1)
        # d = torch.norm(self.rigid_body_states[:, cube_index, :3] - self.rigid_body_states[:, hand_index, :3], dim=-1)
        dist_reward = 1.0 / (1.0 + (10 * d) ** 2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.01, dist_reward * 2, dist_reward)
        rewards += dist_reward
        # print("dist_reward", dist_reward[0])
        # rewards for euler_z ======================================
        # rot_cube_euler_z = self.get_cube_euler_z().squeeze(-1)
        # rot_gripper_euler_z = self.get_gripper_euler_z().squeeze(-1)
        #
        # euler_z_diff = abs(rot_cube_euler_z - rot_gripper_euler_z)
        # # angle_reward = 1.0 / (1.0 + (10 * (euler_z_diff - 0.1)) ** 2)
        # # angle_reward *= angle_reward
        # # angle_reward = torch.where(euler_z_diff <= 0.005, dist_reward * 2, dist_reward)
        # # rewards += angle_reward
        # angle_reward = euler_z_diff
        # rewards -= angle_reward
        # # print("angle reward", angle_reward[0])
        # ============================================================
        # bool for around enough
        around = (abs(gripper_x - object_x) - 0.01 < 0) & (abs(gripper_y - object_y) - 0.01 < 0) & (abs(gripper_height - object_height - diff_height) - 0.02 < 0)

        # rewards for lift height
        rewards += ((object_height - self.table_dims.z - self.cube_size / 2.) * self.lift_reward_scale * around).view(self.num_envs)
        # rewards += ((gripper_height - self.table_dims.z - self.cube_size / 2. - diff_height) * self.lift_reward_scale * 0.1).view(self.num_envs)
        print("lift height", (object_height - self.table_dims.z - self.cube_size / 2.)[0])

        success_ = torch.where((object_height > (self.table_dims.z + self.cube_size / 2.)) & around,
                               torch.Tensor([[1.]] * self.num_envs).to(self.device),
                               torch.Tensor([[0.]] * self.num_envs).to(self.device)).view(self.num_envs)
        self.success_ids = success_.nonzero(as_tuple=False).squeeze(-1).tolist()
        # bonus for lift height.  bonus需要很大，否则train不出来(只有现在的1/20的时候就不行) max episode length需要大一点(300)
        # 任务越复杂越需要更大的bonus和max episode length，比如cube在中心的时候，bonus是现在的1/2，max episode length也是1/2，
        # 但cube不在中心，就得加大bonus和max episode length，否则gripper不会lift
        # 在max episode length变大的时候bonus要相应变大，否则bonus不明显
        # rewards += torch.where((object_height > (self.table_dims.z + self.cube_size / 2.)) & around,
        #                        (5 * dist_reward).view(self.num_envs, 1),
        #                        torch.Tensor([[0.]] * self.num_envs).to(self.device)).view(self.num_envs)
        # rewards += torch.where((object_height > (self.table_dims.z + self.cube_size / 2.) + 0.0001) & around,
        #                        (10 * dist_reward).view(self.num_envs, 1),
        #                        torch.Tensor([[0.]] * self.num_envs).to(self.device)).view(self.num_envs)
        # rewards += torch.where((object_height > (self.table_dims.z + self.cube_size / 2.) + 0.0002) & around,
        #                        (15 * dist_reward).view(self.num_envs, 1),
        #                        torch.Tensor([[0.]] * self.num_envs).to(self.device)).view(self.num_envs)
        # rewards += torch.where((object_height > (self.table_dims.z + self.cube_size / 2.) + 0.0003) & around,
        #                        (20 * dist_reward).view(self.num_envs, 1),
        #                        torch.Tensor([[0.]] * self.num_envs).to(self.device)).view(self.num_envs)
        # rewards += torch.where((object_height > (self.table_dims.z + self.cube_size / 2.) + 0.0005) & around,
        #                        (25 * dist_reward).view(self.num_envs, 1),
        #                        torch.Tensor([[0.]] * self.num_envs).to(self.device)).view(self.num_envs)
        # rewards += torch.where((object_height > (self.table_dims.z + self.cube_size / 2.) + 0.0008) & around,
        #                        (30 * dist_reward).view(self.num_envs, 1),
        #                        torch.Tensor([[0.]] * self.num_envs).to(self.device)).view(self.num_envs)
        rewards += torch.where((object_height > (self.table_dims.z + self.cube_size / 2.) + 0.01) & around,
                               torch.Tensor([[20.]] * self.num_envs).to(self.device),
                               torch.Tensor([[0.]] * self.num_envs).to(self.device)).view(self.num_envs)
        rewards += torch.where((object_height > (self.table_dims.z + self.cube_size / 2.) + 0.03),
                               torch.Tensor([[30.]] * self.num_envs).to(self.device),
                               torch.Tensor([[0.]] * self.num_envs).to(self.device)).view(self.num_envs)
        rewards += torch.where((object_height > (self.table_dims.z + self.cube_size / 2.) + 0.05),
                               torch.Tensor([[40.]] * self.num_envs).to(self.device),
                               torch.Tensor([[0.]] * self.num_envs).to(self.device)).view(self.num_envs)



        # print("=========================success: %d===============================" % torch.sum(success))
        # print("cube", object_x[0], object_y[0], object_height[0])
        # print("gripper", gripper_x[0], gripper_y[0], gripper_height[0])
        # rewards += torch.where(success, torch.Tensor([[100.]] * self.num_envs).to(self.device),
        #                        torch.Tensor([[0.]] * self.num_envs).to(self.device)).view(self.num_envs)
        # penalty
        # rewards -= self.action_penalty_scale
        # penalty actions
        rewards -= torch.norm(self.control_output, dim=-1) * 0.5
        # print("action penalty", torch.norm(self.control_output, dim=-1)[0] * 0.5)

        # self.reset_buf = torch.where(success.view(self.num_envs), torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
        self.rew_buf = rewards
        # print("reward", rewards)
        # print('================================================')
        # return success_ids

    def compute_observations(self, i):
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.step_graphics(self.sim)

        # =============================================================================================
        # cube state [x, y, z, rx, ry, rz]
        cube_index = self.gym.find_actor_rigid_body_index(self.envs[0], self.cubes[0], "box", gymapi.DOMAIN_ENV)
        cube_xyz = self.rigid_body_states[:, cube_index, :3].view(self.num_envs, 3)
        rot_cube = self.get_cube_euler()
        # cube_state = torch.cat([cube_xyz, rot_cube], dim=-1)

        # =============================================================================================
        # convert cube rz to [0, pi/2]
        rot_cube_z = rot_cube[:, 2].unsqueeze(-1)
        # [-pi, pi] --> [-pi, pi/2]
        rot_cube_z = torch.where(rot_cube_z > np.pi / 2, rot_cube_z - np.pi / 2., rot_cube_z)
        # [-pi, pi/2] --> [-pi/2, pi/2]
        rot_cube_z = torch.where(rot_cube_z < 0, rot_cube_z + np.pi / 2., rot_cube_z)
        # [-pi/2, pi/2] --> [0, pi/2]
        rot_cube_z = torch.where(rot_cube_z < 0, rot_cube_z + np.pi / 2., rot_cube_z)
        cube_state = torch.cat([cube_xyz, rot_cube[:, :2], rot_cube_z], dim=-1)
        # =============================================================================================
        # =============================================================================================
        # gripper state [x, y, z, rz, width]
        # gripper_pos = self.rigid_body_states[:, self.hand_idxs[0], :3].view(self.num_envs, 3)
        # rot_gripper_z = self.get_gripper_euler_z()
        # gripper_width = self.yumi_dof_pos[:, 4] + self.yumi_dof_pos[:, 5]    # 两个gripper的dof都是正数[0, 0.025]
        # gripper_width = gripper_width.view(self.num_envs, 1)
        # gripper_state = torch.cat([gripper_pos, rot_gripper_z, gripper_width], dim=-1)

        # =============================================================================================
        # resnet input data  shape:[num_envs, 3, 256, 256]
        # resnet target data shape:[num_envs, 6]
        # resnet output shape:[num_envs, 5]

        # render the camera sensors
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        #
        # User code to digest tensors
        #
        # get image tensor
        image_tensors = []
        show_image = False
        for j in range(self.num_envs):
            # for j in self.success_ids:
            _image_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[j], self.cameras[j], gymapi.IMAGE_DEPTH)

            # H * W * 3
            image_tensor = gymtorch.wrap_tensor(_image_tensor)
            image_array = image_tensor.cpu().numpy()  # image_array is depth map, not distance map

            # add noise to image
            if self.add_noise_to_image:
                image_array = random_noise(image_array, mode='gaussian', mean=0, var=0.0001, clip=False)
                image_tensor = torch.Tensor(image_array).to(self.device)
            # cal unorganized point cloud =======================================================================
            point_data_unorganized = None
            if show_image:
                depth_image = DepthImage(image_array)
                # camera intrinsics
                camera_intrinsics = CameraIntrinsics(frame='unspecified',
                                                     fx=self.camera_width / (2 * np.tan(np.deg2rad(self.camera_horizontal_fov) / 2.)),
                                                     fy=self.camera_height / (2 * np.tan(np.deg2rad(self.camera_vertical_fov) / 2.)),
                                                     cx=(self.camera_width - 1.0) / 2.,
                                                     cy=(self.camera_height - 1.0) / 2.,
                                                     height=self.camera_height,
                                                     width=self.camera_width,
                                                     skew=0)

                point_normal = depth_image.point_normal_cloud(camera_intr=camera_intrinsics)

                # # TODO: choose point data: whether organized
                point_data_unorganized = point_normal.point_cloud.data.transpose(1, 0)
            # ====================================================================================================
            point_data_organized = self.depth2pointcloud(image_array).reshape(-1, 3)
            
            point_data_tensor = torch.Tensor(point_data_organized).to(self.device)  # [65536, 3]


            if show_image and j == 0:
                # image_array = image_tensor.cpu().numpy()    # image_array is depth map, not distance map

                # -inf implies no depth value, set it to zero. output will be black.
                image_array[image_array == -np.inf] = 0

                # clamp depth image to 10 meters to make output image human friendly
                image_array[image_array < -10] = -10

                # flip the direction so near-objects are light and far objects are dark
                image_array = -255.0 * (image_array / np.min(image_array + 1e-4))

                # Convert to a pillow image and show
                normalized_depth_image = Image.fromarray(image_array.astype(np.uint8), mode="L")
                normalized_depth_image.show()

                pc_1 = trimesh.PointCloud(point_data_unorganized, colors=[0, 255, 0])
                pc_2 = trimesh.PointCloud(point_data_organized, colors=[0, 0, 255])
                scene = trimesh.Scene([pc_1, pc_2])
                scene.show()
                # exit()

            if self.image_mode == 0:
                image_tensor = image_tensor.unsqueeze(-1).repeat(1, 1, 3).permute(2, 0, 1).contiguous()
                image_tensor = torch.where(torch.isinf(image_tensor), torch.zeros_like(image_tensor, device=self.device), image_tensor)
                image_tensor = torch.where(image_tensor < -10, torch.full_like(image_tensor, -10, device=self.device), image_tensor)
                # Normalize
                image_tensor = image_tensor / torch.min(image_tensor)
                # print(image_tensor)
                # print(image_tensor.shape)

                image_tensors.append(image_tensor)

            elif self.image_mode == 1:
                point_data_tensor = point_data_tensor.transpose(1, 0).view(3, self.camera_height, self.camera_width).contiguous()
                image_tensors.append(point_data_tensor)

        self.gym.end_access_image_tensors(self.sim)

        image_tensors = torch.stack(image_tensors)

        # print("min", torch.min(image_tensors))
        # print("max", torch.max(image_tensors))

        # Normalize
        # image_tensors = image_tensors / 255.

        # perception network ======================================================
        # input image
        input_data = image_tensors.view(-1, 3, 256, 256)
        # input_data = image_tensors.view(-1, 3, 256, 256).clone().detach()
        # target: cube state [x, y, z, rx, ry, rz]
        target = cube_state.clone().detach()
        perception_output = self.net.train_network(input_data, target, i=i).detach()

    def reset_idx(self, env_ids):
        # reset yumi
        # ==========================================================================================
        pos = tensor_clamp(
            self.yumi_default_dof_pos.unsqueeze(0) + 0.2 * (
                        torch.rand((len(env_ids), self.num_yumi_dofs), device=self.device) - 0.5),
            self.yumi_dof_lower_limits, self.yumi_dof_upper_limits) if self.gripper_random else tensor_clamp(
            self.yumi_default_dof_pos.unsqueeze(0),
            self.yumi_dof_lower_limits, self.yumi_dof_upper_limits)

        self.yumi_dof_pos[env_ids, :] = pos

        self.yumi_dof_vel[env_ids, :] = torch.zeros_like(self.yumi_dof_vel[env_ids])
        self.yumi_dof_targets[env_ids, :self.num_yumi_dofs] = pos

        multi_env_ids_int32 = self.yumi_indices[env_ids].to(torch.int32).flatten()  # TODO：yumi的gloabl index:0

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.yumi_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.yumi_states[env_ids] = self.default_yumi_states[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        # reset cubes   # TODO: 在pre或者post里 按一定频率检查cube是否还在object space里
        # ==========================================================================================
        # cube_indices = self.global_indices[env_ids, 2].flatten()    # TODO：cube的gloabl index:2
        cube_indices = self.cube_indices[env_ids].to(torch.int32).flatten()
        self.cube_states[env_ids] = self.default_cube_states[env_ids]

        self.cube_states[env_ids, :, :2] += 0.2 * torch.rand((len(env_ids), 1, 2), device=self.device) - 0.1
        if not self.have_gravity:
            self.cube_states[env_ids, :, 2] += 0.2 * torch.rand((len(env_ids), 1), device=self.device)
        arc_on_z = torch.rand((len(env_ids), 1), device=self.device) * np.pi * 2 - np.pi
        axis_angle = torch.cat([torch.zeros((len(env_ids), 2), device=self.device), arc_on_z], dim=-1)
        quat_tensor = conversions.angle_axis_to_quaternion(axis_angle).view(len(env_ids), 1, 4)  # shape is [len(env_ids), 1, 4]
        self.cube_states[env_ids, :, 3:7] = quat_tensor[:, :, [1,2,3,0]]

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(cube_indices), len(cube_indices))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        # print("action", actions[0])
        self.control_output = actions
        # actions : [delta_x, delta_y, delta_z, delta_rz, gripper_command]

        # self.grasp_offset = 0.12  # hand中心到cube表面的距离
        # self.grasp_height = self.table_dims.z + self.cube_size + self.grasp_offset

        gripper_actions = torch.where(actions[:, 4].view(self.num_envs, 1) < 0,
                                      torch.Tensor([[-0.0025, -0.0025]] * self.num_envs).to(self.device),
                                      torch.Tensor([[0.0025, 0.0025]] * self.num_envs).to(self.device))

        action_gripper_width = (gripper_actions[:, 0] + gripper_actions[:, 1]).view(self.num_envs, 1)
        # ======================================================================================================
        # action, gripper pos: Vec3(x, y, z)    # x, y from net output, z down 1cm per step
        action_gripper_pos = actions[:, :3] * self.xyz_scale
        # ======================================================================================================
        # action, gripper rot: (cos(rz), sin(rz))
        rz = actions[:, 3].view(self.num_envs, 1) * self.rz_scale
        action_gripper_rot = torch.cat([torch.cos(rz), torch.sin(rz)], dim=-1)
        self.actions = torch.cat([actions[:, :3] * self.xyz_scale, actions[:, 3].view(self.num_envs, 1) * self.rz_scale, gripper_actions], dim=-1)
        # self.actions = torch.cat([actions[:, :2] * self.xyz_scale, torch.zeros((self.num_envs, 1), device=self.device), actions[:, 3].view(self.num_envs, 1) * self.rz_scale, gripper_actions], dim=-1)
        # self.actions = torch.cat([actions[:, :3] * self.xyz_scale, torch.zeros((self.num_envs, 1), device=self.device) * self.rz_scale, gripper_actions], dim=-1)
        # =======================================================================================================
        # reshape tensor for observation
        # make last_action_vector
        self.last_action_vector = torch.cat([action_gripper_pos, action_gripper_rot, action_gripper_width], dim=-1)
        # targets = self.yumi_dof_pos.view(self.num_envs, self.num_dofs) + self.actions
        targets = self.yumi_dof_targets + self.actions
        # print("targets", targets[0])

        # 上下限截断
        self.yumi_dof_targets[:, :self.num_yumi_dofs] = tensor_clamp(
            targets, self.yumi_dof_lower_limits, self.yumi_dof_upper_limits)
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.yumi_dof_targets))

    def post_physics_step(self):
        self.progress_buf += 1
        # TODO:set reset buf = 1 where cube out of workspace
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

    def get_cube_euler(self):
        cube_index = self.gym.find_actor_rigid_body_index(self.envs[0], self.cubes[0], "box", gymapi.DOMAIN_ENV)
        rot_cube = R.from_quat(self.rigid_body_states[:, cube_index, 3:7].cpu().numpy())
        rot_cube = rot_cube.as_euler("xyz", degrees=False) if self.extrinsic_rotations else rot_cube.as_euler("XYZ", degrees=False)
        rot_cube = torch.Tensor(rot_cube).to(self.device)
        return rot_cube

    def get_cube_euler_z(self):
        cube_index = self.gym.find_actor_rigid_body_index(self.envs[0], self.cubes[0], "box", gymapi.DOMAIN_ENV)
        rot_cube = R.from_quat(self.rigid_body_states[:, cube_index, 3:7].cpu().numpy())
        rot_cube = rot_cube.as_euler("xyz", degrees=False) if self.extrinsic_rotations else rot_cube.as_euler("XYZ", degrees=False)
        rot_cube = torch.Tensor(rot_cube).to(self.device)
        rot_cube_z = rot_cube[:, 2].unsqueeze(-1)
        return rot_cube_z

    def get_gripper_euler(self):
        rot_gripper = R.from_quat(self.rigid_body_states[:, self.hand_idxs[0], 3:7].cpu().numpy())
        rot_gripper = rot_gripper.as_euler("xyz", degrees=False) if self.extrinsic_rotations else rot_gripper.as_euler("XYZ", degrees=False)
        rot_gripper = torch.Tensor(rot_gripper).to(self.device)
        return rot_gripper

    def get_gripper_euler_z(self):
        rot_gripper = R.from_quat(self.rigid_body_states[:, self.hand_idxs[0], 3:7].cpu().numpy())
        rot_gripper = rot_gripper.as_euler("xyz", degrees=False) if self.extrinsic_rotations else rot_gripper.as_euler("XYZ", degrees=False)
        rot_gripper = torch.Tensor(rot_gripper).to(self.device)
        rot_gripper_z = rot_gripper[:, 2].unsqueeze(-1)
        return rot_gripper_z

    def dist2depth(self, dist):
        """
        param dist: The distance data.
        return: The depth data
        """
        if isinstance(dist, list) or hasattr(dist, "shape") and len(dist.shape) > 2:
            return [self.dist2depth(img) for img in dist]
        height, width = dist.shape

        # Camera intrinsics
        cx = (width - 1.) / 2.
        cy = (height - 1.) / 2.

        f = width / (2 * np.tan(np.deg2rad(self.camera_horizontal_fov) / 2.))

        # coordinate distances to principal point
        xs, ys = np.meshgrid(np.arange(dist.shape[1]), np.arange(dist.shape[0]))
        x_opt = np.abs(xs - cx)
        y_opt = np.abs(ys - cy)

        # from 3 equations: [{X == (x-c0)/f0*Z, Y == (y-c1)/f0*Z, X*X + Y*Y + Z*Z = d*d}, {X,Y,Z}]
        depth = dist * f / np.sqrt(x_opt ** 2 + y_opt ** 2 + f ** 2)

        return depth

    def depth2pointcloud(self, depth):
        """
        param dist: The distance data.
        return: The depth data
        """
        if isinstance(depth, list) or hasattr(depth, "shape") and len(depth.shape) > 2:
            return [self.dist2depth(img) for img in depth]
        height, width = depth.shape

        # mask = np.where(depth > 0)
        # x = mask[1]
        # y = mask[0]
        # normalized_x = -(width * 0.5 - x.astype(np.float32)) / width
        # normalized_y = (height * 0.5 - y.astype(np.float32)) / height
        #
        fx = self.camera_width / (2 * np.tan(np.deg2rad(self.camera_horizontal_fov) / 2.))
        fy = self.camera_height / (2 * np.tan(np.deg2rad(self.camera_vertical_fov) / 2.))
        #
        # world_x = normalized_x * depth[y, x] / fx
        # world_y = normalized_y * depth[y, x] / fy
        # world_z = -depth[y, x]
        #
        # ones = np.ones(world_z.shape[0], dtype=np.float32)
        #
        # # pointcloud = np.vstack((world_x, world_y, world_z)).T
        # # print(pointcloud.shape)
        # # exit()
        # return np.vstack((world_x, world_y, world_z)).T

        # Camera intrinsics
        cx = (width - 1.) / 2.
        cy = (height - 1.) / 2.

        indices = np.indices((height, width), dtype=np.float32).transpose(1,2,0)
        indices[..., 0] = np.flipud(
            indices[..., 0])  # pixel indices start at top-left corner. for these equations, it starts at bottom-left
        z_e = depth
        x_e = (indices[..., 1] - cx) * z_e / fx
        y_e = -(indices[..., 0] - cy) * z_e / fy
        xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]

        # f = width / (2 * np.tan(np.deg2rad(self.camera_horizontal_fov) / 2.))
        #
        # # coordinate distances to principal point
        # xs, ys = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        # x_opt = np.abs(xs - cx).astype(np.float32) / width
        # y_opt = np.abs(ys - cy).astype(np.float32) / height
        #
        # X = depth * x_opt / f
        # Y = depth * y_opt / f
        # pointcloud = np.array([X, Y, depth])
        # pointcloud = np.reshape(pointcloud, (3, 256*256)).transpose(1, 0)
        xyz_img = xyz_img

        return xyz_img

def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    with open("YumiCube_for_pretrain.yaml", 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
            print(parsed_yaml)
        except yaml.YAMLError as exc:
            print(exc)
    yumi = YumiCube(parsed_yaml, 'cuda:0', 0, headless=True)
    print("created YumiCube")
    # yumi.gym.prepare_sim(yumi.sim)
    print("prepared sim")

    print("====================================== Simulation start ======================================")
    # while not yumi.gym.query_viewer_has_closed(yumi.viewer):
    num_images = 500000
    # for i in range(num_images//yumi.num_envs + 1):
    for i in range(600):

        # time_start = time.time()
        # step the physics
        yumi.gym.simulate(yumi.sim)

        # time.sleep(3)

        # ====================================================
        # main code
        yumi.reset_idx(torch.arange(yumi.num_envs, device=yumi.device))
        yumi.compute_observations(i)
        # # ===================================================================


        # print('time cost is :', time.time() - time_start)
        if not yumi.headless:
            # render the viewer
            yumi.gym.draw_viewer(yumi.viewer, yumi.sim, True)

            # Wait for dt to elapse in real time to sync viewer with
            # simulation rate. Not necessary in headless.
            yumi.gym.sync_frame_time(yumi.sim)

            # Check for exit condition - user closed the viewer window
            if yumi.gym.query_viewer_has_closed(yumi.viewer):
                break
    if not yumi.headless:
        yumi.gym.destroy_viewer(yumi.viewer)
    # cleanup
    yumi.gym.destroy_sim(yumi.sim)
    # save module
    if not os.path.exists("YumiDepth/nn"):
        os.mkdir("YumiDepth/nn")
    torch.save(yumi.net.state_dict(), "YumiDepth/nn/perception_pretrain_organized_point_cloud.pth")


