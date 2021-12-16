

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


class YumiCollect(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        # 一些基础配置
        self.cfg = cfg
        self.headless = headless
        self.env_spacing = self.cfg["env"]['envSpacing']
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        # self.action_scale = self.cfg["env"]["actionScale"]
        # self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        # self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.num_cubes = self.cfg["env"]["numCubes"]
        # self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        # self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        # self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        # self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        # self.open_reward_scale = self.cfg["env"]["openRewardScale"]
        # self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        # self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.height_reward_scale = self.cfg["env"]["heightRewardScale"]
        self.lift_reward_scale = self.cfg["env"]["liftRewardScale"]

        self.lift_height = self.cfg["env"]["liftHeight"]

        self.xyz_scale = self.cfg["env"]["xyzScale"]
        self.rz_scale = self.cfg["env"]["rzScale"]
        # self.up_times = self.cfg["env"]["upTimes"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.real_feature_input = self.cfg["env"]["realFeatureInput"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        # self.distX_offset = 0.04
        self.dt = 1 / 60.
        # self.dt = 1 / 10.

        # # prop dimensions
        # self.prop_width = 0.08
        # self.prop_height = 0.08
        # self.prop_length = 0.08
        # self.prop_spacing = 0.09
        if self.real_feature_input:
            self._num_obs = 517
        else:
            self._num_obs = 9
        self._num_acts = 5

        self.cfg["env"]["numObservations"] = self._num_obs
        self.cfg["env"]["numActions"] = self._num_acts
        # add ===================================================================
        # image collector
        self.step_counter = 0
        self.image_tensors = []
        self.num_save = 55000

        # table
        self.table_dims = gymapi.Vec3(0.7, 0.7, 0.1)
        # 关于相机
        self.camera_same = self.cfg["env"]["camera"]["same"]
        self.camera_width = self.cfg["env"]["camera"]["width"]# num_envs=64
# observations(num_envs, 514): [perception_output, gripper_pos, gripper_rz, gripper_width]
# actions(num_envs, 5): [delta_x, delta_y, delta_z, delta_rz, gripper_command]

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
from isaacgymenvs.tasks.myutils.resnet_new import resnet18, resnet34
from autolab_core import RigidTransform
from autolab_core import DepthImage, CameraIntrinsics
from torch.utils.tensorboard import SummaryWriter
import trimesh
from skimage.util import random_noise
from isaacgymenvs.tasks.myutils.auto_encoder import AAE, AutoEncoder


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
        # sim_device = 'cuda:1'
        # graphics_device_id = 1
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
        self.modelMode = self.cfg["env"]["modelMode"]
        self.cube_random = self.cfg["env"]["cubeRandom"]
        self.gripper_random = self.cfg["env"]["gripperRandom"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.real_feature_input = self.cfg["env"]["realFeatureInput"]
        self.have_gravity = self.cfg["env"]["haveGravity"]
        self.image_mode = self.cfg["env"]["imageMode"]  # 0: depth repeat to 3 channels    1: organized point cloud
        self.add_noise_to_image = self.cfg["env"]["addNoiseToImage"]
        self.train_perception = self.cfg["env"]["trainPerception"]
        self.load_perception = self.cfg["env"]["loadPerception"]
        self.perception_modle_path = self.cfg["env"]["perceptionModlePath"]
        self.discriminator_path = self.cfg["env"]["discriminatorPath"]
        self.showRGB = self.cfg["env"]["showRGB"]
        self.showDepth = self.cfg["env"]["showDepth"]
        self.showPointCloud = self.cfg["env"]["showPointCloud"]

        self.step_counter = 0
        self.image_tensors = []
        self.num_save = 55000
        # table
        self.table_dims = gymapi.Vec3(0.7, 0.7, 0.1)
        # 关于相机
        self.camera_same = self.cfg["env"]["camera"]["same"]
        self.camera_width = self.cfg["env"]["camera"]["width"]
        self.camera_height = self.cfg["env"]["camera"]["height"]
        self.camera_location = gymapi.Vec3(0.35, 0.3, 0.45)
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

        # hand rot:"intrinsic rotations" or "extrinsic rotations"
        self.extrinsic_rotations = True

        # about control_ik
        self.damping = 0.0

        # /add ===================================================================
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        # # count close
        # self.gripper_close = torch.Tensor([[0.]] * self.num_envs).to(self.device).view(self.num_envs, 1)
        # # count gripper height under grasp height
        # self.count_under_height = torch.Tensor([[0.]] * self.num_envs).to(self.device).view(self.num_envs, 1)
        # # create up flags
        # self.up_flags = torch.Tensor([[0]] * self.num_envs).to(self.device).view(self.num_envs, 1)
        # get gym GPU state tensors
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

        # self.global_indices = torch.arange(self.num_envs * (2 + self.num_cubes), dtype=torch.int32,
        #                                    device=self.device).view(self.num_envs, -1)


        # image
        if self.real_feature_input:
            # assert False
            self.model = models.resnet34(pretrained=True)
            self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))

            self.model.to(self.device)
            self.model.eval()
            self.preprocess = transforms.Compose([  # [1]
                # transforms.Resize(472),                    #[2]
                # transforms.CenterCrop(472),                #[3]
                # transforms.ToTensor(),                     #[4]
                transforms.Normalize(  # [5]
                    mean=[0.485, 0.456, 0.406],  # [6]
                    std=[0.229, 0.224, 0.225]  # [7]
                )])
            self.render_img = True
        else:
            self.render_img = False
        # /add ===================================================================
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.success_ids = []

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
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
        yumi_dof_props['effort'][4] = 200
        yumi_dof_props['effort'][5] = 200


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

            yumi_actor, yumi_start_pose = create_assets.create_yumi(self.gym, env_ptr, yumi_asset, i)
            self.gym.set_actor_dof_properties(env_ptr, yumi_actor, yumi_dof_props)
            yumi_idx = self.gym.get_actor_index(env_ptr, yumi_actor, gymapi.DOMAIN_SIM)
            self.yumi_indices.append(yumi_idx)
            # get inital hand pose
            hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, yumi_actor, "gripper_r_base")
            hand_pose = self.gym.get_rigid_transform(env_ptr, hand_handle)
            self.default_yumi_states.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z,
                                             hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w,
                                             0, 0, 0, 0, 0, 0])
            # exit()
            self.init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            self.init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])
            # get global index of hand in rigid body state tensor
            hand_idx = self.gym.find_actor_rigid_body_index(env_ptr, yumi_actor, "gripper_r_base", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)

            table_actor, table_start_pose = create_assets.create_table(self.gym, env_ptr, self.table_dims, table_asset, i)

            if self.num_cubes > 0:
                self.cube_start.append(self.gym.get_sim_actor_count(self.sim))
                cube_actor, cube_pose = create_assets.create_cube(self.gym, env_ptr, self.cube_middle,
                                                                   self.cube_size, cube_asset, self.table_dims, i)
                self.default_cube_states.append([cube_pose.p.x, cube_pose.p.y, cube_pose.p.z,
                                                 cube_pose.r.x, cube_pose.r.y, cube_pose.r.z, cube_pose.r.w,
                                                 0, 0, 0, 0, 0, 0])
                self.cubes.append(cube_actor)
                cube_idx = self.gym.get_actor_index(env_ptr, cube_actor, gymapi.DOMAIN_SIM)
                self.cube_indices.append(cube_idx)

            # camera_actor = create_assets.create_camera(self.gym, env_ptr, self.camera_location, self.camera_lookat,
            #                                            self.camera_width, self.camera_height)

            camera_actor = create_assets.create_camera(self.gym, env_ptr, self.camera_location, self.camera_lookat, self.camera_width, self.camera_height)

            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)
            self.yumis.append(yumi_actor)
            self.tables.append(table_actor)
            self.cameras.append(camera_actor)
            print("create%d"%i)

        self.yumi_indices = to_torch(self.yumi_indices, dtype=torch.long, device=self.device)
        self.cube_indices = to_torch(self.cube_indices, dtype=torch.long, device=self.device)
        # 无所谓找的是哪个环境的，所有环境里都一样，指的是某个关节在actor里的编号
        print("create end")
        # self.hand_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.yumis[0], "yumi_link_7_r")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.yumis[0], "gripper_r_finger_l")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.yumis[0], "gripper_r_finger_r")
        self.default_cube_states = to_torch(self.default_cube_states, device=self.device, dtype=torch.float
                                            ).view(self.num_envs, self.num_cubes, 13)
        self.default_yumi_states = to_torch(self.default_yumi_states, device=self.device, dtype=torch.float
                                            ).view(self.num_envs, 1, 13)
        print("made default cube states")
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

        # gripper_quat = self.rigid_body_states[:, 3, 3:7].view(self.num_envs, 4)[0]

        # height between hand and table
        # height = (gripper_height - object_height).view(self.num_envs)
        rewards = torch.zeros(self.num_envs, device=self.device)

        # rewards -= abs(height - 0.13) * self.height_reward_scale
        # d = torch.norm(self.rigid_body_states[:, cube_index, :2] - self.rigid_body_states[:, hand_index, :2], dim=-1)
        offset_height = 0.14
        diff_height = offset_height - self.cube_size / 2.
        offset_gripper_pos = torch.cat([self.rigid_body_states[:, hand_index, :2], (self.rigid_body_states[:, hand_index, 2].unsqueeze(-1) - diff_height)], dim=-1)
        # print(self.rigid_body_states[:, hand_index, :3].size())
        # print(offset_gripper_pos.size())
        # exit()
        # rewards for distance
        d = torch.norm(self.rigid_body_states[:, cube_index, :3] - offset_gripper_pos, dim=-1)
        dist_reward = 1.0 / (1.0 + (10 * d) ** 2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.005, dist_reward * 2, dist_reward)
        rewards += dist_reward

        # rewards for angle_axis
        angle_axis_z_cube = conversions.quaternion_to_angle_axis(self.rigid_body_states[:, cube_index, [6,3,4,5]])
        angle_axis_z_gripper = conversions.quaternion_to_angle_axis(self.rigid_body_states[:, hand_index, [6,3,4,5]])
        angle_axis_z_diff = abs(angle_axis_z_cube[:, 2] - angle_axis_z_gripper[:, 2])
        angle_reward = 1.0 / (1.0 + (10 * angle_axis_z_diff) ** 2)
        angle_reward *= angle_reward
        # angle_reward = torch.where(angle_axis_z_diff <= 0.005, dist_reward * 2, dist_reward)
        rewards += angle_reward

        around = (abs(gripper_x - object_x) - 0.005 < 0) & (abs(gripper_y - object_y) - 0.005 < 0) & (abs(gripper_height - object_height) - 0.005 < 0)

        # reswards for lift height
        rewards += ((object_height - self.table_dims.z - self.cube_size / 2.) * self.lift_reward_scale * around).view(self.num_envs)
        print("lift height", (object_height - self.table_dims.z - self.cube_size / 2.)[0].item())


        # around = d < 0.01
        # print("around", around * rewards)
        # exit()
        success_ = torch.where((object_height > (self.table_dims.z + self.cube_size / 2.)) & around,
                               torch.Tensor([[1.]] * self.num_envs).to(self.device),
                               torch.Tensor([[0.]] * self.num_envs).to(self.device)).view(self.num_envs)
        self.success_ids = success_.nonzero(as_tuple=False).squeeze(-1).tolist()
        # bonus for lift height.  bonus需要很大，否则train不出来(只有现在的1/20的时候就不行) max episode length需要大一点(300)
        # 任务越复杂越需要更大的bonus和max episode length，比如cube在中心的时候，bonus是现在的1/2，max episode length也是1/2，
        # 但cube不在中心，就得加大bonus和max episode length，否则gripper不会lift
        # 在max episode length变大的时候bonus要相应变大，否则bonus不明显
        rewards += torch.where((object_height > (self.table_dims.z + self.cube_size / 2.)) & around,
                               (5 * dist_reward).view(self.num_envs, 1),
                               torch.Tensor([[0.]] * self.num_envs).to(self.device)).view(self.num_envs)
        rewards += torch.where(object_height > (self.table_dims.z + self.cube_size / 2.) + 0.01,
                               torch.Tensor([[10.]] * self.num_envs).to(self.device),
                               torch.Tensor([[0.]] * self.num_envs).to(self.device)).view(self.num_envs)
        rewards += torch.where(object_height > (self.table_dims.z + self.cube_size / 2.) + 0.03,
                               torch.Tensor([[30.]] * self.num_envs).to(self.device),
                               torch.Tensor([[0.]] * self.num_envs).to(self.device)).view(self.num_envs)
        rewards += torch.where(object_height > (self.table_dims.z + self.cube_size / 2.) + 0.05,
                               torch.Tensor([[50.]] * self.num_envs).to(self.device),
                               torch.Tensor([[0.]] * self.num_envs).to(self.device)).view(self.num_envs)



        # print("=========================success: %d===============================" % torch.sum(success))
        # print("cube", object_x[0], object_y[0], object_height[0])
        # print("gripper", gripper_x[0], gripper_y[0], gripper_height[0])
        # rewards += torch.where(success, torch.Tensor([[100.]] * self.num_envs).to(self.device),
        #                        torch.Tensor([[0.]] * self.num_envs).to(self.device)).view(self.num_envs)
        # penalty
        # rewards -= self.action_penalty_scale
        # penalty reward
        rewards -= torch.norm(self.control_output, dim=-1) * 0.5
        print("action penalty", torch.norm(self.control_output, dim=-1)[0] * 0.5)

        # self.reset_buf = torch.where(success.view(self.num_envs), torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
        self.rew_buf = rewards
        print("reward", rewards[0])
        print('================================================')
        # return success_ids

    def compute_observations(self):
        # obsevations:
        #   images: perception_output: [num_envs, 512]
        #   last action:
        #       gripper pos: Vec3(x, y, z)
        #       gripper rot: (cos(rz), sin(rz))
        #       gripper width: float
        #   state:
        #       gripper width: float
        #       gripper height: float

        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_jacobian_tensors(self.sim)
        if self.real_feature_input:
            self.gym.step_graphics(self.sim)
        # self.gym.step_graphics(self.sim)
        cube_index = self.gym.find_actor_rigid_body_index(self.envs[0], self.cubes[0], "box", gymapi.DOMAIN_ENV)
        angle_axis_z_cube = conversions.quaternion_to_angle_axis(
            self.rigid_body_states[:, cube_index, [6, 3, 4, 5]])
        angle_axis_z_gripper = conversions.quaternion_to_angle_axis(
            self.rigid_body_states[:, self.hand_idxs[0], [6, 3, 4, 5]])
        # =============================================================================================
        # state, gripper pos: Vec3(x, y, z)
        # [num_envs, 3]
        gripper_pos = self.rigid_body_states[:, self.hand_idxs[0], :3].view(self.num_envs, 3)
        # =============================================================================================
        # state, gripper width: float (meter)
        # [num_envs, 1]
        gripper_width = self.yumi_dof_pos[:, 4] + self.yumi_dof_pos[:, 5]    # 两个gripper的dof都是正数[0, 0.025]
        gripper_width = gripper_width.view(self.num_envs, 1)
        # =============================================================================================
        # state, gripper height: float
        # [num_envs, 1]
        # gripper_height = self.rigid_body_states[:, self.hand_idxs[0], 2].view(self.num_envs, 1)
        # =============================================================================================
        # make state_vector
        state_vector = torch.cat([gripper_pos, angle_axis_z_gripper[:, 2].unsqueeze(-1), gripper_width], dim=-1)
        # state_vector = torch.cat([angle_axis_z_gripper[:, 2].unsqueeze(-1), gripper_width], dim=-1)
        # =============================================================================================
        # make info_vector
        # info_vector = torch.cat([state_vector, self.last_action_vector], dim=-1)
        # =============================================================================================
        # resnet input  shape:[num_envs, 3, camera_height, camera_width]
        # resnet output shape:[num_envs, 512]
        # info_vector   shape:[num_envs, 8]
        if self.real_feature_input:
            # images =====================================================================
            # render the camera sensors
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            #
            # User code to digest tensors
            #
            # get image tensor
            image_tensors = []

            for j in range(self.num_envs):
                # for j in self.success_ids:
                _image_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[j], self.cameras[j],
                                                                     gymapi.IMAGE_COLOR)
                # H * W * 3
                image_tensor = gymtorch.wrap_tensor(_image_tensor)[:, :, :3].permute(2, 0, 1).contiguous()
                image_tensors.append(image_tensor)

                show_image = True
                if show_image and len(self.success_ids) > 0 and j == self.success_ids[0]:
                # if show_image:
                    self.success_ids = []
                    image_array = image_tensor.permute(1, 2, 0).cpu().numpy()
                    image = Image.fromarray(image_array).convert("RGB")
                    image.show()
                    # exit()
            self.gym.end_access_image_tensors(self.sim)

            image_tensors = torch.stack(image_tensors)
            # Normalize
            image_tensors = image_tensors / 255.
            image_tensors = self.preprocess(image_tensors)

            perception_output = self.model(image_tensors.view(-1, 3, 256, 256)).squeeze()   # torch.Size([num_envs, 512])

            self.obs_buf = torch.cat([perception_output, state_vector], dim=-1)
            # self.obs_buf = torch.cat([perception_output, info_vector], dim=-1)
        if not self.real_feature_input:
            # for collection -------------------------------------------------------
            success_step = 200000000
            if self.step_counter > success_step:
                # images =====================================================================
                # render the camera sensors
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)
                #
                # User code to digest tensors
                #
                # get image tensor
                # image_tensors = []
                # show_image = True
                for j in range(self.num_envs):
                    # get image tensor in one env =======================================================================
                    # for j in self.success_ids:
                    image_kind = gymapi.IMAGE_COLOR if self.image_mode == 2 else gymapi.IMAGE_DEPTH
                    _image_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[j], self.cameras[j],
                                                                         image_kind)

                    # H * W * channel
                    image_tensor = gymtorch.wrap_tensor(_image_tensor)  # image_array is depth map, not distance map
                    self.image_tensors.append(image_tensor)
                self.gym.end_access_image_tensors(self.sim)
            # for collection --------------------------------------------------------

            object_xyz = self.rigid_body_states[:, cube_index, :3].view(self.num_envs, 3)
            object_quat = self.rigid_body_states[:, cube_index, 3:7].view(self.num_envs, 4)
            gripper_quat = self.rigid_body_states[:, self.hand_idxs[0], 3:7].view(self.num_envs, 4)

            self.obs_buf = torch.cat([object_xyz, angle_axis_z_cube[:, 2].unsqueeze(-1), gripper_pos,
                                      angle_axis_z_gripper[:, 2].unsqueeze(-1), gripper_width], dim=-1)

        print("run %d" % self.step_counter, end="\t")
        print(self.num_save // self.num_envs + 1, end="\t")
        # save if num is enough --------------------------
        if self.step_counter > (self.num_save // self.num_envs + 1 + success_step):
            store_image_tensors = self.image_tensors = torch.stack(image_tensors)
            if not os.path.exists("myutils/image_tensors"):
                os.mkdir("myutils/image_tensors")
            torch.save(store_image_tensors.cpu(), "myutils/image_tensors/image_tensors_from_success_cpu.pt")
            print("save success !")
            print(store_image_tensors.shape)
        # save if num is enough --------------------------

        self.step_counter += 1
        print("nan", torch.isnan(self.obs_buf).int().sum().item(), end="\t")
        # index = torch.isnan(self.obs_buf).int().nonzero(as_tuple=False)

        if torch.isnan(self.obs_buf).int().sum().item() > 0:
            # x[x.nonzero(as_tuple=True)]gives all nonzero values of tensor x.
            print(self.obs_buf[torch.isnan(self.obs_buf).int().nonzero(as_tuple=True)])
            print(self.obs_buf[torch.isnan(self.obs_buf).int().nonzero(as_tuple=True)])
            index = torch.isnan(self.obs_buf).int().nonzero(as_tuple=False)
            print(index)
            ids = torch.unique(index[0])
            print(ids)
            print(ids[1])
            print('---------------------')
            print("save", torch.save(image_tensors[ids[1]], "image_tensors_obs_nan.pt"))
            print(image_tensors[ids[1]])
            with open("obs_nan.txt", "a") as f:
                x = image_tensors[ids].cpu().numpy().tolist()
                strNums = [str(x_i) for x_i in x]
                str1 = ",".join(strNums)
                f.write("runs %d\t" % self.step_counter)
                f.write(str1)
                f.write("\n")
                f.write("-----------------------------------------------------------------------")
        return self.obs_buf

    def reset_idx(self, env_ids):
        # reset yumi
        # ==========================================================================================
        pos = tensor_clamp(
            self.yumi_default_dof_pos.unsqueeze(0) + 0.0 * (
                        torch.rand((len(env_ids), self.num_yumi_dofs), device=self.device) - 0.5),
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
        print("reset")

        # reset cubes   # TODO: 在pre或者post里 按一定频率检查cube是否还在object space里
        # ==========================================================================================
        if self.num_cubes > 0:
            # cube_indices = self.global_indices[env_ids, 2].flatten()    # TODO：cube的gloabl index:2
            cube_indices = self.cube_indices[env_ids].to(torch.int32).flatten()
            self.cube_states[env_ids] = self.default_cube_states[env_ids]
            # self.cube_states[env_ids, :, :2] += 0.2 * torch.rand((len(env_ids), 1, 2), device=self.device) - 0.1
            # arc_on_z = torch.rand((len(env_ids), 1), device=self.device) * np.pi * 2 - np.pi
            # axis_angle = torch.cat([torch.zeros((len(env_ids), 2), device=self.device), arc_on_z], dim=-1)
            # quat_tensor = conversions.angle_axis_to_quaternion(axis_angle).view(len(env_ids), 1, 4)  # shape is [len(env_ids), 1, 4]
            # self.cube_states[env_ids, :, 3:7] = quat_tensor[:, :, [1,2,3,0]]

            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(cube_indices), len(cube_indices))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        print("action", actions[0])
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


def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    with open("../cfg/task/YumiCube.yaml", 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
            print(parsed_yaml)
        except yaml.YAMLError as exc:
            print(exc)
    yumi = YumiCollect(parsed_yaml, 'cuda:0', 0, headless=True)
    print("created YumiCube")
    # yumi.gym.prepare_sim(yumi.sim)
    print("prepared sim")

    # # action4: 其他不变，夹爪随机开合
    # gripper_rand = torch.rand(yumi.num_envs, device=yumi.device).view(yumi.num_envs, 1)
    # gripper_action = torch.where(gripper_rand > 0.5,
    #                              torch.Tensor([[0., 0.]] * yumi.num_envs).to(yumi.device),
    #                              torch.Tensor([[0.025, 0.025]] * yumi.num_envs).to(yumi.device))
    # gripper_action = gripper_action.view(yumi.num_envs, 2)
    print("====================================== Simulation start ======================================")
    i = 0
    # while not yumi.gym.query_viewer_has_closed(yumi.viewer):
    open = True
    while True:
        time_start = time.time()
        # step the physics
        yumi.gym.simulate(yumi.sim)
        yumi.gym.fetch_results(yumi.sim, True)

        # # render the camera sensors
        # yumi.gym.render_all_camera_sensors(yumi.sim)

        # yumi.compute_observations()
        # yumi.gym.refresh_actor_root_state_tensor(yumi.sim)
        # yumi.gym.refresh_dof_state_tensor(yumi.sim)
        # yumi.gym.refresh_rigid_body_state_tensor(yumi.sim)
        # yumi.gym.refresh_jacobian_tensors(yumi.sim)
        # d_action = torch.zeros_like(yumi.yumi_dof_pos, device=yumi.device).view(yumi.num_envs, 9)
        # yumi.pre_physics_step(d_action)
        # ====================================================
        # main code

        # # ===================================================================
        # # config d_actions  ps.这里是delta actions，不是actions
        # # Maximum velocity: [0:7]4.e+02  [7:9]1.e+06
        # # actions:
        # # 1 其他不变，z下降1cm
        # # 2 其他不变 x, y 随机[0，0.01]
        # # 3 其他不变，yaw旋转
        # # 4 其他不变，夹爪随机开合
        # # 5 其他不变，夹爪开 (不用担心超出dof limit，之后函数里有截断，正的就是不断打开）
        # # 6 其他不变，夹爪合 (不用担心超出dof limit，之后函数里有截断，负的就是不断闭合）
        # # 7 其他不变 x+=0.01
        # # 8 其他不变 y+=0.01
        # # ===================================================================
        # # action = yumi.yumi_dof_pos.contiguous().squeeze(-1)
        # d_action = torch.zeros_like(yumi.yumi_dof_pos, device=yumi.device).view(yumi.num_envs, 9)
        # # # ==========================================================================================
        # # # action1: 其他不变，z下降1cm
        # # pos_err1 = torch.Tensor([0, 0, -0.01]).to(yumi.device).repeat(yumi.num_envs).view(yumi.num_envs, 3)
        # # orn = torch.Tensor([1, 0, 0, 0]).to(yumi.device).repeat(yumi.num_envs).view(yumi.num_envs, 4)
        # # orn_err1 = orientation_error(orn, orn)
        # # dpose1 = torch.cat([pos_err1, orn_err1], -1).unsqueeze(-1)
        # # # ==========================================================================================
        # # # action2: 其他不变 x, y 随机[0，0.01]
        # # # pos_err2_xy = 0.2 * (torch.rand((yumi.num_envs, 2), device=yumi.device) - 0.5)
        # # pos_err2_xy = 0.1 * (torch.rand((yumi.num_envs, 2), device=yumi.device))
        # #
        # # pos_err2_z = torch.Tensor([0.]).to(yumi.device).repeat(yumi.num_envs).view(yumi.num_envs, 1)
        # # pos_err2 = torch.cat([pos_err2_xy, pos_err2_z], -1).view(yumi.num_envs, 3)
        # # orn = torch.Tensor([1, 0, 0, 0]).to(yumi.device).repeat(yumi.num_envs).view(yumi.num_envs, 4)
        # # orn_err2 = orientation_error(orn, orn)
        # # dpose2 = torch.cat([pos_err2, orn_err2], -1).unsqueeze(-1)
        # # # print(dpose2)
        # # # print(dpose2.size())
        # # # ==========================================================================================
        # # # action3: 其他不变，yaw旋转
        # # # 最大角度: max_angle（弧度）
        # # # 四元数[cos(angle/2), 0, 0, sin(angle/2)]
        # # # cos(angle/2)  [0, max_cos]
        # # # sin(angle/2)  [-max_sin, max_sin](正负都有）  now[0, max_sin](只向正方向移动)
        # # max_angle = np.pi / 12.0
        # # max_cos = np.cos(max_angle / 2)
        # # max_sin = np.sin(max_angle / 2)
        # # rand_cos_tensor = max_cos * torch.rand(yumi.num_envs, device=yumi.device).view(yumi.num_envs, 1)
        # # # rand_sin_tensor = 2 * max_sin * torch.rand(yumi.num_envs, device=yumi.device).view(yumi.num_envs, 1) - max_sin
        # # rand_sin_tensor = max_sin * torch.rand(yumi.num_envs, device=yumi.device).view(yumi.num_envs, 1)
        # # tensor0 = torch.Tensor([0.]).to(yumi.device).repeat(yumi.num_envs).view(yumi.num_envs, 1)
        # # pos_err3 = torch.Tensor([0, 0, 0.]).to(yumi.device).repeat(yumi.num_envs).view(yumi.num_envs,
        # #                                                                                3)  # shape [yumi.num_envs, 3]
        # # orn0 = torch.Tensor([1, 0, 0, 0]).to(yumi.device).repeat(yumi.num_envs).view(yumi.num_envs, 4)
        # # goal_orn = torch.cat([rand_cos_tensor, rand_sin_tensor, tensor0, tensor0], -1)
        # # orn_err3 = orientation_error(goal_orn, orn0)
        # # dpose3 = torch.cat([pos_err3, orn_err3], -1).unsqueeze(-1)
        # # # ==========================================================================================
        # # # action4: 其他不变，夹爪随机开合
        # # gripper_rand = torch.rand(yumi.num_envs, device=yumi.device).view(yumi.num_envs, 1)
        # # gripper_action4 = torch.where(gripper_rand > 0.5,
        # #                              torch.Tensor([[-0.025, -0.025]] * yumi.num_envs).to(yumi.device),
        # #                              torch.Tensor([[0.025, 0.025]] * yumi.num_envs).to(yumi.device))
        # # gripper_action4 = gripper_action4.view(yumi.num_envs, 2)
        # # # ==========================================================================================
        # # # action5: 其他不变，夹爪开 (不用担心超出dof limit，之后函数里有截断，正的就是不断打开）
        # # gripper_action5 = torch.Tensor([[0.025, 0.025]] * yumi.num_envs).to(yumi.device).view(yumi.num_envs, 2)
        # # # ==========================================================================================
        # # # action6: 其他不变，夹爪合 (不用担心超出dof limit，之后函数里有截断，负的就是不断闭合）
        # # gripper_action6 = torch.Tensor([[-0.025, -0.025]] * yumi.num_envs).to(yumi.device).view(yumi.num_envs, 2)
        # # # ==========================================================================================
        # # # action7: 其他不变 x+=0.01
        # # pos_err7 = torch.Tensor([0.01, 0., 0.]).to(yumi.device).repeat(yumi.num_envs).view(yumi.num_envs, 3)
        # # orn = torch.Tensor([1, 0, 0, 0]).to(yumi.device).repeat(yumi.num_envs).view(yumi.num_envs, 4)
        # # orn_err7 = orientation_error(orn, orn)
        # # dpose7 = torch.cat([pos_err7, orn_err7], -1).unsqueeze(-1)
        # # # ==========================================================================================
        # # action8: 其他不变 y+=0.01
        # pos_err8 = torch.Tensor([0., 0.01, 0.]).to(yumi.device).repeat(yumi.num_envs).view(yumi.num_envs, 3)
        # orn = torch.Tensor([1, 0, 0, 0]).to(yumi.device).repeat(yumi.num_envs).view(yumi.num_envs, 4)
        # orn_err8 = orientation_error(orn, orn)
        # dpose8 = torch.cat([pos_err8, orn_err8], -1).unsqueeze(-1)
        # # # ==========================================================================================
        # # # choose action 1-8
        # # # ==========================================================================================
        # # action1-3,7-8(下面两行）
        # dpose = dpose8  # TODO:choose action
        # # d_action[:, :7] = control_ik(dpose, yumi.device, yumi.j_eef, yumi.num_envs, yumi.damping)
        # d_action[:, :7] = control_ik(dpose, yumi.device, yumi.j_eef, yumi.num_envs)
        # # # action4-6（下面一行）
        # # # d_action[:, 7:9] = gripper_action5
        # # Deploy actions
        # yumi.pre_physics_step(d_action)
        # # print(yumi.operational_pos - dpose8.view(yumi.num_envs, 6))
        # # # 验证过：在control_ik里damping=0，operational_pos与dpose近似相等，误差e-07以下
        # # exit()
        # # yumi.gym.set_dof_position_target_tensor(yumi.sim, gymtorch.unwrap_tensor(d_action))
        # # =====================================================
        # # =====================================================
        # # =====================================================
        # [dx, dy, dz, drz, gripper_cmd]
        # if i % 6 == 0:
        operational_action = torch.Tensor([0., 0.0, 0, 1, -1.]).to(yumi.device).repeat(yumi.num_envs).view(yumi.num_envs, 5)
        # operational_action = torch.Tensor([-0.9581,  1.0000,  1.0000, -0.5416, -0.4064]).to(yumi.device).repeat(yumi.num_envs).view(yumi.num_envs, 5)
        yumi.pre_physics_step(operational_action)
        yumi.post_physics_step()
        # yumi.compute_observations()
        # yumi.compute_reward()
        # if open:
        #     # d_action = torch.zeros_like(yumi.yumi_dof_pos, device=yumi.device).view(yumi.num_envs, 9)
        #     gripper_action5 = torch.Tensor([[0.025, 0.025]] * yumi.num_envs).to(yumi.device).view(yumi.num_envs, 2)
        #     yumi.yumi_dof_targets[:, 7:9] = gripper_action5
        #     targets = yumi.yumi_dof_targets
        #     yumi.gym.set_dof_position_target_tensor(yumi.sim, gymtorch.unwrap_tensor(targets))
        #
        #     width = yumi.yumi_dof_pos[:, 7, 0] + yumi.yumi_dof_pos[:, 8, 0]
        #     print("width:", width)
        #     open = False
        # else:
        #     # d_action = torch.zeros_like(yumi.yumi_dof_pos, device=yumi.device).view(yumi.num_envs, 9)
        #     gripper_action5 = torch.Tensor([[0., 0.]] * yumi.num_envs).to(yumi.device).view(yumi.num_envs, 2)
        #     yumi.yumi_dof_targets[:, 7:9] = gripper_action5
        #     targets = yumi.yumi_dof_targets
        #     yumi.gym.set_dof_position_target_tensor(yumi.sim, gymtorch.unwrap_tensor(targets))
        #
        #     width = yumi.yumi_dof_pos[:, 7, 0] + yumi.yumi_dof_pos[:, 8, 0]
        #     print("width:", width)
        #     # open = True
        # # ===================================================================================================
        # # ===================================================================================================
        # # # test reset
        # # if i % 100 == 0:
        # #     yumi.reset_idx(torch.arange(yumi.num_envs, device=yumi.device))
        # #     print("reset%d" % (i // 100))
        #
        # # ===================================================================================================
        # # ===================================================================================================
        # # save image
        # if i % 1 == 0:
        #     save.save_images(yumi.gym, yumi.sim, yumi.num_envs, yumi.envs, yumi.cameras, "yumi_image_wrist", i)

        i += 1
        print("run%d" % i)
        # print('time cost is :', time.time() - time_start)
        # communicate physics to graphics system
        yumi.gym.step_graphics(yumi.sim)
        if not yumi.headless:
            # render the viewer
            yumi.gym.draw_viewer(yumi.viewer, yumi.sim, True)

            # Wait for dt to elapse in real time to sync viewer with
            # simulation rate. Not necessary in headless.
            yumi.gym.sync_frame_time(yumi.sim)

            # Check for exit condition - user closed the viewer window
            if yumi.gym.query_viewer_has_closed(yumi.viewer):
                break

    # cleanup
    if not yumi.headless:
        yumi.gym.destroy_viewer(yumi.viewer)
    yumi.gym.destroy_sim(yumi.sim)
