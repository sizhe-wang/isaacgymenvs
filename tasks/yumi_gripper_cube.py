# 1123
# 添加了obeservation和action的部分
# 添加了headless   # 需要更改“YumiCube.yaml"里的“env：enableCameraSensors”为True，需要更改vec_task.py第78行（原来的读取enableCameraSensors有bug）
# 添加了 info_vector(包含除resnet输出外的其他全部需要作为control input的信息
#       info_vetor(size[num_envs,13])包含state(pos(3) + rot(2) + width(1) + height(1)) + last_action(pos(3) + rot(2) + width(1))
# 添加了control_input(size[num_envs,525])包含resnet output(512) + info vector(13)
# 调试：
# obsevations:
#   【checked】images: perception_output: [num_envs, 512]
#   last action:
#       【checked】gripper pos: Vec3(x, y, z)
#       【checked】gripper rot: (cos(rz), sin(rz))     or Vec3(rx, ry, rz)   rx, ry固定
#       【checked】gripper width: float
#   state:
#       【checked】gripper width: float
#       【checked】gripper height real得不到          need?
#       【checked】gripper pos: Vec3(x, y, z)        need?
#       【checked】gripper rot: (cos, sin) 只绕z轴    need?
# TODO：怎么验证四元数转换到欧拉角对不对,用的是scipy.spatial.transform.Rotation，应该用"intrinsic rotations" or "extrinsic rotations"?
# TODO：control_ik里的damping
# TODO：rewards
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
# dpose: 末端执行器的微分运动 delta的[p.x, p.y, p.z, r.x, r.y, r.z, r.w]
# u: 关节的微分运动 （只考虑arm，不考虑finger，所以是7个dof）
# j_eef：arm的雅可比矩阵
# 没有阻尼的话 u = torch.inverse(j_eef) @ dpose
# 但是yumi设置的是DOF_MODE_POS（其他选项是DOF_MODE_NONE，DOF_MODE_EFFORT，DOF_MODE_VEL），阻尼必须非零
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
        # self.sim_params = sim_params
        # self.physics_engine = physics_engine
        # self.cfg["device_type"] = device_type
        # self.cfg["device_id"] = device_id
        # self.cfg["headless"] = headless
        # 关于强化学习
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        # 关于状态
        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.num_cubes = self.cfg["env"]["numCubes"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self.cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.height_reward_scale = self.cfg["env"]["heightRewardScale"]
        self.lift_height = self.cfg["env"]["liftHeight"]

        self.xyz_scale = self.cfg["env"]["xyzScale"]
        self.rz_scale = self.cfg["env"]["rzScale"]
        self.up_times = self.cfg["env"]["upTimes"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.real_feature_input = self.cfg["env"]["realFeatureInput"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1 / 60.
        # self.dt = 1 / 10.

        # # prop dimensions
        # self.prop_width = 0.08
        # self.prop_height = 0.08
        # self.prop_length = 0.08
        # self.prop_spacing = 0.09
        if self.real_feature_input:
            self._num_obs = 520
        else:
            self._num_obs = 15
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
        self.camera_location = gymapi.Vec3(0.35, 0.3, 0.45)
        self.camera_lookat = gymapi.Vec3(0.35, 0., self.table_dims.z)

        # cube
        self.cube_size = 0.035
        self.cube_spacing = self.cube_size + 0.005
        self.cube_middle = gymapi.Transform()
        self.cube_middle.p = gymapi.Vec3(0.35, 0., self.table_dims.z + self.cube_size / 2. + 0.0002)

        # object reset space
        self.object_space_lower = gymapi.Vec3(0.05, -0.25, 0.)
        self.object_space_upper = gymapi.Vec3(0.65, 0.25, self.table_dims.z + self.cube_size / 2. + 0.0002)

        # prepare some lists
        self.envs = []
        self.cube_idxs = []

        # hand rot:"intrinsic rotations" or "extrinsic rotations"
        self.extrinsic_rotations = True

        # about control_ik
        self.damping = 0.0


        # /add ===================================================================
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
        print("supper init finished")
        # count close
        self.gripper_close = torch.Tensor([[0.]] * self.num_envs).to(self.device).view(self.num_envs, 1)
        # count gripper height under grasp height
        self.count_under_height = torch.Tensor([[0.]] * self.num_envs).to(self.device).view(self.num_envs, 1)
        # create up flags
        self.up_flags = torch.Tensor([[0]] * self.num_envs).to(self.device).view(self.num_envs, 1)
        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        print("got gpu tensors")

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        print("refreshed tensors")

        # create some wrapper tensors for different slices
        # down_states = [28.67, -67.79, -19.64, -79.44, 117.26, -19.22, -1.01, 0.025, 0.025]
        # down_states = [0.5003859, -1.1831587, -0.01762783, -0.34278267, -1.38648956, 2.04657308, -0.33545228, 0.025,
        #                0.025]
        # self.yumi_default_dof_pos = to_torch([0.5003859, -1.1831587, -0.01762783, -0.34278267, -1.38648956,
        #                                       2.04657308, -0.33545228, 0.025, 0.025], device=self.device)
        # 靠近桌面
        # self.yumi_default_dof_pos = to_torch([0.9985358, -1.4063797, -0.8577260, 0.9911097, -1.9251643,
        #                                       1.5174509, 0.9186748, 0.025, 0.025], device=self.device)
        # 几乎贴近桌面
        self.yumi_default_dof_pos = to_torch([0, 0, 0, 0, 0.025, 0.025], device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # TODO：dof_state_tensor: shape (num_envs * 9(num_dofs), 2)
        # 2: 0:position, 1:velocity

        # self.yumi_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_yumi_dofs]
        # self.yumi_dof_pos = self.yumi_dof_state[:, 0]
        # self.yumi_dof_vel = self.yumi_dof_state[:, 1]
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.yumi_dof_pos = self.dof_state[:, 0].view(self.num_envs, self.num_dofs)
        self.yumi_dof_vel = self.dof_state[:, 1].view(self.num_envs, self.num_dofs)
        # self.yumi_dof_pos = self.dof_state[..., 0]
        # self.yumi_dof_vel = self.dof_state[..., 1]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        # print(self.rigid_body_states.size())
        # exit()
        self.num_bodies = self.rigid_body_states.shape[1]
        print("created tensors")

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)
        # TODO：root tensor: shape (num_envs, 3, 13)(cube只有一个的时候）
        # 3: 0:yumi, 1:table, 2:cube

        # add ===================================================================
        self.cube_states = self.root_state_tensor[:, 2].view(self.num_envs, 1, -1)  # TODO: 确定cube在env中的序号是不是2: 是
        # /add ===================================================================


        self.yumi_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * (2 + self.num_cubes), dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)
        print("finished something about cubes")

        # image
        if self.real_feature_input:
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
        print("reset finished")

    def create_sim(self):
        print("create sim")
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.dt = self.dt
        self.sim_params.substeps = 2
        self.sim_params.use_gpu_pipeline = self.cfg["sim"]["use_gpu_pipeline"]
        if self.physics_engine == gymapi.SIM_PHYSX:
            self.sim_params.physx.solver_type = 1
            self.sim_params.physx.num_position_iterations = 8
            self.sim_params.physx.num_velocity_iterations = 1
            self.sim_params.physx.rest_offset = 0.000
            self.sim_params.physx.contact_offset = 0.0002
            self.sim_params.physx.friction_offset_threshold = 0.001
            self.sim_params.physx.friction_correlation_distance = 0.0005
            self.sim_params.physx.num_threads = self.cfg["sim"]["physx"]["num_threads"]
            self.sim_params.physx.use_gpu = self.cfg["sim"]["physx"]["use_gpu"]
            self.sim_params.physx.max_gpu_contact_pairs = 2097152
            # 增大max_gpu_contact_pairs来解决场景中碰撞较多时GPU显存溢出的问题
        else:
            raise Exception("This example can only be used with PhysX")
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
        self.yumi_asset = yumi_asset

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
        # print("yumi_dof_props", yumi_dof_props)
        # exit()
        # [( True, -2.94, 2.94, 1, 3.14, 300., 4.e+02,  80., 0., 0.)
        # ( True, -2.5 , 0.76, 1, 3.14, 300., 4.e+02,  80., 0., 0.)
        # ( True, -2.94, 2.94, 1, 3.14, 300., 4.e+02,  80., 0., 0.)
        # ( True, -2.16, 1.4 , 1, 3.14, 300., 4.e+02,  80., 0., 0.)
        # ( True, -5.06, 5.06, 1, 6.98, 300., 4.e+02,  80., 0., 0.)
        # ( True, -1.54, 2.41, 1, 6.98, 300., 4.e+02,  80., 0., 0.)
        # ( True, -4.  , 4.  , 1, 6.98, 300., 4.e+02,  80., 0., 0.)
        # ( True,  0.  , 0.03, 1, 2.  , 200., 1.e+06, 100., 0., 0.)
        # ( True,  0.  , 0.03, 1, 2.  , 200., 1.e+06, 100., 0., 0.)]
        # compute aggregate size
        num_yumi_bodies = self.gym.get_asset_rigid_body_count(yumi_asset)
        num_yumi_shapes = self.gym.get_asset_rigid_shape_count(yumi_asset)
        num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)
        num_cube_bodies = self.gym.get_asset_rigid_body_count(cube_asset)
        num_cube_shapes = self.gym.get_asset_rigid_shape_count(cube_asset)
        max_agg_bodies = num_yumi_bodies + num_table_bodies + self.num_cubes * num_cube_bodies + 7
        max_agg_shapes = num_yumi_shapes + num_table_shapes + self.num_cubes * num_cube_shapes + 3

        self.yumis = []
        self.tables = []
        self.default_cube_states = []
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
        print("made default cube states")
        # self.init_data()
        # print("init data finished")

    def compute_reward(self):
        # print("compute reward")
        cube_index = self.gym.find_actor_rigid_body_index(self.envs[0], self.cubes[0], "box", gymapi.DOMAIN_ENV)
        hand_index = self.hand_idxs[0]

        object_height = self.rigid_body_states[:, cube_index, 2].view(self.num_envs, 1)
        object_x = self.rigid_body_states[:, cube_index, 0].view(self.num_envs, 1)
        object_y = self.rigid_body_states[:, cube_index, 1].view(self.num_envs, 1)
        gripper_x = self.rigid_body_states[:, hand_index, 0].view(self.num_envs, 1)
        gripper_y = self.rigid_body_states[:, hand_index, 1].view(self.num_envs, 1)
        gripper_height = self.rigid_body_states[:, hand_index, 2].view(self.num_envs, 1)
        # print("object", object_x[0], object_y[0], object_height[0])
        # print("yumi", gripper_x[0], gripper_y[0], self.gripper_height[0])
        # print("object", object_x[1], object_y[1], object_height[1])
        # print("yumi", gripper_x[1], gripper_y[1], self.gripper_height[1])

        # height between hand and table
        height = (gripper_height - object_height).view(self.num_envs)
        rewards = torch.zeros(self.num_envs, device=self.device)

        rewards -= abs(height - 0.125) * self.height_reward_scale * 3
        rewards -= abs(gripper_x - object_x).view(self.num_envs) * self.height_reward_scale * 3
        rewards -= abs(gripper_y - object_y).view(self.num_envs) * self.height_reward_scale * 3
        rewards += (object_height - self.table_dims.z - self.cube_size / 2.).view(self.num_envs) * self.height_reward_scale * 5

        # print("diff", (gripper_x - object_x)[0], (gripper_y - object_y)[0], (height - 0.12)[0])
        # reward if success
        # first , object height > certain height
        # hand height > certain height + 0.1
        success = (object_height > self.lift_height) & (self.gripper_height > self.lift_height + 0.1)
        print("=========================success: %d===============================" % torch.sum(success))

        self.success = success.view(self.num_envs, 1)
        rewards += torch.where(success, torch.Tensor([[10.]] * self.num_envs).to(self.device),
                               torch.Tensor([[0.]] * self.num_envs).to(self.device)).view(self.num_envs)
        # penalty
        rewards -= self.action_penalty_scale
        # reset if max length reached
        # print(self.reset_buf.size())
        # exit()
        self.reset_buf = torch.where(self.success.view(self.num_envs), torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
        self.rew_buf = rewards
        # print("reward", rewards)

    def compute_observations(self):
        # print("compute observation")
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
        self.gym.refresh_jacobian_tensors(self.sim)
        if self.real_feature_input:
            self.gym.step_graphics(self.sim)

        # images =====================================================================
        if self.render_img:
            # render the camera sensors
            self.gym.render_all_camera_sensors(self.sim)

            self.gym.start_access_image_tensors(self.sim)
            #
            # User code to digest tensors
            #
            # get image tensor
            image_tensors = []

            for j in range(self.num_envs):
                _image_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[j], self.cameras[j], gymapi.IMAGE_COLOR)
                # H * W * 3
                image_tensor = gymtorch.wrap_tensor(_image_tensor)[:, :, :3].permute(2, 0, 1).contiguous()
                image_tensors.append(image_tensor)

                show_image = False
                if show_image and j == 0:
                    image_array = image_tensor.permute(1, 2, 0).cpu().numpy()
                    image = Image.fromarray(image_array).convert("RGB")
                    image.show()
                    # exit()
            self.gym.end_access_image_tensors(self.sim)

            image_tensors = torch.stack(image_tensors)
            # Normalize
            image_tensors = image_tensors / 255.
            image_tensors = self.preprocess(image_tensors)

            self.perception_output = self.model(image_tensors.view(-1, 3, 256, 256)).squeeze()
            # torch.Size([num_envs, 512])
        # =============================================================================================
        # state, gripper pos: Vec3(x, y, z)
        # [num_envs, 3]
        self.gripper_pos = self.rigid_body_states[:, self.hand_idxs[0], :3].view(self.num_envs, 3)
        # =============================================================================================
        # state, gripper width: float (meter)
        # [num_envs, 1]
        self.gripper_width = self.yumi_dof_pos[:, 4] + self.yumi_dof_pos[:, 5]    # 两个gripper的dof都是正数[0, 0.025]
        self.gripper_width = self.gripper_width.view(self.num_envs, 1)
        # =============================================================================================
        # state, gripper height: float
        # [num_envs, 1]
        self.gripper_height = self.rigid_body_states[:, self.hand_idxs[0], 2].view(self.num_envs, 1)
        # =============================================================================================
        # make state_vector
        self.state_vector = torch.cat([self.gripper_width, self.gripper_height], dim=-1)
        # =============================================================================================
        # make info_vector
        self.info_vector = torch.cat([self.state_vector, self.last_action_vector], dim=-1)
        # =============================================================================================
        # resnet input  shape:[num_envs, 3, camera_height, camera_width]
        # resnet output shape:[num_envs, 512]
        # info_vector   shape:[num_envs, 8]
        if self.real_feature_input:
            self.control_input = torch.cat([self.perception_output, self.info_vector], dim=-1)
        if not self.real_feature_input:
            # [cube_x, cube_y, cube_z, cube_cos(rz), cube_sin(rz),
            # gripper_x, gripper_y, gripper_z, gripper_cos(rz), gripper_sin(rz), gripper_width]
            cube_index = self.gym.find_actor_rigid_body_index(self.envs[0], self.cubes[0], "box", gymapi.DOMAIN_ENV)
            object_xyz = self.rigid_body_states[:, cube_index, :3].view(self.num_envs, 3)
            object_quat = self.rigid_body_states[:, cube_index, 3:7].view(self.num_envs, 4)
            gripper_quat = self.rigid_body_states[:, self.hand_idxs[0], 3:7].view(self.num_envs, 4)

            # print("object", object_xyz)
            self.control_input = torch.cat([object_xyz, object_quat, self.gripper_pos, gripper_quat, self.gripper_width], dim=-1)
        return self.control_input

    def reset_idx(self, env_ids):
        # print(env_ids)
        # exit()
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # reset yumi
        # ==========================================================================================
        # # reset yumi + noise
        pos = tensor_clamp(
            self.yumi_default_dof_pos.unsqueeze(0) + 0.0 * (
                        torch.rand((len(env_ids), self.num_yumi_dofs), device=self.device) - 0.5),
            self.yumi_dof_lower_limits, self.yumi_dof_upper_limits)
        # print(pos.size())
        # ==========================================================================================
        # reset yumi without noise
        # pos = tensor_clamp(self.yumi_default_dof_pos.repeat(len(env_ids)).view(len(env_ids), self.num_yumi_dofs),
        #                    self.yumi_dof_lower_limits, self.yumi_dof_upper_limits)
        # ==========================================================================================
        # print("pos", pos.size())
        # exit()
        # print(self.yumi_dof_pos.size())

        self.yumi_dof_pos[env_ids, :] = pos

        self.yumi_dof_vel[env_ids, :] = torch.zeros_like(self.yumi_dof_vel[env_ids])
        self.yumi_dof_targets[env_ids, :self.num_yumi_dofs] = pos
        # print("reset")

        # reset cubes   # TODO: 在pre或者post里 按一定频率检查cube是否还在object space里
        # ==========================================================================================
        if self.num_cubes > 0:
            # cube_indices = self.global_indices[env_ids, 2].flatten()    # TODO：cube的gloabl index:2
            cube_indices = self.cube_indices[env_ids].to(torch.int32).flatten()
            self.cube_states[env_ids] = self.default_cube_states[env_ids]
            self.cube_states[env_ids, :, :2] += 0.2 * torch.rand((len(env_ids), 1, 2), device=self.device) - 0.1
            arc_on_z = torch.rand((len(env_ids), 1), device=self.device) * np.pi * 2 - np.pi
            axis_angle = torch.cat([torch.zeros((len(env_ids), 2), device=self.device), arc_on_z], dim=-1)
            quat_tensor = conversions.angle_axis_to_quaternion(axis_angle).view(len(env_ids), 1, 4)  # shape is [len(env_ids), 1, 4]
            self.cube_states[env_ids, :, 3:7] = quat_tensor[:, :, [1,2,3,0]]

            # reset to default
            # self.cube_states[env_ids] = self.default_cube_states[env_ids].view(self.num_envs, 13)
            # ==========================================================================================
            # reset to random
            # resize_root_tensor_cube = torch.tensor([self.cube_middle.p.x, self.cube_middle.p.y, self.cube_middle.p.z,
            #                                         1, 0, 0, 0, 0, 0, 0, 0, 0, 0])\
            #     .to(self.device).repeat(self.num_envs).view(self.num_envs, 1, 13)
            # print("here1")
            # # random
            # resize_root_tensor_cube[:, 0, :2] += 0.2 * torch.rand((self.num_envs, 2), device=self.device) - 0.1
            # print("here2")
            # root_tensor_cube = resize_root_tensor_cube.view(self.num_envs, 13)
            # print("here3")
            # self.cube_states[env_ids] = root_tensor_cube[env_ids].view(len(env_ids), 1, -1)

            # ==========================================================================================
            # do cube reset

            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(cube_indices), len(cube_indices))
        # print("config reset cubes")
        multi_env_ids_int32 = self.yumi_indices[env_ids].to(torch.int32).flatten()  # TODO：yumi的gloabl index:0
        # multi_env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device).view(self.num_envs)[env_ids]

        # dof_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        # multi_env_ids_int32 = dof_indices[env_ids, :].flatten()
        # TODO:dof_position_target_tensor yumi是几: 是0
        # dof_state_tensor: shape (num_envs * 9(num_dofs), 2)
        # 2: 0:position, 1:velocity
        # do yumi dof target position reset
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.yumi_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))

        # print("do yumi dof target position reset")
        # do yumi dof reset

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        # print("do yumi dof reset")

        # flush the buf

        # print(env_ids)
        # print(self.progress_buf.size())
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        # print("do action")
        # print("actions:", actions)

        # actions : [delta_x, delta_y, delta_z, delta_rz, gripper_command]


        # print("dof", self.yumi_dof_pos[0])
        self.grasp_offset = 0.12  # hand中心到cube表面的距离
        self.grasp_height = self.table_dims.z + self.cube_size + self.grasp_offset

        gripper_actions = torch.where(actions[:, 4].view(self.num_envs, 1) < 0,
                                      torch.Tensor([[-0.0025, -0.0025]] * self.num_envs).to(self.device),
                                      torch.Tensor([[0.0025, 0.0025]] * self.num_envs).to(self.device))

        self.action_gripper_width = (gripper_actions[:, 0] + gripper_actions[:, 1]).view(self.num_envs, 1)
        # ======================================================================================================
        # action, gripper pos: Vec3(x, y, z)    # x, y from net output, z down 1cm per step
        self.action_gripper_pos = actions[:, :3] * self.xyz_scale
        # ======================================================================================================
        # action, gripper rot: (cos(rz), sin(rz))
        rz = actions[:, 3].view(self.num_envs, 1) * self.rz_scale
        self.action_gripper_rot = torch.cat([torch.cos(rz), torch.sin(rz)], dim=-1)
        self.actions = torch.cat([actions[:, :3] * self.xyz_scale, actions[:, 3].view(self.num_envs, 1) * self.rz_scale, gripper_actions], dim=-1)
        # =======================================================================================================
        # reshape tensor for observation
        # make last_action_vector
        self.last_action_vector = torch.cat([self.action_gripper_pos, self.action_gripper_rot, self.action_gripper_width], dim=-1)
        targets = self.yumi_dof_pos.view(self.num_envs, self.num_dofs) + self.actions

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
        # print("cube properties", self.gym.get_actor_rigid_body_properties(self.envs[0], self.cubes[0])[0].mass)
        # exit()

#####################################################################
###=========================jit functions=========================###
#####################################################################
# @torch.jit.script
# def compute_yumi_reward(
#     device, reset_buf, progress_buf, height, object_height, hand_height, certain_height,
#     num_envs, action_penalty_scale, height_reward_scale, max_episode_length
# ):
#     # height between hand and table
#     rewards = torch.zeros(num_envs)
#     rewards += height * height_reward_scale
#     # reward if success
#     # first , object height > certain height
#     # hand height > certain height + 0.1
#     success = (object_height > certain_height) & (hand_height > certain_height + 0.1)
#     success = torch.Tensor(success, device=device).view(num_envs, 1)
#     torch.where(success, torch.Tensor([[10.]] * num_envs, device=device), torch.Tensor([[0.]] * num_envs, device=device))
#     # penalty
#     rewards -= action_penalty_scale * progress_buf
#     # reset if max length reached
#     reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
#     return rewards, reset_buf

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
    yumi = YumiCube(parsed_yaml, 'cuda:0', 0, headless=True)
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

