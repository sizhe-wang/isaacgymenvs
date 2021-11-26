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

        num_obs = 520
        num_acts = 5

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_acts
        # add ===================================================================
        # 关于相机
        self.camera_same = self.cfg["env"]["camera"]["same"]
        self.camera_width = self.cfg["env"]["camera"]["width"]
        self.camera_height = self.cfg["env"]["camera"]["height"]
        self.camera_location = gymapi.Vec3(0, 0, 1.0)
        self.camera_lookat = gymapi.Vec3(0.3, -0.2, 0)

        # table
        self.table_dims = gymapi.Vec3(0.7, 1.5, 0.1)

        # cube
        self.cube_size = 0.035
        self.cube_spacing = self.cube_size + 0.005
        self.cube_middle = gymapi.Transform()
        self.cube_middle.p = gymapi.Vec3(0.3, 0., (self.table_dims.z + self.cube_size) / 2. + 0.05)

        # object reset space
        self.object_space_lower = gymapi.Vec3(0.05, -0.25, 0.)
        self.object_space_upper = gymapi.Vec3(0.65, 0.25, (self.table_dims.z + self.cube_size) / 2. + 0.05)

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
        self.yumi_default_dof_pos = to_torch([1.5072273, -1.3169852, -1.4733930, 0.7639197, -2.4363170,
                                              1.4893883, 1.2426147, 0.025, 0.025], device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # TODO：dof_state_tensor: shape (num_envs * 9(num_dofs), 2)
        # 2: 0:position, 1:velocity

        # self.yumi_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_yumi_dofs]
        # self.yumi_dof_pos = self.yumi_dof_state[:, 0]
        # self.yumi_dof_vel = self.yumi_dof_state[:, 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        print("created tensors")
        # add ===================================================================
        self.cube_dof_pos = self.rigid_body_states[:, self.cube_idxs, :3]
        self.cube_dof_vel = self.rigid_body_states[:, self.cube_idxs, 3:7]
        # /add ===================================================================
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)
        # TODO：root tensor: shape (num_envs, 3, 13)(cube只有一个的时候）
        # 3: 0:yumi, 1:table, 2:cube

        # add ===================================================================
        self.cube_states = self.root_state_tensor[:, 2]  # TODO: 确定cube在env中的序号是不是2: 是
        # /add ===================================================================

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.yumi_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * (2 + self.num_cubes), dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)
        print("finished something about cubes")
        # add ===================================================================
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "yumi")
        self.jacobian = gymtorch.wrap_tensor(_jacobian)
        # get link index of yumi hand, which we will use as end effector
        yumi_link_dict = self.gym.get_asset_rigid_body_dict(self.yumi_asset)
        yumi_hand_index = yumi_link_dict["gripper_r_base"]
        # jacobian entries corresponding to yumi hand
        self.j_eef = self.jacobian[:, yumi_hand_index - 1, :, :7]
        self.yumi_dof_pos = self.dof_state[:, 0].view(self.num_envs, 9, 1)
        self.yumi_dof_vel = self.dof_state[:, 1].view(self.num_envs, 9, 1)
        print("jacobian finished")
        # /add ===================================================================
        # add ===================================================================
        # image
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
            self.sim_params.physx.rest_offset = 0.0
            self.sim_params.physx.contact_offset = 0.001
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

        yumi_dof_stiffness = to_torch([4000, 4000, 4000, 4000, 4000, 4000, 4000, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
        # yumi_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
        yumi_dof_damping = to_torch([10, 10, 10, 10, 10, 10, 10, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)
        # yumi_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

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
        self.yumi_dof_speed_scales[[7, 8]] = 0.1
        # yumi_dof_props['effort'][7] = 200
        yumi_dof_props['effort'][7] = 20
        # yumi_dof_props['effort'][8] = 200
        yumi_dof_props['effort'][8] = 20
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
        # # compute aggregate size
        # num_yumi_bodies = self.gym.get_asset_rigid_body_count(yumi_asset)
        # num_yumi_shapes = self.gym.get_asset_rigid_shape_count(yumi_asset)
        # num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        # num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)
        # num_cube_bodies = self.gym.get_asset_rigid_body_count(cube_asset)
        # num_cube_shapes = self.gym.get_asset_rigid_shape_count(cube_asset)
        # max_agg_bodies = num_yumi_bodies + num_table_bodies + self.num_cubes * num_cube_bodies
        # max_agg_shapes = num_yumi_shapes + num_table_shapes + self.num_cubes * num_cube_shapes

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

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            yumi_actor, yumi_start_pose = create_assets.create_yumi(self.gym, env_ptr, yumi_asset, i)
            self.gym.set_actor_dof_properties(env_ptr, yumi_actor, yumi_dof_props)
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

            # camera_actor = create_assets.create_camera(self.gym, env_ptr, self.camera_location, self.camera_lookat,
            #                                            self.camera_width, self.camera_height)

            camera_actor = create_assets.create_camera_attach(self.gym, env_ptr, self.camera_width, self.camera_height,
                                                              hand_handle)

            self.envs.append(env_ptr)
            self.yumis.append(yumi_actor)
            self.tables.append(table_actor)
            self.cameras.append(camera_actor)
            print("create%d"%i)

        # 无所谓找的是哪个环境的，所有环境里都一样，指的是某个关节在actor里的编号
        print("create end")
        self.hand_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.yumis[0], "yumi_link_7_r")
        print("found hand handle")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.yumis[0], "gripper_r_finger_l")
        print("found left finger handle")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.yumis[0], "gripper_r_finger_r")
        print("found right finger handle")
        self.default_cube_states = to_torch(self.default_cube_states, device=self.device, dtype=torch.float
                                            ).view(self.num_envs, self.num_cubes, 13)
        print("made default cube states")
        # self.init_data()
        # print("init data finished")

    # def init_data(self):
    #     hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.yumis[0], "yumi_link_7_r")
    #     lfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.yumis[0], "gripper_r_finger_l")
    #     rfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.yumis[0], "gripper_r_finger_r")
    #
    #     hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
    #     lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
    #     rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)
    #
    #     finger_pose = gymapi.Transform()
    #     finger_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5
    #     finger_pose.r = lfinger_pose.r
    #
    #     hand_pose_inv = hand_pose.inverse()     # 逆
    #     grasp_pose_axis = 1
    #     # TODO:yumi_local_grasp_pose是什么？什么计算原理？直接控制手掌位置？绝对位置？
    #     yumi_local_grasp_pose = hand_pose_inv * finger_pose
    #     yumi_local_grasp_pose.p += gymapi.Vec3(*get_axis_params(0.04, grasp_pose_axis))
    #     self.yumi_local_grasp_pos = to_torch([yumi_local_grasp_pose.p.x, yumi_local_grasp_pose.p.y,
    #                                             yumi_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
    #     self.yumi_local_grasp_rot = to_torch([yumi_local_grasp_pose.r.x, yumi_local_grasp_pose.r.y,
    #                                             yumi_local_grasp_pose.r.z, yumi_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))
    #
    #     drawer_local_grasp_pose = gymapi.Transform()
    #     drawer_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.01, grasp_pose_axis, 0.3))
    #     drawer_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
    #     self.drawer_local_grasp_pos = to_torch([drawer_local_grasp_pose.p.x, drawer_local_grasp_pose.p.y,
    #                                             drawer_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
    #     self.drawer_local_grasp_rot = to_torch([drawer_local_grasp_pose.r.x, drawer_local_grasp_pose.r.y,
    #                                             drawer_local_grasp_pose.r.z, drawer_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))
    #
    #     self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
    #     self.drawer_inward_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
    #     self.gripper_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
    #     self.drawer_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
    #
    #     self.yumi_grasp_pos = torch.zeros_like(self.yumi_local_grasp_pos)
    #     self.yumi_grasp_rot = torch.zeros_like(self.yumi_local_grasp_rot)
    #     self.yumi_grasp_rot[..., -1] = 1  # xyzw
    #     self.drawer_grasp_pos = torch.zeros_like(self.drawer_local_grasp_pos)
    #     self.drawer_grasp_rot = torch.zeros_like(self.drawer_local_grasp_rot)
    #     self.drawer_grasp_rot[..., -1] = 1
    #     self.yumi_lfinger_pos = torch.zeros_like(self.yumi_local_grasp_pos)
    #     self.yumi_rfinger_pos = torch.zeros_like(self.yumi_local_grasp_pos)
    #     self.yumi_lfinger_rot = torch.zeros_like(self.yumi_local_grasp_rot)
    #     self.yumi_rfinger_rot = torch.zeros_like(self.yumi_local_grasp_rot)

    def compute_reward(self):
        object_height = self.rigid_body_states[:, self.cubes[0], 2].view(self.num_envs, 1)
        # print("self.object_height", self.object_height.size())
        # print("self.gripper_height", self.gripper_height.size())
        # exit()
        # self.rew_buf[:], self.reset_buf[:] = compute_yumi_reward(
        #     self.device, self.reset_buf, self.progress_buf, self.grasp_height - self.table_dims.z, self.object_height, self.gripper_height, self.lift_height
        #     self.num_envs, self.action_penalty_scale, self.height_reward_scale, self.max_episode_length
        # )
        # height between hand and table
        height = (self.gripper_height - self.table_dims.z).view(self.num_envs)
        rewards = torch.zeros(self.num_envs, device=self.device)
        # print("height:", height.size())
        # print("height * self.height_reward_scale", (height * self.height_reward_scale).size())
        # print("gripper_height", self.gripper_height.size())
        # exit()
        rewards += (self.env_spacing - height) * self.height_reward_scale
        # reward if success
        # first , object height > certain height
        # hand height > certain height + 0.1
        success = (object_height > self.lift_height) & (self.gripper_height > self.lift_height + 0.1)
        # print(success.size())
        # print(success)
        # exit()
        self.success = success.view(self.num_envs, 1)
        torch.where(success, torch.Tensor([[10.]] * self.num_envs).to(self.device),
                    torch.Tensor([[0.]] * self.num_envs).to(self.device))
        # penalty
        # print("self.progress_buf", self.progress_buf.size())
        # print("self.action_penalty_scale * self.progress_buf", (self.action_penalty_scale * self.progress_buf).size())
        rewards -= self.action_penalty_scale * self.progress_buf
        # reset if max length reached
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
        self.rew_buf = rewards
        # print("rewards", self.rew_buf)

    def compute_observations(self):
        # obsevations:
        #   images: perception_output: [num_envs, 512]
        #   last action:
        #       gripper pos: Vec3(x, y, z)
        #       gripper rot: (cos(rz), sin(rz))     or Vec3(rx, ry, rz)   rx, ry固定
        #       gripper width: float
        #   state:
        #       gripper width: float
        #       gripper height real得不到          ? add
        #       gripper pos: Vec3(x, y, z)        ? delete
        #       gripper rot: (cos, sin) 只绕z轴    ? delete

        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_jacobian_tensors(self.sim)

        # self.gym.render_all_camera_sensors(self.sim)

        # camera_tensor = gym.get_camera_image_gpu_tensor(sim, env, cam_handle, gymapi.IMAGE_COLOR)
        # torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
        # images =====================================================================
        if self.render_img:
            # render the camera sensors
            self.gym.render_all_camera_sensors(self.sim)
            # save.save_images(gym, sim, num_envs, envs, camera_handles, "yumi_image")
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
                if show_image:
                    image_array = image_tensor.permute(1, 2, 0).cpu().numpy()
                    image = Image.fromarray(image_array).convert("RGB")
                    image.show()
                    # exit()
            self.gym.end_access_image_tensors(self.sim)

            image_tensors = torch.stack(image_tensors)
            # Normalize
            image_tensors = image_tensors / 255.
            image_tensors = self.preprocess(image_tensors)
            # print("image_tensors:")
            # print(image_tensors)
            # print(image_tensors.shape)  # torch.Size([num_envs, 3, 256, 256])
            self.perception_output = self.model(image_tensors.view(-1, 3, 256, 256)).squeeze()
            # print("perception output:")
            # print(self.perception_output)
            # print(self.perception_output.size())    # torch.Size([num_envs, 512])
            # exit()
        # =============================================================================================
        # state, gripper pos: Vec3(x, y, z)
        # [num_envs, 3]
        # self.rigid_body_states曾view(self.num_envs, -1, 13)，franka_cube_ik里是[num_envs * num_rigid_bodies, 13],
        # 所以序号不一样，第0个env里hand序号10，每个env里hand序号都和第一个一样
        self.gripper_pos = self.rigid_body_states[:, self.hand_idxs[0], :3].view(self.num_envs, 3)
        # print(self.gripper_pos.size())  # torch.Size([16, 3])
        # =============================================================================================
        # state, gripper rot: Vec3(rx, ry, rz)
        # [num_envs, 3]
        self.gripper_rot = R.from_quat(self.rigid_body_states[:, self.hand_idxs[0], 3:7].cpu()).as_euler('xyz', degrees=False)\
            if self.extrinsic_rotations else R.from_quat(self.rigid_body_states[:, self.hand_idxs[0], 3:7].cpu()).as_euler('XYZ', degrees=False)
        # as_euler(): 'xyz' extrinsic rotations; 'XYZ' intrinsic rotations TODO:extrinsic or intrinsic
        # print(self.gripper_rot.shape)   # list: (num_envs, 3)
        # exit()
        self.gripper_rot = torch.Tensor(self.gripper_rot).to(self.device)
        self.gripper_rot = torch.transpose(torch.stack([torch.cos(self.gripper_rot[:, 2]), torch.sin(self.gripper_rot[:, 2])]), 0, 1)
        # print(self.gripper_rot.size())  # torch.Size([num_envs, 2])
        # exit()
        # =============================================================================================
        # state, gripper width: float (meter)
        # [num_envs, 1]
        self.gripper_width = self.yumi_dof_pos[:, 7, 0] + self.yumi_dof_pos[:, 8, 0]    # 两个gripper的dof都是正数[0, 0.025]
        self.gripper_width = self.gripper_width.view(self.num_envs, 1)
        # print(self.gripper_width)
        # print(self.gripper_width.size())    # torch.Size([num_envs, 1])
        # exit()
        # =============================================================================================
        # state, gripper height: float
        # [num_envs, 1]
        self.gripper_height = self.rigid_body_states[:, self.hand_idxs[0], 2]
        self.gripper_height = self.gripper_height.view(self.num_envs, 1)
        # print("height", self.gripper_height)
        # print(self.gripper_height.size())   # torch.Size([num_envs, 1])
        # exit()
        # =============================================================================================
        # make state_vector
        # self.state_vector = torch.cat([self.gripper_pos, self.gripper_rot, self.gripper_width, self.gripper_height], dim=-1)
        self.state_vector = torch.cat([self.gripper_width, self.gripper_height], dim=-1)
        # print(state_vector)
        # print(self.state_vector.size())  # torch.Size([num_envs, 2])  # 2 = width(1) + height(1)
        # exit()
        # =============================================================================================
        # make info_vector
        self.info_vector = torch.cat([self.state_vector, self.last_action_vector], dim=-1)
        # print(self.info_vector)
        # print(self.info_vector.size())  # torch.Size([num_envs, 8])
        # 8 = state(width(1) + height(1)) + last_action(pos(3) + rot(2) + width(1))
        # exit()
        # =============================================================================================
        # make PPO input(control_input) # TODO: PPO input shape
        # resnet input  shape:[num_envs, 3, camera_height, camera_width]
        # resnet output shape:[num_envs, 512]
        # info_vector   shape:[num_envs, 8]
        self.control_input = torch.cat([self.perception_output, self.info_vector], dim=-1)
        # print(self.control_input.size())    # torch.Size([num_envs, 520])
        # exit()
        return self.control_input

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # reset yumi
        # ==========================================================================================
        # # reset yumi + noise
        # pos = tensor_clamp(
        #     self.yumi_default_dof_pos.unsqueeze(0) + 0.25 * (
        #                 torch.rand((len(env_ids), self.num_yumi_dofs), device=self.device) - 0.5),
        #     self.yumi_dof_lower_limits, self.yumi_dof_upper_limits)
        # ==========================================================================================
        # reset yumi without noise
        pos = tensor_clamp(self.yumi_default_dof_pos.repeat(self.num_envs).view(self.num_envs, self.num_yumi_dofs),
                           self.yumi_dof_lower_limits, self.yumi_dof_upper_limits)
        # ==========================================================================================

        self.yumi_dof_pos[env_ids, :] = pos.view(self.num_envs, 9, 1)
        self.yumi_dof_vel[env_ids, :] = torch.zeros_like(self.yumi_dof_vel[env_ids])
        self.yumi_dof_targets[env_ids, :self.num_yumi_dofs] = pos
        print("config reset yumi")

        # reset cubes   # TODO: 在pre或者post里 按一定频率检查cube是否还在object space里
        # ==========================================================================================
        if self.num_cubes > 0:
            cube_indices = self.global_indices[env_ids, 2].flatten()    # TODO：cube的gloabl index:2
            # reset to default
            # self.cube_states[env_ids] = self.default_cube_states[env_ids].view(self.num_envs, 13)
            # ==========================================================================================
            # reset to random
            resize_root_tensor_cube = torch.tensor([self.cube_middle.p.x, self.cube_middle.p.y, self.cube_middle.p.z,
                                                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0])\
                .to(self.device).repeat(self.num_envs * 3).view(self.num_envs, 3, 13)

            resize_root_tensor_cube[:, 2, :2] += 0.2 * torch.rand((self.num_envs, 2), device=self.device) - 0.1
            # resize_root_tensor_cube[:, 2, 3: 7] += 0.2 * (torch.rand((self.num_envs, 4), device=self.device) - 0.1)
            root_tensor_cube = resize_root_tensor_cube.view(self.num_envs * 3, 13)
            self.cube_states[env_ids] = root_tensor_cube[env_ids].view(self.num_envs, 13)

            # ==========================================================================================
            # do cube reset
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(cube_indices), len(cube_indices))
        print("config reset cubes")
        multi_env_ids_int32 = self.global_indices[env_ids, 0].flatten() # TODO：yumi的gloabl index:0
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
        print("do yumi dof target position reset")
        # do yumi dof reset
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        print("do yumi dof reset")

        # flush the buf
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        # actions: delta num_envs * [x, y, z, rx, ry, rz]   [-1.0, 1.0]
        # do actions
        #   action:
        #       gripper pos: Vec3(x, y, z)      # x, y: net output, z: down 0.01m per step
        #       gripper rot: (cos(rz), sin(rz))     or Vec3(rx, ry, rz)   rx, ry固定  # rz: net output
        #       gripper width: float            # 到一定高度就尝试一次抓取: 低于一定高度就闭合夹爪
        self.actions = actions.clone()  # 没找到action在哪里设置的： action是网络的输出
        # print("net output", actions)
        # print(actions.size())
        # exit()

        # actions: 变化量，actions.size()是[num_envs, num_dofs]
        # 提取所需的action ======================================================================================
        # operational_pos [dx, dy, dz, drz, gripper_cmd]
        operational_pos = actions.clone()
        # 提取所需的action ======================================================================================
        # 提取所需的action ======================================================================================
        # # 将actions[:, :7]的joint space的q转换为operational space的x
        # operational_pos = (self.j_eef @ self.actions[:, :7].unsqueeze(-1)).view(self.num_envs, 6)
        # # self.operational_pos = operational_pos  # TODO:delete this line
        # # TODO: control_ik里的公式，damping代表什么，有没有必要解operational_pos的时候考虑damping，考虑damping怎么计算
        # # 验证过：在control_ik里damping=0，operational_pos与dpose近似相等，误差e-07级别及以下
        # # print(operational_pos)
        # # print(operational_pos.size())     # torch.Size([num_envs, 6])
        # # exit()
        # ======================================================================================================
        # ======================================================================================================
        # gripper ========================================
        # action, gripper width: float
        self.grasp_offset = 0.12  # hand中心到cube表面的距离
        self.grasp_height = self.table_dims.z + self.cube_size + self.grasp_offset
        # print(self.grasp_height)
        # 低于self.grasp_height就闭合夹爪, 否则维持原样（已经抓到的继续闭合，还没抓的继续打开）
        # print((self.rigid_body_states[:, self.hand_idxs[0], 2].view(self.num_envs, 1) < self.grasp_height).size())
        # print((torch.Tensor([[0., 0.]] * self.num_envs).to(self.device)).size())
        # print((self.yumi_dof_pos[:, 7:9, :].view(self.num_envs, -1)).size())
        # =================================================================================
        # # 低于self.grasp_height就闭合夹爪, 否则维持原样（已经抓到的继续闭合，还没抓的继续打开）
        # gripper_actions = torch.where(self.rigid_body_states[:, self.hand_idxs[0], 2].view(self.num_envs, 1) < self.grasp_height,
        #                                 torch.Tensor([[-0.025, -0.025]] * self.num_envs).to(self.device),
        #                                 self.yumi_dof_pos[:, 7:9, :].view(self.num_envs, -1))
        # =================================================================================
        # gripper_cmd [-1,1] [-1, 0]close:-=0.0025 [0, 1]open:+=0.0025
        gripper_actions = torch.where(operational_pos[:, 4].view(self.num_envs, 1) < 0,
                                      torch.Tensor([[-0.0025, -0.0025]] * self.num_envs).to(self.device),
                                      torch.Tensor([[0.0025, 0.0025]] * self.num_envs).to(self.device))

        # print(gripper_actions)
        # print(gripper_actions.size())
        # TODO:【checked】reset的时候self.dof_state[:, 7:9]要设为打开【0.025， 0.025】---->确保__init__里self.yumi_default_dof_pos[7:9]要设为打开【0.025， 0.025】
        self.action_gripper_width = (gripper_actions[:, 0] + gripper_actions[:, 1]).view(self.num_envs, 1)
        # print(self.action_gripper_width)
        # print(self.action_gripper_width.size())
        # exit()
        # =======================================================================================================
        # ======================================================================================================
        # action, gripper pos: Vec3(x, y, z)    # x, y from net output, z down 1cm per step
        self.action_gripper_pos = operational_pos[:, :3] * self.xyz_scale
        # print(self.action_gripper_pos)
        # print(self.action_gripper_pos.size())       # torch.Size([num_envs, 3])
        # exit()
        # # z:down 1cm per step
        # # now_operational_pos = (self.j_eef @ self.rigid_body_states[:, self.hand_idxs[0], :7].unsqueeze(-1)).view(self.num_envs, 6)
        # # self.action_gripper_pos[:, 2] = -0.01
        #
        # # count close
        # self.gripper_close = torch.where(operational_pos[:, 4].view(self.num_envs, 1) < 0,
        #                                  self.gripper_close + 1,
        #                                  torch.Tensor([[0.]] * self.num_envs).to(self.device))
        #
        #
        #
        # # get gripper height
        # self.gripper_height = self.rigid_body_states[:, self.hand_idxs[0], 2]
        # self.gripper_height = self.gripper_height.view(self.num_envs, 1)
        # # count gripper height under grasp height
        # self.count_under_height = torch.where(self.gripper_height < self.grasp_height,
        #                                       self.count_under_height + 1,
        #                                       torch.Tensor([[0.]] * self.num_envs).to(self.device))
        # # if count_under_height > 10, set up flags for up_times(20) times
        # self.up_flags = torch.where(self.count_under_height >= 10,
        #                             torch.Tensor([[self.up_times]] * self.num_envs).to(self.device),
        #                             torch.where(self.up_flags > 0, self.up_flags - 1,
        #                             torch.Tensor([[0]] * self.num_envs).to(self.device)))
        # # print("self.count_under_height", self.count_under_height)
        # # # print("torch.Tensor([[self.up_times]] * self.num_envs).to(self.device)", torch.Tensor([[self.up_times]] * self.num_envs).to(self.device).size())
        # # print("self.up_flags", self.up_flags)
        # # # print("self.up_flags", self.up_flags.size())
        # # exit()
        # # 0.01 if gripper height under grasp height 10 times recently, else -0.01
        # # print("self.gripper_close", self.gripper_close.size())
        # # print("self.gripper_height", self.gripper_height.size())
        # # print("make", torch.Tensor([[0.01]] * self.num_envs).to(self.device).size())
        # # print("self.action_gripper_pos[:, 2]", self.action_gripper_pos[:, 2].size())
        # # exit()
        # # self.action_gripper_pos[:, 2] = torch.where((self.gripper_close >= 10) | (self.gripper_height < self.grasp_height),
        # #                                             torch.Tensor([[0.01]] * self.num_envs).to(self.device),
        # #                                             torch.Tensor([[-0.01]] * self.num_envs).to(self.device)).view(self.num_envs)
        # self.action_gripper_pos[:, 2] = torch.where(self.up_flags > 0,
        #                                             torch.Tensor([[self.xyz_scale]] * self.num_envs).to(self.device),
        #                                             torch.Tensor([[-self.xyz_scale]] * self.num_envs).to(self.device)).view(self.num_envs)

        # ======================================================================================================
        # action, gripper rot: (cos(rz), sin(rz))
        rz = operational_pos[:, 3].view(self.num_envs, 1) * self.rz_scale
        self.action_gripper_rot = torch.cat([torch.cos(rz), torch.sin(rz)], dim=-1)
        # print(self.action_gripper_rot)   # torch.Size([num_envs, 2])
        # rot_euler: rx, ry 保持不变，rz from net output
        rot_euler = torch.Tensor([[0., 0., 0.] * self.num_envs]).to(self.device).view(self.num_envs, 3)
        rot_euler[:, 2] = rz.view(self.num_envs)
        # print(rot_euler)     # torch.Size([num_envs, 2])
        # exit()
        # ======================================================================================================
        # # gripper ========================================
        # # action, gripper width: float
        # self.grasp_offset = 0.12    # hand中心到cube表面的距离
        # self.grasp_height = self.table_dims.z + self.cube_size + self.grasp_offset
        # # print(self.grasp_height)
        # # 低于self.grasp_height就闭合夹爪, 否则维持原样（已经抓到的继续闭合，还没抓的继续打开）
        # # print((self.rigid_body_states[:, self.hand_idxs[0], 2].view(self.num_envs, 1) < self.grasp_height).size())
        # # print((torch.Tensor([[0., 0.]] * self.num_envs).to(self.device)).size())
        # # print((self.yumi_dof_pos[:, 7:9, :].view(self.num_envs, -1)).size())
        # # =================================================================================
        # # # 低于self.grasp_height就闭合夹爪, 否则维持原样（已经抓到的继续闭合，还没抓的继续打开）
        # # gripper_actions = torch.where(self.rigid_body_states[:, self.hand_idxs[0], 2].view(self.num_envs, 1) < self.grasp_height,
        # #                                 torch.Tensor([[-0.025, -0.025]] * self.num_envs).to(self.device),
        # #                                 self.yumi_dof_pos[:, 7:9, :].view(self.num_envs, -1))
        # # =================================================================================
        # # gripper_cmd [-1,1] [-1, 0]close:-=0.0025 [0., 1]open:+=0.0025
        # gripper_actions = torch.where(operational_pos[:, 4].view(self.num_envs, 1) < 0,
        #                                 torch.Tensor([[-0.0025, -0.0025]] * self.num_envs).to(self.device),
        #                                 torch.Tensor([[0.0025, 0.0025]] * self.num_envs).to(self.device))
        #
        # # print(gripper_actions)
        # # print(gripper_actions.size())
        # # TODO:【checked】reset的时候self.dof_state[:, 7:9]要设为打开【0.025， 0.025】---->确保__init__里self.yumi_default_dof_pos[7:9]要设为打开【0.025， 0.025】
        # self.action_gripper_width = (gripper_actions[:, 0] + gripper_actions[:, 1]).view(self.num_envs, 1)
        # # print(self.action_gripper_width)
        # # print(self.action_gripper_width.size())
        # # exit()
        # # =======================================================================================================
        # =======================================================================================================
        # net output输出全部9个dof (joint space) ，在简化的action (operational space)下，有一部分是启发式的，
        # 从上面的计算得到operational space下的delta x[x, y, z, rx, ry, rz],
        # 再转换到joint space下的前7个dof，再合并grippper action得到最终的control delta action (9 dofs, joint space)
        # =======================================================================================================
        # 得到operational space下的delta x[x, y, z, rx, ry, rz]
        script_operational_pos = torch.cat([self.action_gripper_pos, rot_euler], dim=-1)  # TODO:device
        # print(script_operational_pos)
        # print(script_operational_pos.size())      # torch.Size([num_envs, 6])
        # 转换到joint space下的前7个dof
        # print("script_operational_pos.unsqueeze(-1)", script_operational_pos.unsqueeze(-1))
        # print("self.j_eef", self.j_eef)
        script_joint_q = control_ik(script_operational_pos.unsqueeze(-1), self.device, self.j_eef, self.num_envs, self.damping)   # TODO: damping
        # print(script_joint_q)
        # print(script_joint_q.size())    # torch.Size([num_envs, 7])
        # 合并grippper action得到最终的control delta action (9 dofs, joint space)
        self.actions = torch.cat([script_joint_q, self.yumi_dof_pos[:, 7:9, :].view(self.num_envs, -1) + gripper_actions], dim=-1)  # TODO:shape, device
        # print("final dof actions", self.actions)
        # print("script_joint_q", script_joint_q)
        # print("script_operational_pos", script_operational_pos)
        # print()
        # print(self.actions.size())  # torch.Size([num_envs, 9])
        # exit()
        # =======================================================================================================
        # reshape tensor for observation
        # make last_action_vector
        self.last_action_vector = torch.cat([self.action_gripper_pos, self.action_gripper_rot, self.action_gripper_width], dim=-1)
        # print(self.last_action_vector)
        # print(self.last_action_vector.size())    # torch.Size([num_envs, 6]) # 6 = pos(3) + rot(2) + width(1)
        # exit()

        # targets = self.yumi_dof_targets + self.yumi_dof_speed_scales * self.dt * self.actions * self.action_scale
        targets = self.yumi_dof_targets + self.actions
        # print(self.actions)
        # exit()
        # targets = self.yumi_dof_targets.clone().to(self.device)
        # targets[:, :7] = self.yumi_dof_targets[:, :7] + self.yumi_dof_speed_scales[:7] * self.dt * self.actions[:, :7] * self.action_scale
        # targets[:, 7:9] = self.yumi_dof_speed_scales[7:9] * self.dt * self.actions[:, 7:9] * self.action_scale

        # targets[:, 7:9] = self.actions[:, 7:9]
        # targets = self.yumi_dof_targets[:, :self.num_yumi_dofs] + actions
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
    yumi = YumiCube(parsed_yaml, 'cuda:0', 0, headless=False)
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
        operational_action = torch.Tensor([0., 1, 1, 0, 0.]).to(yumi.device).repeat(yumi.num_envs).view(yumi.num_envs, 5)
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
        print('time cost is :', time.time() - time_start)
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

