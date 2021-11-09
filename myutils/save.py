# TODO: save something, images only currently.
import os
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from myutils.print_info import printInfo

import math
import numpy as np
import torch


def save_images(gym, sim, num_envs, envs, camera_handles, file_name):
    if not os.path.exists(file_name):
        os.mkdir(file_name)
    for j in range(num_envs):
        # color_image = gym.get_camera_image(sim, env, camera_handles[0], gymapi.IMAGE_COLOR)
        rgb_filename = file_name + "/rgb_env%d.png" % j
        gym.write_camera_image_to_file(sim, envs[j], camera_handles[0], gymapi.IMAGE_COLOR, rgb_filename)
        # print("camera_handles")
        # print(camera_handles)
        if printInfo.print_save_process:
            print("saved%s" % rgb_filename)
    print("Successfully saved %d images" % num_envs)
