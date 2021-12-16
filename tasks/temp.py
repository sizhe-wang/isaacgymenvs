#init

# image collector
        self.step_counter = 0
        self.image_tensors = []
        self.num_save = 55000


#observation

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

    self.obs_buf = torch.cat([object_xyz, rot_cube, gripper_pos, rot_gripper_z, gripper_width], dim=-1)
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