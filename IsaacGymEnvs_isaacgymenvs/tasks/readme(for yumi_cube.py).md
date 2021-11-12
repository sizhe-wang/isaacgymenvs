# yumi_cube.py

## 主要功能

在isaacgym的rl环境下搭建yumi环境（文件夹`IsaacGymEnvs/isaacgymenvs/tasks`）

* 调用`pre_physics_step(d_action)`实现以下action
  1. 其他不变，z下降1cm
  2. 其他不变 x, y 随机[0，0.01]
  3. 其他不变，yaw旋转
  4. 其他不变，夹爪随机开合
  5. 其他不变，夹爪开 (不用担心超出dof limit，之后函数里有截断，正的就是不断打开）
  6. 其他不变，夹爪合 (不用担心超出dof limit，之后函数里有截断，负的就是不断闭合）
  7. 其他不变 x+=0.01
  8. 其他不变 y+=0.01
* 每100帧reset

### 更改以选择action：

* yumi_cube.py第613-617行
  1. 若选择action1-3，7-8：解注下面第2、3行，注释第5行，并修改第2行`dpose = dpose1`的`1`为1-3，7-8
      ```python
      # action1-3,7-8(下面两行）
      dpose = dpose1  # TODO:choose action
      d_action[:, :7] = control_ik(dpose, yumi.device, yumi.j_eef, yumi.num_envs)
      # # action4-6（下面一行）
      # d_action[:, 7:9] = gripper_action5
      ```

  2. 若选择action4-6：解注下面第5行，注释第2、3行，并修改第5行`d_action[:, 7:9] = gripper_action5`的`5`为4-6
      ```python
      # action1-3,7-8(下面两行）
      # dpose = dpose1  # TODO:choose action
      # d_action[:, :7] = control_ik(dpose, yumi.device, yumi.j_eef, yumi.num_envs)
      # # action4-6（下面一行）
      d_action[:, 7:9] = gripper_action5
      ```

### 更改以设置yumi初始dof：

* yumi_cube.py第127-128行
  ```python
  self.yumi_default_dof_pos = to_torch([0.5003859, -1.1831587, -0.01762783, -0.34278267, -1.38648956,
                                        2.04657308, -0.33545228, 0.012, 0.012], device=self.device)
  ```