# yumi_cube.py

## version 3

#### 主要功能

1. 添加了obeservation和action的部分
2. 添加了headless   
    1. 需要更改`YumiCube.yaml`里的`env：enableCameraSensors`为`True`
    2. 需要更改`vec_task.py`第78行（原来的读取`enableCameraSensors`有bug）
3. 添加了 info_vector
    * info_vector包含除resnet输出外的其他全部需要作为control input的信息
    * info_vetor (size`[num_envs,13]`)
      * 包含`state(pos(3) + rot(2) + width(1) + height(1))` +` last_action(pos(3) + rot(2) + width(1))`
4. 添加了control_input
    * (size`[num_envs,525]`)包含`resnet output(512)` +` info vector(13)`
5. camera附着在手腕，更新了`myutils/create_assets.py`

#### TODO

1. 四元数和欧拉角
    * 怎么验证四元数转换到欧拉角对不对
      * 用的是`scipy.spatial.transform.Rotation`库
    * `as_euler`应该用"`intrinsic rotations`" or "`extrinsic rotations`"?
      * 在将hand rot从四元数转换到欧拉角时:"intrinsic rotations" or "extrinsic rotations"？
        in `__init__`
        ```python
        # hand rot:"intrinsic rotations" or "extrinsic rotations"
        self.extrinsic_rotations = True
        ```

        目前是extrinsic rotations
2. control_ik里的damping
    * control_ik里的公式，damping代表什么，有没有必要解operational_x的时候考虑damping，考虑damping怎么计算
      * in `pre_physics_step`
      * control_ik:
        ```python
        j_eef_T = torch.transpose(j_eef, 1, 2)
        lmbda = torch.eye(6, device=device) * (damping ** 2)    # torch.eye(6) 6维单位阵
        u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7) 
        ```

      * 目前解算operational space 的x:
        ![\dot{\textbf{x}} = \textbf{J} ](https://s0.wp.com/latex.php?latex=%5Cdot%7B%5Ctextbf%7Bx%7D%7D+%3D+%5Ctextbf%7BJ%7D+%5C%3B+%5Cdot%7B%5Ctextbf%7Bq%7D%7D&bg=ffffff&fg=555555&s=0&c=20201002)
        ```python
        operational_x = (self.j_eef @ self.actions[:, :7].unsqueeze(-1)).view(self.num_envs, 6)
        ```

        没有考虑damping
3. rewards还没写

## version 2

#### 主要功能

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

#### 更改以选择action：

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

#### 更改以设置yumi初始dof：

* yumi_cube.py第127-128行
  ```python
          self.yumi_default_dof_pos = to_torch([0.5003859, -1.1831587, -0.01762783, -0.34278267, -1.38648956,
                                                2.04657308, -0.33545228, 0.012, 0.012], device=self.device)
  ```