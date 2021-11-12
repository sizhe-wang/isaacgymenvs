### 对文件的说明

1. 文件夹isaacgym_python_examples
    1. yumi_env.py——不包含rl
        * path: `isaacgym/python/examples`

        * 附加：`isaacgym/python/examples/myutils`
    2. yumi_action_test.py——不包含rl，对action的测试
        * path: `isaacgym/python/examples`

        * 附加：`isaacgym/python/examples/myutils`
2. 文件夹IsaacGymEnvs_isaacgymenvs
    1. yumi_cube.py——rl下，对action的测试
        * path: `IsaacGymEnvs/isaacgymenvs/tasks`

        * 附加：
          1. `IsaacGymEnvs/isaacgymenvs/tasks/myutils`
          2. `IsaacGymEnvs/isaacgymenvs/cfg/train/YumiCubePPO.yaml`
          3. `IsaacGymEnvs/isaacgymenvs/cfg/task/YumiCube.yaml`

        * 需要改动：
          `IsaacGymEnvs/isaacgymenvs/tasks/__init__.py`
          1. 加一个import:
              ```python
              from isaacgymenvs.tasks.yumi_cube import YumiCube
              ```
          2. 在isaacgym_task_map中，加一项（大概在第58行）：
              ```python
              "YumiCube": YumiCube,
              ```

### 注意事项

* `isaacgym/python/examples`和`IsaacGymEnvs/isaacgymenvs/tasks`下的两个`myutils`中的文件有一些不同