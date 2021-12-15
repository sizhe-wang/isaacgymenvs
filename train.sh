#xvfb-run python train.py task=YumiCube headless=True num_envs=1024
python train.py task=YumiCube headless=True num_envs=256 graphics_device_id=2 sim_device='cuda:2' rl_device='cuda:2' ppo_device='cuda:2'
#python train.py task=YumiCube headless=True num_envs=256 graphics_device_id=1 sim_device='cuda:1' rl_device='cuda:1' ppo_device='cuda:1'
