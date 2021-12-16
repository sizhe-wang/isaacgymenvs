#xvfb-run python train.py task=YumiCube headless=True num_envs=1004
python train.py task=YumiCollect headless=true num_envs=256 graphics_device_id=0 sim_device='cuda:0' rl_device='cuda:0' ppo_device='cuda:0'
#python train.py task=YumiCube headless=True num_envs=256 graphics_device_id=1 sim_device='cuda:1' rl_device='cuda:1' ppo_device='cuda:1'
