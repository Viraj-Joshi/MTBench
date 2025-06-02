for i in 41 42 43 44 45
# 1/5 environments are flat ground to encourage moving forward
do
	cmd="python isaacgymenvs/train.py \
		num_envs=4096 \
		task=go1_no_penalty \
		reward_scale=1 \
		experiment=0131_ppo_go1_hard_seed_$i \
		task_id=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] \
		train=meta-world-go1-PPO-Vanilla \
		seed=$i \
		wandb_activate=True \
		wandb_entity=meta-world \
		wandb_project=meta-world-ig \
		sim_device=cuda:$1 \
		rl_device=cuda:$1 \
		graphics_device_id=0 \
		record_videos=True \
		max_iterations=2000"
	echo $cmd
	$cmd
done