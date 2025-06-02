for i in 10 # 11 12 13 14 15 16 17 18 19
# 1/5 environments are flat ground to encourage moving forward
do
	cmd="python isaacgymenvs/train.py \
		num_envs=4096 \
		task=go1_no_penalty \
		reward_scale=1 \
		experiment=0124_ppo_go1_single_$i \
		task_id=[$i,$i,$i,$i,20] \
		train=meta-world-go1-PPO-Vanilla \
		seed=43 \
		wandb_activate=True \
		wandb_entity=meta-world \
		wandb_project=meta-world-ig \
		sim_device=cuda:$1 \
		rl_device=cuda:$1 \
		graphics_device_id=0 \
		record_videos=True \
		max_iterations=5000"
	echo $cmd
	$cmd
done