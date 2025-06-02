for i in 0 1 2 3 4 5 6 7 8 9
do
	cmd="python isaacgymenvs/train.py \
		num_envs=4096 \
		task=go1 \
		reward_scale=1 \
		experiment=0120_ppo_go1_$i \
		task_id=[$i] \
		train=meta-world-go1-PPO-Vanilla \
		seed=43 \
		wandb_activate=True \
		wandb_entity=meta-world \
		wandb_project=meta-world-ig \
		sim_device=cuda:$1 \
		rl_device=cuda:$1 \
		graphics_device_id=0 \
		record_videos=True \
		max_iterations=10000"
	echo $cmd
	$cmd
done