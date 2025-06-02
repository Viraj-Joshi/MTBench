for i in 41 42 43 44 45
do
	cmd="python isaacgymenvs/train.py \
		num_envs=4096 \
		task=final_go1_easy \
		reward_scale=1 \
		experiment=0220_ppo_go1_flat_vanilla_seed_$i \
		task_id=[20] \
		train=go1-PPO-clip-Vanilla \
		seed=$i \
		wandb_activate=True \
		wandb_entity=meta-world \
		wandb_project=meta-world-ig \
		sim_device=cuda:$1 \
		rl_device=cuda:$1 \
		graphics_device_id=0 \
		record_videos=False \
		max_iterations=1000"
	echo $cmd
	$cmd
done