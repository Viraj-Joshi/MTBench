#!/bin/bash

for i in 42 43 44 45 46 47 48 49 50 51
do

	cmd="python isaacgymenvs/train.py \
		task_id=[40,38,31,15,17,18,4,28,48,49] \
		task_counts=[512,512,512,512,512,512,512,512,512,512] \
		num_envs=5120 \
		task=meta-world-v2 \
		reward_scale=100 \
		termination_on_success=False \
		experiment=0123_MT10_ppo_pcgrad \
		train=meta-world-PCGrad-PPO \
		seed=$i \
		wandb_activate=True \
		wandb_entity=meta-world \
		wandb_project=meta-world-ig \
		sim_device=cuda:$1 \
		rl_device=cuda:$1 \
		graphics_device_id=0 \
		record_videos=False \
		max_iterations=2000"
	echo $cmd
	$cmd
done