#!/bin/bash

for i in 42 43 44 45 46 47 48 49 50 51
do

	cmd="python isaacgymenvs/train.py \
		task_id=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49] \
		task_counts=[122,122,122,122,122,122,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123] \
		num_envs=6144 \
		task=meta-world-v2 \
		reward_scale=100 \
		termination_on_success=False \
		experiment=0123_MT50_ppo_pcgrad \
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