#!/bin/bash

e=6144
t=$((1000000000 / (e * 32)+1))
for i in {42..51}
do
	cmd="python isaacgymenvs/train.py \
		task_id=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49] \
		task_counts=[122,122,122,122,122,122,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123,123] \
		num_envs=$e \
		task=meta-world-v2 \
		fixed=False \
		reward_scale=100 \
		termination_on_success=False \
		experiment=ppo_cagrad_mt50_rand_envs_${e}_seed_${i} \
		train=meta-world-CAGrad-PPO \
		seed=$i \
		wandb_activate=True \
		wandb_project=IsaacGym \
		sim_device=cuda:0 \
		rl_device=cuda:0 \
		graphics_device_id=0 \
		record_videos=False \
		headless=True \
		max_iterations=$t"
    echo $cmd
    $cmd
done
