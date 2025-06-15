#!/bin/bash
# 128 gradient updates turns into 3 hours per run


for i in 42
do
	python isaacgymenvs/train.py \
	task_id=[4] \
	exempted_tasks=[] \
	task_counts=[4096] \
	num_envs=4096 \
	experiment=vanilla_fast_td3_mt10_rand_envs_${e}_seed_$i  \
	task=meta-world-v2 \
	train=meta-world-mt-fast-TD3 \
	fixed=False \
	seed=$i \
	record_videos=False \
	wandb_activate=False \
	wandb_project=IsaacGym \
	sim_device=cuda:1 \
	rl_device=cuda:1 \
	reward_scale=100 \
	termination_on_success=False \
	max_iterations=49000 \
	headless=True \
	nstep=16 \
	gradient_steps_per_itr=32
done