#!/bin/bash
# mt 10 is 4,16,17,18,28,31,38,40,48,49

for task_id in 4 16 17 18 28 31 38 40 48 49
do
	for i in 42
	do
		python isaacgymenvs/train.py \
		task_id=[$task_id] \
		exempted_tasks=[] \
		task_counts=[4096] \
		num_envs=4096 \
		task=meta-world-v2 \
		train=meta-world-mt-soft-modularization-SAC \
		fixed=True \
		seed=$i \
		record_videos=False \
		wandb_activate=False \
		wandb_project=IsaacGym \
		sim_device=cuda:2 \
		rl_device=cuda:2 \
		reward_scale=100 \
		termination_on_success=False \
		max_iterations=4000 \
		headless=True
	done
done