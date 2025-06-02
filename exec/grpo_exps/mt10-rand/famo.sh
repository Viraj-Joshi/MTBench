#!/bin/bash

for e in 4096
do
	task_counts=[410,410,410,410,410,410,409,409,409,409]
	# task_counts=[2458,2458,2458,2458,2458,2458,2457,2457,2457,2457]
	t=$(( (1000000000 + (e * 150) - 1) / (e * 150) ))
	for i in 42
	do
		python isaacgymenvs/train.py \
		task_id=[4,16,17,18,28,31,38,40,48,49] \
		task_counts=$task_counts \
		experiment=grpo_famo_mt10_rand_${e}_seed_$i \
		num_envs=$e \
		task=meta-world-v2 \
		train=meta-world-mt10-famo-GRPO \
		fixed=False \
		seed=$i \
		wandb_activate=True \
		wandb_project=IsaacGym \
		sim_device=cuda:2 \
		rl_device=cuda:2 \
		reward_scale=100 \
		termination_on_success=False \
		max_iterations=$t \
        record_videos=False \
		headless=True
	done
done