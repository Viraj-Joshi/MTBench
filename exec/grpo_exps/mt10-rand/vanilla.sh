#!/bin/bash
task_counts=[410,410,410,410,410,410,409,409,409,409]
for e in 4096
do
	t=$(( (1000000000 + (e * 150) - 1) / (e * 150) ))
	for i in 45 46 47 48 49 50 51
	do
		python isaacgymenvs/train.py \
		task_id=[4,16,17,18,28,31,38,40,48,49] \
		task_counts=$task_counts \
		experiment=05_26_grpo_vanilla_mt10_rand_envs_${e}_seed_$i \
		num_envs=$e \
		task=meta-world-v2 \
		train=meta-world-mt10-vanilla-GRPO \
		fixed=False \
		seed=$i \
		record_videos=False \
		wandb_activate=True \
		wandb_project=IsaacGym \
		sim_device=cuda:0 \
		rl_device=cuda:0 \
		reward_scale=100 \
		termination_on_success=False \
		max_iterations=$t \
		headless=True
	done
done