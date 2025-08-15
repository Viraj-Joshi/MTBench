#!/bin/bash
# 128 gradient updates turns into 3 hours per run

e=4096
for i in 48 49 50 51
do
	python isaacgymenvs/train.py \
	task_id=[4,16,17,18,28,31,38,40,48,49] \
	task_counts=[410,410,410,410,410,410,409,409,409,409] \
	num_envs=$e \
	experiment=sac_vanilla_mt10_rand_envs_${e}_seed_${i} \
	task=meta-world-v2 \
	train=meta-world-mt-SAC \
	fixed=False \
	seed=$i \
	record_videos=False \
	wandb_activate=False \
	wandb_project=IsaacGym \
	sim_device=cuda:0 \
	rl_device=cuda:0 \
	reward_scale=100 \
	termination_on_success=False \
	max_iterations=48830 \
	headless=True \
	gradient_steps_per_itr=32
done