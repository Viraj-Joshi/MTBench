#!/bin/bash
# 128 gradient updates turns into 3 hours per run

e=4096
for i in {43..46}
do
	python isaacgymenvs/train.py \
	task_id=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49] \
	task_counts=[82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,81,81,81,81] \
	num_envs=$e \
	experiment=sac_vanilla_mt50_rand_envs_${e}_seed_${i} \
	task=meta-world-v2 \
	train=meta-world-mt-SAC \
	fixed=False \
	seed=$i \
	record_videos=False \
	wandb_activate=False \
	wandb_project=IsaacGym \
	sim_device=cuda:1 \
	rl_device=cuda:1 \
	reward_scale=100 \
	termination_on_success=False \
	max_iterations=61036 \
	headless=True \
	nstep=16 \
	gradient_steps_per_itr=32
done