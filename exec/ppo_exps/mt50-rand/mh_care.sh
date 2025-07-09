#!/bin/bash

task_counts=[492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491]
e=24576
t=$((1000000000 / (e * 32)+1))
for i in 51
do
    python isaacgymenvs/train.py \
	task_id=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49] \
	task_counts=$task_counts \
	num_envs=$e \
	experiment=mhppo_care_mt50_rand_envs_${e}_seed_${i} \
	task=meta-world-v2 \
	train=meta-world-mt50-care-MHPPO \
	seed=$i \
	fixed=False \
	wandb_activate=True \
	wandb_project=IsaacGym \
    sim_device=cuda:0 \
    rl_device=cuda:0 \
	record_videos=False \
	reward_scale=100 \
	termination_on_success=False \
	max_iterations=$t \
	headless=True
done
