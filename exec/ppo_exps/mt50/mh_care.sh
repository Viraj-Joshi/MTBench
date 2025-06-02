#!/bin/bash

for i in {42..51}
do
    python isaacgymenvs/train.py \
	task_id=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49] \
  	exempted_tasks=[] \
	task_counts=[128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128] \
	num_envs=6400 \
	experiment=mhppo_care_mt50_fixed_seed_$i \
	task=meta-world-v2 \
	train=meta-world-mt50-care-MHPPO \
	seed=$i \
	fixed=True \
	wandb_activate=True \
	wandb_project=IsaacGym \
    sim_device=cuda:2 \
    rl_device=cuda:2 \
	record_videos=False \
	reward_scale=100 \
	termination_on_success=False \
	max_iterations=1221 \
	headless=True
done