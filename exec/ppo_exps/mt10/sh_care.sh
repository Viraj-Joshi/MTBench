#!/bin/bash

for i in {42..51}
do
    python isaacgymenvs/train.py \
	task_id=[4,16,17,18,28,31,38,40,48,49] \
  	exempted_tasks=[] \
	task_counts=[512,512,512,512,512,512,512,512,512,512] \
	num_envs=5120 \
	experiment=shppo_care_mt10_fixed_seed_$i \
	task=meta-world-v2 \
	train=meta-world-mt-care-SHPPO \
	seed=$i \
	fixed=True \
	wandb_activate=True \
	wandb_project=IsaacGym \
    sim_device=cuda:3 \
    rl_device=cuda:3 \
	record_videos=False \
	reward_scale=100 \
	termination_on_success=False \
	max_iterations=1908 \
	headless=True
done