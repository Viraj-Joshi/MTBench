#!/bin/bash

for i in 42
do
    python isaacgymenvs/train.py \
	task_id=[4,16,17,18,28,31,38,40,48,49] \
  	exempted_tasks=[] \
	task_counts=[512,512,512,512,512,512,512,512,512,512] \
	num_envs=5120 \
	task=meta-world-discrete-v2 \
	train=meta-world-mt-soft-modularization-PQ \
	experiment=pqn_soft_modularization_mt10_rand_envs_${e}_seed_$i \
	seed=$i \
	fixed=False \
	wandb_activate=False \
	wandb_project=IsaacGym \
    sim_device=cuda:3 \
    rl_device=cuda:3 \
	reward_scale=100 \
	termination_on_success=False \
	max_iterations=50000 \
	headless=True
done