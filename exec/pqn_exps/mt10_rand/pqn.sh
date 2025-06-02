#!/bin/bash

e=8192
task_counts=[820,820,820,820,820,820,818,818,818,818]
for i in 42 43 44 45 46 47 48 49 50 51
do
    python isaacgymenvs/train.py \
	task_id=[4,16,17,18,28,31,38,40,48,49] \
	task_counts= $task_counts \
	num_envs=$e \
	experiment=vanilla_pqn_mt10_rand_envs_${e}_seed_$i \
	task=meta-world-discrete-v2 \
	train=meta-world-mt-PQ \
	seed=$i \
	fixed=False \
	wandb_activate=False \
	wandb_project=IsaacGym \
	record_videos=False \
    sim_device=cuda:0 \
    rl_device=cuda:0 \
	reward_scale=100 \
	termination_on_success=False \
	max_iterations=50000 \
	headless=True
done