#!/bin/bash

task_id=$(python -c 'print(str(list(range(50))).replace(" ", ""))')
num_envs=24576
num_tasks=50
task_counts=$(python -c "
num_envs = ${num_envs}
num_tasks = ${num_tasks}
base_count = num_envs // num_tasks
remainder = num_envs % num_tasks
subgroup_counts = [base_count + 1] * remainder + [base_count] * (num_tasks - remainder)
task_counts = subgroup_counts
print(str(task_counts).replace(' ', ''))")
units=[512,512,512]

for i in 43
do
	cmd="python isaacgymenvs/train.py \
		task_id=${task_id} \
		task_counts=${task_counts} \
		num_envs=${num_envs} \
		task=meta-world-v2 \
		fixed=False \
		reward_scale=100 \
		termination_on_success=False \
		experiment=ppo_vanilla_mt50_rand_seed_${i} \
		train=meta-world-mt50-vanilla-asymmetric-PPO \
		seed=$i \
		wandb_activate=True \
		wandb_project=SAPG \
		sim_device=cuda:1 \
		rl_device=cuda:1 \
		headless=True \
		record_videos=False \
		reward_scale=100 \
		termination_on_success=False \
		max_iterations=7500 \
		units=$units \
		learn_task_embedding=False"
	echo $cmd
	$cmd
done