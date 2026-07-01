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

for i in 46
do
	cmd="python isaacgymenvs/train.py \
		task_id=${task_id} \
		task_counts=${task_counts} \
		num_envs=${num_envs} \
		task=meta-world-sapg-v2 \
		fixed=False \
		reward_scale=100 \
		termination_on_success=False \
		experiment=sapg_mt50_rand_expl_0p012_512x3_seed_${i}\
		train=meta-world-mt50-vanilla-SAPG-PPO \
		seed=$i \
		headless=True \
		wandb_activate=True \
		wandb_project=SAPG \
		sim_device=cuda:0 \
		rl_device=cuda:0 \
		record_videos=False \
		max_iterations=7500"
	echo $cmd
	$cmd
done