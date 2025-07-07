#!/bin/bash

# 8 16 has 3rd highest success rate
# 32 16 has highest success rate
task_counts=[410,410,410,410,410,410,409,409,409,409]
for gradient_steps_per_itr in 4
do
	for nstep in 8
	do
		for i in 42
		do
			python isaacgymenvs/train.py \
			task_id=[4,16,17,18,28,31,38,40,48,49] \
			task_counts=$task_counts \
			num_envs=4096 \
			experiment=vanilla_fast_td3_mt10_g_${gradient_steps_per_itr}_n_${nstep}  \
			task=meta-world-v2 \
			train=meta-world-mt-fast-TD3 \
			fixed=False \
			seed=$i \
			record_videos=False \
			wandb_activate=False \
			wandb_project=IsaacGym \
			sim_device=cuda:2 \
			rl_device=cuda:2 \
			reward_scale=100 \
			termination_on_success=False \
			max_iterations=24415 \
			headless=True \
			gradient_steps_per_itr=$gradient_steps_per_itr \
			nstep=$nstep
		done
	done
done