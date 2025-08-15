#!/bin/bash

task_counts=[164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,163,163,163,163,163,163,163,163]
for gradient_steps_per_itr in 2
do
	for nstep in 8
	do
		for i in 42
		do
			python isaacgymenvs/train.py \
			task_id=[4,16,17,18,28,31,38,40,48,49] \
			task_counts=$task_counts \
			num_envs=8192 \
			experiment=vanilla_fast_td3_mt50_seed_${i} \
			task=meta-world-v2 \
			train=meta-world-mt-fast-TD3 \
			fixed=False \
			seed=$i \
			record_videos=False \
			wandb_activate=False \
			wandb_project=IsaacGym \
			sim_device=cuda:0 \
			rl_device=cuda:0 \
			reward_scale=1 \
			termination_on_success=False \
			max_iterations=24000 \
			headless=True \
			gradient_steps_per_itr=$gradient_steps_per_itr \
			nstep=$nstep
		done
	done
done