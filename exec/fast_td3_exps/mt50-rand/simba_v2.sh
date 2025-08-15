#!/bin/bash

task_counts=[164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,163,163,163,163,163,163,163,163]
for gradient_steps_per_itr in 2
do
	for nstep in 8
	do
		for i in 42 43 44 45 46 47 48 49 50 51
		do
			python isaacgymenvs/train.py \
			task_id=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49] \
			task_counts=$task_counts \
			num_envs=8192 \
			experiment=fast_td3_simba_v2_mt50_rand_seed_${i} \
			task=meta-world-v2 \
			train=meta-world-mt50-fast-TD3-simba-v2 \
			fixed=False \
			seed=$i \
			record_videos=False \
			wandb_activate=False \
			wandb_project=IsaacGym \
			sim_device=cuda:0 \
			rl_device=cuda:0 \
			reward_scale=1 \
			termination_on_success=False \
			max_iterations=61037 \
			headless=True \
			gradient_steps_per_itr=$gradient_steps_per_itr \
			nstep=$nstep
		done
	done
done