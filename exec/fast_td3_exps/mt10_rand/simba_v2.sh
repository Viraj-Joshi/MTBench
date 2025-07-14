#!/bin/bash

# 8 16 has 3rd highest success rate
# 32 16 has highest success rate
task_counts=[410,410,410,410,410,410,409,409,409,409]
for gradient_steps_per_itr in 2
do
	for nstep in 8
	do
		for i in 42
		do
			python isaacgymenvs/train.py \
			task_id=[4,16,17,18,28,31,38,40,48,49] \
			task_counts=$task_counts \
			num_envs=4096 \
			experiment=fast_td3_simba_v2_mt10_rand_seed_${i} \
			task=meta-world-v2 \
			train=meta-world-mt-fast-TD3-simba-v2 \
			fixed=False \
			seed=$i \
			record_videos=False \
			wandb_activate=False \
			wandb_project=IsaacGym \
			sim_device=cuda:1 \
			rl_device=cuda:1 \
			reward_scale=1 \
			termination_on_success=False \
			max_iterations=12000 \
			headless=True \
			gradient_steps_per_itr=$gradient_steps_per_itr \
			nstep=$nstep
		done
	done
done