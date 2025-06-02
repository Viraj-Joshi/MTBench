#!/bin/bash

for e in 24576
do
	task_counts=[2458,2458,2458,2458,2458,2458,2457,2457,2457,2457]
	t=$(( (1000000000 + (e * 32) - 1) / (e * 32) ))
	for i in 48 49 50 51
	do
		cmd="python isaacgymenvs/train.py \
			task_id=[4,16,17,18,28,31,38,40,48,49] \
			task_counts=$task_counts \
			num_envs=$e \
			task=meta-world-v2 \
			fixed=False \
			reward_scale=100 \
			termination_on_success=False \
			experiment=05_11_mhppo_care_mt10_rand_envs_${e}_seed_$i \
			train=meta-world-mt10-care-MHPPO \
			seed=$i \
			wandb_activate=True \
			wandb_project=IsaacGym \
			headless=True \
			sim_device=cuda:0 \
			rl_device=cuda:0 \
			record_videos=False \
			max_iterations=$t"
		echo $cmd
		$cmd
	done
done