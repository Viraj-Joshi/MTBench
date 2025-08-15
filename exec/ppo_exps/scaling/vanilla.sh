#!/bin/bash

e=24576
task_counts=[2458,2458,2458,2458,2458,2458,2457,2457,2457,2457]
t=$((1000000000 / (e * 32)+1))
for width in 1 2 3 4 5
do
	if [ $width -eq 1 ]; then
		units=[768,768,768] # ~1.2M params
	elif [ $width -eq 2 ]; then
		units=[1500,1500,1500] # ~4.6M params
	elif [ $width -eq 3 ]; then
		units=[3000,3000,3000] # ~18.2M params
	elif [ $width -eq 4 ]; then
		units=[6000,6000,6000] # ~72M params
	elif [ $width -eq 5 ]; then
		units=[12000,12000,12000] # ~288M params
	fi
	for i in 42 43 44 45 46
	do
		cmd="python isaacgymenvs/train.py \
			task_id=[4,16,17,18,28,31,38,40,48,49] \
			task_counts=$task_counts \
			num_envs=$e \
			task=meta-world-v2 \
			fixed=False \
			reward_scale=100 \
			termination_on_success=False \
			experiment=07_28_ppo_vanilla_TE_mt10_rand_scaling_width_${width}_seed_${i} \
			train=meta-world-mt10-vanilla-asymmetric-PPO \
			seed=$i \
			wandb_activate=True \
			wandb_project=IsaacGym \
			sim_device=cuda:0 \
			rl_device=cuda:0 \
			headless=True \
			record_videos=False \
			reward_scale=100 \
			termination_on_success=False \
			max_iterations=$t \
			units=$units \
			learn_task_embedding=True"
		echo $cmd
		$cmd
	done
done
