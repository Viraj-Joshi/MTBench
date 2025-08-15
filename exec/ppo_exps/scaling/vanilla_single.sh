#!/bin/bash

e=4096
t=$((50000000 / (e * 32)+1))
for task in 16 17 18 28 38 40 48 49
do
	for width in 4 3 2 1
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
		for i in 42 43 44
		do
			cmd="python isaacgymenvs/train.py \
				task_id=[$task] \
				task_counts=[4096] \
				num_envs=$e \
				task=meta-world-v2 \
				fixed=False \
				reward_scale=100 \
				termination_on_success=False \
				experiment=08_13_ppo_vanilla_single_task_${task}_rand_scaling_width_${width}_seed_${i} \
				train=meta-world-mt50-vanilla-asymmetric-PPO \
				seed=$i \
				wandb_activate=True \
				wandb_project=IsaacGym \
				sim_device=cuda:2 \
				rl_device=cuda:2 \
				headless=True \
				record_videos=False \
				reward_scale=100 \
				termination_on_success=False \
				max_iterations=$t \
				units=$units \
				learn_task_embedding=False"
			echo $cmd
			$cmd
		done
	done
done
