#!/bin/bash

for e in 4096 8192 16384 32768
do
	if [ $e -eq 4096 ]; then
		task_counts=[82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,82,81,81,81,81]
	elif [ $e -eq 8192 ]; then
		task_counts=[164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,163,163,163,163,163,163,163,163]
	elif [ $e -eq 16384 ]; then
		task_counts=[328,328,328,328,328,328,328,328,328,328,328,328,328,328,328,328,328,328,328,328,328,328,328,328,328,328,328,328,328,328,328,328,328,328,327,327,327,327,327,327,327,327,327,327,327,327,327,327,327,327]
	elif [ $e -eq 32768 ]; then
		task_counts=[656,656,656,656,656,656,656,656,656,656,656,656,656,656,656,656,656,656,655,655,655,655,655,655,655,655,655,655,655,655,655,655,655,655,655,655,655,655,655,655,655,655,655,655,655,655,655,655,655,655]
	fi
	t=$((500000000 / (e * 32)+1))
	for width in 2 3 4
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
				task_id=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49] \
				task_counts=$task_counts \
				num_envs=$e \
				task=meta-world-v2 \
				fixed=False \
				reward_scale=100 \
				termination_on_success=False \
				experiment=08_12_ppo_vanilla_mt50_rand_scaling_envs_${e}_width_${width}_seed_${i} \
				train=meta-world-mt50-vanilla-asymmetric-PPO \
				seed=$i \
				wandb_activate=True \
				wandb_project=IsaacGym \
				sim_device=cuda:1 \
				rl_device=cuda:1 \
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
done
