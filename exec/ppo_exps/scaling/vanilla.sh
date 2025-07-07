#!/bin/bash

e=24576
task_counts=[2458,2458,2458,2458,2458,2458,2457,2457,2457,2457]
# task_counts=[492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491]
t=$((1000000000 / (e * 32)+1))
for width in 3
do
	if [ $width -eq 1 ]; then
		units=[256,256,256] # ~400k params
	elif [ $width -eq 2 ]; then
		units=[768,768,768] # ~1M params
	elif [ $width -eq 3 ]; then
		units=[1024,1024,1024] # ~4M params
	elif [ $width -eq 4 ]; then
		units=[2048,2048,2048]
	elif [ $width -eq 5 ]; then
		units=[4096,4096,4096]
	fi
	for i in 42 43 44
	do
		cmd="python isaacgymenvs/train.py \
			task_id=[4,16,17,18,28,31,38,40,48,49] \
			task_counts=$task_counts \
			num_envs=$e \
			task=meta-world-v2 \
			fixed=False \
			reward_scale=100 \
			termination_on_success=False \
			experiment=07_06_ppo_vanilla_mt10_rand_scaling_width_${width}_seed_${i} \
			train=meta-world-mt10-vanilla-asymmetric-PPO \
			seed=$i \
			wandb_activate=False \
			wandb_project=IsaacGym \
			sim_device=cuda:0 \
			rl_device=cuda:0 \
			headless=True \
			record_videos=False \
			reward_scale=100 \
			termination_on_success=False \
			max_iterations=$t \
			units=$units"
		echo $cmd
		$cmd
	done
done