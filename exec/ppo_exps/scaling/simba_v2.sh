#!/bin/bash

e=24576
task_counts=[2458,2458,2458,2458,2458,2458,2457,2457,2457,2457]
# task_counts=[492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,492,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491,491]
t=$((250000000 / (e * 32)+1))
for width in 1 2 3 4 5
do
	if [ $width -eq 1 ]; then
		critic_hidden_dim=256  # ~1.1M params
	elif [ $width -eq 2 ]; then
		critic_hidden_dim=512 # ~4.5M params
	elif [ $width -eq 3 ]; then
		critic_hidden_dim=1024 # ~17.9M params
	elif [ $width -eq 4 ]; then
		critic_hidden_dim=2048 # ~71M params
	elif [ $width -eq 5 ]; then
		critic_hidden_dim=4096 # ~285M params
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
			experiment=ppo_simbav2_TE_mt10_rand_scaling_width_${width}_seed_${i} \
			train=meta-world-mt10-simba-v2-PPO \
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
			critic_hidden_dim=$critic_hidden_dim \
			learn_task_embedding=True"
		echo $cmd
		$cmd
	done
done
