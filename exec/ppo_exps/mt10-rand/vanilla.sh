#!/bin/bash
task_counts=[410,410,410,410,410,410,409,409,409,409]
for e in 4096
do
	t=$(( (1000000000 + (e * 32) - 1) / (e * 32) ))
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
			experiment=05_28_ppo_vanilla_mt10_rand_envs_${e}_seed_${i} \
			train=meta-world-mt10-vanilla-PPO\
			seed=$i \
			wandb_activate=True \
			wandb_project=IsaacGym \
			sim_device=cuda:0 \
			rl_device=cuda:0 \
			record_videos=False \
			max_iterations=$t"
		echo $cmd
		$cmd
	done
done