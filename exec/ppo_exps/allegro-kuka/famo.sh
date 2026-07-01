#!/bin/bash
# Multi-task Allegro-Kuka (regrasping / throw / reorientation) via MTBench rl_games PPO
#
# task ids:  0 -> regrasping, 1 -> throw, 2 -> reorientation
task_ids="[0,1,2]"
num_tasks=3

for e in 24576
do

	base=$(( e / num_tasks ))
	rem=$(( e % num_tasks ))
	counts=()
	for ((k=0; k<num_tasks; k++)); do
		if [ $k -lt $rem ]; then counts+=($((base + 1))); else counts+=($base); fi
	done
	task_counts="[$(IFS=,; echo "${counts[*]}")]"

	# ~2e10 env steps total
	t=$(( (20000000000 + (e * 16) - 1) / (e * 16) ))

	for i in 42
	do
		cmd="python isaacgymenvs/train.py \
			task=AllegroKukaMT \
			task_id=$task_ids \
			task_counts=$task_counts \
			num_envs=$e \
			train=allegro-kuka-FAMO-lstm-PPO \
			train.params.network.critic.units=[768,512,256] \
			train.params.config.learning_rate=5e-4 \
			train.params.network.separate=False \
			experiment=ppo_famo_allegro-kuka_mt_shared_lstm_adaptive-lr-5e-4_${e}_seed_${i} \
			seed=$i \
			wandb_activate=True \
			wandb_project=reppo \
			sim_device=cuda:3 \
			rl_device=cuda:3 \
			headless=True \
			capture_video=False \
			max_iterations=$t"
		echo $cmd
		$cmd
	done
done
