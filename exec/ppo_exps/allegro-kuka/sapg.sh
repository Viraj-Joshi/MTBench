#!/bin/bash
# Multi-task Allegro-Kuka (regrasping / throw / reorientation) via MTBench rl_games PPO
#
# task ids:  0 -> regrasping, 1 -> throw, 2 -> reorientation
#
# Uses SAPG with a shared recurrent (LSTM) trunk (allegro-kuka-SAPG-lstm-PPO):
# shared asymmetric_a2c trunk (obs -> LSTM -> MLP) with per-policy embeddings. SAPG splits
# the envs into `block_size`-sized policy blocks: num_policies = num_envs // block_size
# (24576 / 4096 = 6 policies here). 

# For the feedforward SAPG baseline instead, set train=meta-world-mt50-vanilla-SAPG-PPO and
# add train.params.config.horizon_length=16.
task_ids="[0,1,2]"
num_tasks=3
horizon=16   # overridden below via train.params.config.horizon_length

# keep num_envs a multiple of block_size (4096) so the policy split is clean
for e in 24576
do
	# split e envs across tasks; first `rem` tasks get one extra (sums to e)
	base=$(( e / num_tasks ))
	rem=$(( e % num_tasks ))
	counts=()
	for ((k=0; k<num_tasks; k++)); do
		if [ $k -lt $rem ]; then counts+=($((base + 1))); else counts+=($base); fi
	done
	task_counts="[$(IFS=,; echo "${counts[*]}")]"

	# ~2e10 env steps total
	t=$(( (20000000000 + (e * horizon) - 1) / (e * horizon) ))

	for i in 42
	do
		cmd="python isaacgymenvs/train.py \
			task=AllegroKukaMT \
			task_id=$task_ids \
			task_counts=$task_counts \
			num_envs=$e \
			train=allegro-kuka-SAPG-lstm-PPO \
			train.params.config.learning_rate=5e-4 \
			train.params.config.block_size=4096 \
			experiment=ppo_sapg_allegro-kuka_mt_shared_lstm_lr_3e-4_mb_16384_me_2_expl_coef_0p005_${e}_seed_${i} \
			seed=$i \
			wandb_activate=True \
			wandb_project=reppo \
			sim_device=cuda:1 \
			rl_device=cuda:1 \
			headless=True \
			capture_video=False \
			max_iterations=$t"
		echo $cmd
		$cmd
	done
done
