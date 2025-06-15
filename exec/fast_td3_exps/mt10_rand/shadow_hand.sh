#!/bin/bash

for i in 42
do
	python isaacgymenvs/train.py \
	task=ShadowHand \
	num_envs=4096 \
	experiment=vanilla_fast_td3_shadowhand_rand_envs_${e}_seed_$i  \
	train=ShadowHandTD3 \
	record_videos=False \
	wandb_activate=False \
	wandb_project=IsaacGym \
	sim_device=cuda:1 \
	rl_device=cuda:1 \
	max_iterations=49000 \
	headless=True \
	nstep=16 \
	gradient_steps_per_itr=32
done