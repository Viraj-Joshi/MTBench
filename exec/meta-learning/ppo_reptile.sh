#!/bin/bash

for i in 42
do
    python isaacgymenvs/train.py \
	task_id=[31] \
  	task_counts=[8192]\
	task=meta-world-ml-one \
	train=meta-world-ML-ReptilePPO \
	seed=$i \
	wandb_activate=True \
	wandb_project=IsaacGym \
    sim_device=cuda:0 \
    rl_device=cuda:0 \
    graphics_device_id=0
done