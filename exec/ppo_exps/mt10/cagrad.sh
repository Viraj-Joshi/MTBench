#!/bin/bash

for i in 42 43 44 45 46 47 48 49 50 51
do
	cmd="python isaacgymenvs/train.py \
<<<<<<< HEAD
		task_id=[40,38,31,15,17,18,4,28,48,49] \
=======
		task_id=[6,16,17,18,28,31,38,40,48,49] \
		exempted_tasks=[] \
>>>>>>> locomotion_devel
		task_counts=[512,512,512,512,512,512,512,512,512,512] \
		num_envs=5120 \
		task=meta-world-v2 \
		reward_scale=100 \
		termination_on_success=False \
<<<<<<< HEAD
		experiment=0123_MT10_ppo_cagrad \
=======
		experiment=0104_mt10_ppo_CAGrad \
>>>>>>> locomotion_devel
		train=meta-world-CAGrad-PPO \
		seed=$i \
		wandb_activate=False \
		wandb_entity=meta-world \
		wandb_project=meta-world-ig \
		sim_device=cuda:$1 \
		rl_device=cuda:$1 \
		graphics_device_id=0 \
		record_videos=False \
		max_iterations=2000"
    echo $cmd
    $cmd
done