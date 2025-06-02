# This is only used to evaluate single task
cmd="python isaacgymenvs/train.py \
	num_envs=21 \
	task=final_go1_test \
	test=True \
	reward_scale=1 \
	checkpoint=$3 \
	task_id=[$2] \
	train=go1-PPO-Vanilla \
	seed=43 \
	wandb_activate=False \
	wandb_entity=meta-world \
	wandb_project=meta-world-ig \
	sim_device=cuda:$1 \
	rl_device=cuda:$1 \
	graphics_device_id=0 \
	record_videos=True \
	max_iterations=10000"
echo $cmd
$cmd