for i in 20
# 1/5 environments are flat ground to encourage moving forward
do
	cmd="python isaacgymenvs/train.py \
		num_envs=4096 \
		task=go1_no_penalty_easy \
		reward_scale=1 \
		experiment=0206_ppo_go1_single_easy_$i \
		task_id=[$i] \
		train=meta-world-go1-PPO-Vanilla \
		seed=43 \
		wandb_activate=True \
		wandb_entity=meta-world \
		wandb_project=meta-world-ig \
		sim_device=cuda:$1 \
		rl_device=cuda:$1 \
		graphics_device_id=0 \
		record_videos=True \
		max_iterations=4000"
	echo $cmd
	$cmd
done