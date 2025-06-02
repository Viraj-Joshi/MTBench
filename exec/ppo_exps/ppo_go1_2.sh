for i in 10 # 11 12 13 14 15 16 17 18 19
# 
do
	cmd="python isaacgymenvs/train.py \
		num_envs=4096 \
		task=go1_no_penalty \
		reward_scale=1 \
		experiment=0111_ppo_go1_no_penalty_mix_$i \
		task_id=[$i,20] \
		train=meta-world-go1-PPO-Vanilla \
		seed=43 \
		wandb_activate=True \
		wandb_entity=meta-world \
		wandb_project=meta-world-ig \
		sim_device=cuda:$1 \
		rl_device=cuda:$1 \
		graphics_device_id=0 \
		record_videos=True \
		max_iterations=10000"
	echo $cmd
	$cmd
done