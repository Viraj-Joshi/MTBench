for i in 41 42 43 44 45
 
do
	cmd="python isaacgymenvs/train.py \
		num_envs=4096 \
		task=final_go1_hard \
		reward_scale=1 \
		experiment=0220_ppo_go1_hard_famo_seed_$i \
		task_id=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] \
		train=go1-PPO-clip-FAMO \
		seed=$i \
		wandb_activate=True \
		wandb_entity=meta-world \
		wandb_project=meta-world-ig \
		sim_device=cuda:$1 \
		rl_device=cuda:$1 \
		graphics_device_id=0 \
		record_videos=False \
		checkpoint=runs/pretrained/go1_flat_vanilla.pth \
		max_iterations=4000"
	echo $cmd
	$cmd
done