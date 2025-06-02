# MTBench: Benchmarking Massively Parallelized Multi-Task Reinforcement Learning for Robotics Tasks

[Paper](https://openreview.net/forum?id=z0MM0y20I2) (RLC 2025)

Multi-task Reinforcement Learning (MTRL) has emerged as a critical training paradigm for applying reinforcement learning (RL) to a set of complex real-world robotic tasks, which demands a generalizable and robust policy. At the same time, massively parallelized training has gained popularity, not only for significantly accelerating data collection through GPU-accelerated simulation but also for enabling diverse data collection across multiple tasks by simulating heterogeneous scenes in parallel. However, existing MTRL research has largely been limited to off-policy methods like SAC in the low-parallelization regime. MTRL could capitalize on the higher asymptotic performance of on-policy algorithms, whose batches require data from the current policy, and as a result, take advantage of massive parallelization offered by GPU-accelerated simulation. To bridge this gap, we introduce a massively parallelized Multi-Task Benchmark for robotics (MTBench), an open-sourced benchmark featuring a broad distribution of 50 manipulation tasks and 20 locomotion tasks, implemented using the GPU-accelerated simulator IsaacGym. MTBench also includes four base RL algorithms combined with seven state-of-the-art MTRL algorithms and architectures, providing a unified framework for evaluating their performance. Our extensive experiments highlight the superior speed of evaluating MTRL approaches using MTBench, while also uncovering unique challenges that arise from combining massive parallelism with MTRL. 

## Installation ⚙️

Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions in the documentation. 

Ensure that Isaac Gym works on your system by running one of the examples from the `python/examples`
directory, like `joint_monkey.py`. Follow troubleshooting steps described in the Isaac Gym Preview 4
install instructions if you have any trouble running the samples.

Once Isaac Gym is installed and samples work within your current python environment, install this repo:

```bash
pip install -e .
pip install skrl
pip install moviepy
```


## Basic Structure
```
MTBench/
├── assets
│   ├── assets_v2/
│       ├── unified_objects/            # Meta-World assets
|   ├──urdf
│       ├── franka_description/         # franka robot assets
|       ├── go1/                        # go1 assets 
├── exec/                               # Bash scripts to run experiments
├── isaacgymenvs/
│   ├── cfg/                            # Hydra configs for tasks and training    
│   ├── learning/                       # RLGames MTRL Algorithms and approaches
│       ├── networks/                   # MTRL architectures
│   ├── tasks/                
│       ├── franka/                     
│           ├── vec_task/               
│               ├── franka_base.py          # Meta-World base class
│               ├── task_fns/               # Meta-World task functions
│       ├── locomotion/                 
│           ├── legged_base.py          # Parkour base class
│           ├── set_terrains/           # Parkour curriculum
|   ├── train.py                        # Entry point for training and evaluation
├── scripts/                            # Scripts for visualization
``` 

## Environments 🌍

| Environment | Reference | Location |
|-------------|-------------|-------------|
| Meta-World  | https://arxiv.org/abs/1910.10897 | [franka_base.py](isaacgymenvs/tasks/franka/vec_task/franka_base.py) |
| Parkour | https://arxiv.org/abs/2411.01775 |  [legged_base.py](isaacgymenvs/tasks/locomotion/legged_base.py)


In this repo, we provide two task sets, Meta-World and Parkour, which have base classes `franka_base.py` and `legged_base.py` respectively. All tasks are defined on top of these base classes and are located in the `isaacgymenvs/tasks/franka` and  `isaacgymenvs/tasks/locomotion`folders. 

To add a new task to Meta-World, add it to the task id to task map in `isaacgymenvs/tasks/franka/vec_task/franka_base.py` and implement the task with the required functions (`create_envs`, `compute_observations`, `compute_reward`, and `reset_env`) in a new file in the `task_fns` folder. 

### Basic Usage
```python
import isaacgym
import isaacgymenvs
import torch

num_envs = 4096

envs = isaacgymenvs.make(
    seed=0, 
    task="meta-world-v2", 
    num_envs=num_envs, 
    sim_device="cuda:0",
    rl_device="cuda:0",   
    headless=True,
)
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()
for _ in range(20):
 random_actions = 2.0 * torch.rand((num_envs,) + envs.action_space.shape, device = 'cuda:0') - 1.0
 envs.step(random_actions)
```


## MTRL Approaches 🐙

### MT-PPO
| Approach | Reference | Location |
|-------------|-------------|-------------|
| Vanilla  | -      | [MTA2CAgent](isaacgymenvs/learning/mt_a2c_agent.py#L58)
| Multihead Vanilla  | -      | [MultiHeadA2CBuilder](isaacgymenvs/learning/networks/multihead_a2c_builder.py#L8)
| PCGrad  | https://arxiv.org/abs/2001.06782 | [PCGradA2CAgent](isaacgymenvs/learning/mt_a2c_agent.py#L542)
| CAGrad  | https://arxiv.org/abs/2110.14048       | [CAGradA2CAgent](isaacgymenvs/learning/mt_a2c_agent.py#L573)
| FAMO    | https://arxiv.org/abs/2306.03792       | [FAMOA2CAgent](isaacgymenvs/learning/mt_a2c_agent.py#L632)
| Soft-Modularization  | https://arxiv.org/abs/2003.13661 | [SoftModularizedA2CBuilder](isaacgymenvs/learning/networks/soft_modularization_a2c_builder.py)
| CARE  | https://arxiv.org/abs/2102.06177 | [CAREA2CBuilder](isaacgymenvs/learning/networks/care_a2c_builder.py)
| PaCo  | https://arxiv.org/abs/2210.11653  | [PACOA2CBuilder](isaacgymenvs/learning/networks/paco_a2c_builder.py)
| MOORE  | https://arxiv.org/abs/2311.11385 | [MOOREA2CBuilder](isaacgymenvs/learning/networks/moore_a2c_builder.py)

### MT-SAC
| Approach | Reference | Location |
|-------------|-------------|-------------|
| Vanilla  | -      | [MTSACAgent](isaacgymenvs/learning/mt_sac_agent.py#L22)
| Soft-Modularization | https://arxiv.org/abs/2003.13661 | [SoftModularizedSACBuilder](isaacgymenvs/learning/networks/soft_modularization_sac_builder.py)

### MT-PQN
| Approach | Reference | Location |
|-------------|-------------|-------------|
| PQN | https://arxiv.org/abs/2407.04811 | [Algorithm](isaacgymenvs/learning/pqn_agent.py#L613); [Architecture](isaacgymenvs/learning/networks/pq_builder.py)
| PQN + Soft-Modularization | - | [SoftModularizedPQBuilder](isaacgymenvs/learning/networks/soft_modularized_pq_builder.py)

### MT-GRPO
| Approach | Reference | Location |
|-------------|-------------|-------------|
| GRPO | https://arxiv.org/abs/2402.03300 | [MTGRPOAgent](isaacgymenvs/learning/grpo_agent.py#L46)
| GRPO + FAMO| - | [FAMOGRPOAgent](isaacgymenvs/learning/grpo_agent.py#L431)

## Extending the Benchmark 📝
By simply adding agents in the learning folder, you can easily extend the benchmark with new MTRL approaches.
By simply adding networks in the learning/networks folder, you can easily extend the benchmark with new MTRL architectures.

## Training 📈

All experimental results are reproducible from the bash executables in the ```exec``` folder. The bash scripts are organized by environment and MTRL approach, so you can easily find the one you need. They all call ```train.py```, which means the key arguments are below. To reproduce any PPO MTRL approach ```x``` in evaluation setting ```y```, simply run the following command:

``` exec/ppo_exps/y/x.sh ```

### Loading trained models // Checkpoints

Checkpoints are saved in the folder `runs/EXPERIMENT_NAME/nn` where `EXPERIMENT_NAME`
defaults to the task name, but can also be overridden via the `experiment` argument.

To load a trained checkpoint and continue training, use the `checkpoint` argument:

```bash
python train.py task=meta-world-v2 checkpoint=runs/x/nn/x.pth
```

To load a trained checkpoint and only perform inference (no training), pass `test=True`
as an argument, along with the checkpoint name. To avoid rendering overhead, you may
also want to run with fewer environments using `num_envs=64`:

```bash
python train.py task=meta-world-v2 checkpoint=runs/x/nn/x.pth test=True num_envs=64
```

### Configuration and command line arguments

We use [Hydra](https://hydra.cc/docs/intro/) to manage the config. Note that this has some
differences from previous incarnations in older versions of Isaac Gym.

Key arguments to the `train.py` script are:

* `task=TASK` - selects which task to use. Any of `meta-world-v2`, `go1-benchmark` (these correspond to the config for each environment in the folder `isaacgymenvs/cfg/task`)
* `train=TRAIN` - selects which training config to use.
* `task_id` - selects which task IDs to use for training. This is a comma-separated list of integers, e.g. `task_id=[5,8,1]` will select the first three tasks in task map in the base .
* `task_counts` - an array of integers that specifies how many environments to use for each task. For example, `task_counts=[512, 512, 512, ... ]` will use 512 environments for each of the first three tasks. The number of task counts must match the number of tasks specified in `task_id`. 
* `num_envs=NUM_ENVS` - the number of total parallel environments to use and must match the sum of the task counts specified in `task_counts`. For example, if you have 10 tasks with 512 environments each, you would set `num_envs=5120`.
* `seed=SEED` - sets a seed value for randomizations, and overrides the default seed set up in the task config
* `sim_device=SIM_DEVICE_TYPE` - Device used for physics simulation. Set to `cuda:0` (default) to use GPU and to `cpu` for CPU. Follows PyTorch-like device syntax.
* `rl_device=RL_DEVICE` - Which device / ID to use for the RL algorithm. Defaults to `cuda:0`, and also follows PyTorch-like device syntax.
* `graphics_device_id=GRAPHICS_DEVICE_ID` - Which Vulkan graphics device ID to use for rendering. Defaults to 0. **Note** - this may be different from CUDA device ID, and does **not** follow PyTorch-like device syntax.
* `pipeline=PIPELINE` - Which API pipeline to use. Defaults to `gpu`, can also set to `cpu`. When using the `gpu` pipeline, all data stays on the GPU and everything runs as fast as possible. When using the `cpu` pipeline, simulation can run on either CPU or GPU, depending on the `sim_device` setting, but a copy of the data is always made on the CPU at every step.
* `test=TEST`- If set to `True`, only runs inference on the policy and does not do any training.
* `checkpoint=CHECKPOINT_PATH` - Set to path to the checkpoint to load for training or testing.
* `headless=HEADLESS` - Whether to run in headless mode.
* `experiment=EXPERIMENT` - Sets the name of the experiment.
* `max_iterations=MAX_ITERATIONS` - Sets how many iterations to run for.
* `exempted_tasls=[]` - an optional list of task ids that does not randomize the start step count per env. We find that it improves performance, so we always randomize the starting step count per env.

Hydra also allows setting variables inside config files directly as command line arguments. As an example, to set the discount rate for a rl_games training run, you can use `train.params.config.gamma=0.999`. Similarly, variables in task configs can also be set. For example, `task.env.enableDebugVis=True`.

#### Hydra Notes

Default values for each of these are found in the `isaacgymenvs/config/config.yaml` file.

The way that the `task` and `train` portions of the config works are through the use of config groups.
You can learn more about how these work [here](https://hydra.cc/docs/tutorials/structured_config/config_groups/)
The actual configs for `task` are in `isaacgymenvs/config/task/<TASK>.yaml` and for train in `isaacgymenvs/config/train/<TASK>PPO.yaml`.

In some places in the config you will find other variables referenced (for example,
 `num_actors: ${....task.env.numEnvs}`). Each `.` represents going one level up in the config hierarchy.
 This is documented fully [here](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation).

## WandB support

You can run [WandB](https://wandb.ai/) with Isaac Gym Envs by setting `wandb_activate=True` flag from the command line. You can set the group, name, entity, and project for the run by setting the `wandb_group`, `wandb_name`, `wandb_entity` and `wandb_project` set. Make sure you have WandB installed with `pip install wandb` before activating.

## Capture videos during training

Videos of the agents gameplay during training can be toggled by the `record_videos=True` flag and are uploaded to WandB automatically.


## Citing

Please cite this work as:

```
@inproceedings{
joshi2025benchmarking,
title={Benchmarking Massively Parallelized Multi-Task Reinforcement Learning for Robotics Tasks},
author={Viraj Joshi and Zifan Xu and Bo Liu and Peter Stone and Amy Zhang},
booktitle={Reinforcement Learning Conference},
year={2025},
url={https://openreview.net/forum?id=z0MM0y20I2}
}
```
