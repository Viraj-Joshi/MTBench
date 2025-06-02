import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

import isaacgym
from isaacgymenvs.tasks.franka.vec_task.franka_base import TASK_IDX_TO_NAME

# Load data
log_dir = "runs/1011_sac_single_task"
exp_folders = os.listdir(log_dir)
exp_folders = sorted(exp_folders, key=lambda x: int(x.split("1011-simgle_task-")[-1].split("-non_termination-")[0]))

task_indices = []
task_names = []
final_success_rates = []
ever_success_rates = []
final_rewards = []

for i, exp_folder in enumerate(exp_folders):
    p = os.path.join(log_dir, exp_folder)
    if not os.path.exists(p):
        final_success_rates.append(-0.1)
        final_rewards.append(-1.0)
        task_indices.append(exp_folder.split("1011-simgle_task-")[-1].split("-non_termination-")[0])
        task_names.append(TASK_IDX_TO_NAME[int(task_indices[-1])])
        continue
    try:
        ef = [f for f in os.listdir(p) if f.startswith("events.out.tfevents")][0]
        p = os.path.join(p, ef)
        ea = event_accumulator.EventAccumulator(p)
        ea.Reload()
        tags = ea.Tags()["scalars"]
        data = {k: ea.Scalars(k) for k in tags}
        # df = pd.DataFrame({k: [x.value for x in v] for k, v in data.items()})

        task_indice = exp_folder.split("1011-simgle_task-")[-1].split("-non_termination-")[0]
        for k, v in data.items():
            if k == "episode_cumulative/success":
                ever_success_rates.append(np.mean([x.value for x in v[-10:]]))
            if k == f"Episode/task_{task_indice}_reward":
                final_rewards.append(np.mean([x.value for x in v[-10:]]))
            if k == f"Episode/task_{task_indice}_success":
                final_success_rates.append(np.mean([x.value for x in v[-10:]]))
        task_indices.append(task_indice)
        task_names.append(TASK_IDX_TO_NAME[int(task_indices[-1])])

        print(f"Processing {exp_folder}: {final_success_rates[-1]}")
    except Exception as e:
        # import ipdb; ipdb.set_trace()
        print("Error in processing", exp_folder)
        print(e)
        pass

# barplot for the success rate
fig, ax = plt.subplots(1, 1, figsize=(18, 8))
ax.bar(task_indices, final_success_rates)
plt.xlabel("Task index")
plt.ylabel("Final Success rate")
# plt.xticks(rotation=90)
plt.title("Success rate for each task")
file_name = os.path.basename(__file__)
if not os.path.exists(f"debug/{file_name}"):
    os.makedirs(f"debug/{file_name}")
plt.savefig(f"debug/{file_name}/final_success_rate.png")
plt.close()

# barplot for the success rate
fig, ax = plt.subplots(1, 1, figsize=(18, 8))
ax.bar(task_indices, final_success_rates)
plt.xlabel("Task index")
plt.ylabel("Ever Success rate")
# plt.xticks(rotation=90)
plt.title("Success rate for each task")
file_name = os.path.basename(__file__)
if not os.path.exists(f"debug/{file_name}"):
    os.makedirs(f"debug/{file_name}")
plt.savefig(f"debug/{file_name}/ever_success_rate.png")
plt.close()

# barplot for the success rate
fig, ax = plt.subplots(1, 1, figsize=(18, 8))
ax.bar(task_indices, final_rewards)
plt.xlabel("Task index")
plt.ylabel("Episodic Reward")
# plt.xticks(rotation=90)
plt.title("Episodic reward for each task")
file_name = os.path.basename(__file__)
if not os.path.exists(f"debug/{file_name}"):
    os.makedirs(f"debug/{file_name}")
plt.savefig(f"debug/{file_name}/reward.png")
plt.close()

