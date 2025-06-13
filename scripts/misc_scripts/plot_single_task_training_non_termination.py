import os
from collections import defaultdict

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Load data
log_dir = "runs/"
exp_folders = [f for f in os.listdir(log_dir) if f.startswith("0405-meta-world-mt-PPO-non-termination-task") \
               and int(f.split("_")[-1].split("-")[0]) < 10]
exp_folders = sorted(exp_folders, key=lambda x: int(x.split("task-")[-1].split("_")[0]))

result_log = defaultdict(lambda: defaultdict(list))

for i, exp_folder in enumerate(exp_folders):
    p = os.path.join(log_dir, exp_folder, "summaries")
    if not os.path.exists(p):
        print(f"Skipping {exp_folder}")
        continue

    try:
        task_indice = int(exp_folder.split("task-")[-1].split("_")[0])
        ef = [f for f in os.listdir(p) if f.startswith("events.out.tfevents")][0]
        p = os.path.join(p, ef)
        ea = event_accumulator.EventAccumulator(p)
        ea.Reload()
        tags = ea.Tags()["scalars"]
        data = {k: ea.Scalars(k) for k in tags if k in [f"Episode/task_{task_indice}_success", f"Episode/task_{task_indice}_reward"]}
        df = pd.DataFrame({k: [x.value for x in v] for k, v in data.items()})
        result_log[task_indice]["success"].append(np.mean(df[f"Episode/task_{task_indice}_success"].iloc[-10:]))
        result_log[task_indice]["return"].append(np.mean(df[f"Episode/task_{task_indice}_reward"].iloc[-10:]))
    except Exception as e:
        print(e)
        print(f"Skipping {exp_folder}")
        continue

    print(f"Processing {exp_folder}: {result_log[task_indice]['success'][-1]}", end="\r")

# barplot for the success rate
fig, ax = plt.subplots(1, 1, figsize=(18, 8))
task_indices = [str(k) for k in list(result_log.keys()) if len(result_log[k]["success"]) > 0]
successes_mean = [np.mean(result_log[task_idx]["success"]) for task_idx in result_log.keys() if len(result_log[task_idx]["success"]) > 0]
successes_std = [np.std(result_log[task_idx]["success"]) for task_idx in result_log.keys() if len(result_log[task_idx]["success"]) > 0]
returns_mean = [np.mean(result_log[task_idx]["return"]) for task_idx in result_log.keys() if len(result_log[task_idx]["return"]) > 0]
returns_std = [np.std(result_log[task_idx]["return"]) for task_idx in result_log.keys() if len(result_log[task_idx]["return"]) > 0]

ax.bar(task_indices, successes_mean, yerr=successes_std)
plt.xlabel("Task index")
plt.ylabel("Success rate")
# plt.xticks(rotation=90)
plt.title("Success rate for each task")
file_name = os.path.basename(__file__)
if not os.path.exists(f"debug/{file_name}"):
    os.makedirs(f"debug/{file_name}")
plt.savefig(f"debug/{file_name}/success_rate.png")

# barplot for the success rate
fig, ax = plt.subplots(1, 1, figsize=(18, 8))
ax.bar(task_indices, returns_mean, yerr=returns_std)
plt.xlabel("Task index")
plt.ylabel("Episodic Reward")
# plt.xticks(rotation=90)
plt.title("Episodic reward for each task")
file_name = os.path.basename(__file__)
if not os.path.exists(f"debug/{file_name}"):
    os.makedirs(f"debug/{file_name}")
plt.savefig(f"debug/{file_name}/reward.png")

