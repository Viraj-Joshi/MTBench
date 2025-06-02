import os
from collections import defaultdict
import re

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Load data
log_dir = "runs"
exp_folders = [s for s in os.listdir(log_dir) if s.startswith("tune_famo")]
task_indices = [0, 4, 5, 6, 7, 9, 10, 11]

final_success_rates = []
final_rewards = []

reward_logs = defaultdict(list)
success_logs = defaultdict(list)

for i, exp_folder in enumerate(exp_folders):
    print("Processing: %d/%d" %(i, len(exp_folders), ), end="\r")
    try:
        p = os.path.join(log_dir, exp_folder, "summaries")

        ef = [f for f in os.listdir(p) if f.startswith("events.out.tfevents")][0]
        p = os.path.join(p, ef)
    
        ea = event_accumulator.EventAccumulator(p)
        ea.Reload()
        tags = ea.Tags()["scalars"]
        data = {k: ea.Scalars(k) for k in tags}
        df = pd.DataFrame({k: [x.value for x in v] for k, v in data.items()})

        hyperparams = tuple(float(e) for e in exp_folder.split("_")[2:-1] if re.match(r"\d+\.\d+", e))

        success = []
        rewards = []
        for t in task_indices:
            success.append(np.mean(df[f"Episode/task_{t}_success"].iloc[-10:]))
            rewards.append(np.mean(df[f"Episode/task_{t}_reward"].iloc[-10:]))

        reward_logs[hyperparams].append(np.mean(rewards))
        success_logs[hyperparams].append(np.mean(success))
    except Exception as e:
        print(f"Error processing {exp_folder}: {e}")

print("\n\n")

for k in reward_logs.keys():
    print(f"Num support: {len(reward_logs[k])};", end="\t")
    print(f"Hyperparams: {k};", end="\t")
    print(f"Mean reward: {np.mean(reward_logs[k])};", end="\t")
    print(f"Mean success rate: {np.mean(success_logs[k])};")