import os
from collections import defaultdict
import yaml

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


if __name__ == "__main__":

    log_dir = "runs/"
    exp_folders = [f for f in os.listdir(log_dir) if f.startswith("0416-single-task")]
    print(exp_folders)

    w_random_init_progress_results = defaultdict(lambda: defaultdict(list))
    wo_random_init_progress_results = defaultdict(lambda: defaultdict(list))

    for i, exp_folder in enumerate(exp_folders):
        sp = os.path.join(log_dir, exp_folder, "summaries")
        cp = os.path.join(log_dir, exp_folder, "config.yaml")
        if not os.path.exists(sp):
            print(f"Skipping {exp_folder}")
            continue

        try:
            config = yaml.load(open(cp, "r"), Loader=yaml.FullLoader)
            init_at_random_progress = config["train"]["params"]["config"]["init_at_random_progress"]
            task_ids = config["task_id"]

            ef = [f for f in os.listdir(sp) if f.startswith("events.out.tfevents")][0]
            p = os.path.join(sp, ef)
            ea = event_accumulator.EventAccumulator(p)
            ea.Reload()
            tags = ea.Tags()["scalars"]
            data = {k: ea.Scalars(k) for k in tags if k.startswith("Episode/task")}
            min_len = min([len(v) for v in data.values()])
            print(exp_folder, min_len)
            df = pd.DataFrame({k: [x.value for x in v][:min_len] for k, v in data.items()})

            if init_at_random_progress:
                results_logger = w_random_init_progress_results
            else:
                results_logger = wo_random_init_progress_results

            for tid in task_ids:
                results_logger["success"][tid].append(np.mean(df[f"Episode/task_{tid}_success"].iloc[-5:]))
                results_logger["return"][tid].append(np.mean(df[f"Episode/task_{tid}_reward"].iloc[-5:]))

        except Exception as e:
            print(e)
            print(f"Skipping {exp_folder}")
            continue

        # print(f"Processing {exp_folder}: {results_logger['success'][tid][-1]}", end="\r")

    # barplot for success rate
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    task_indices = [str(k) for k in list(w_random_init_progress_results["success"].keys()) if len(w_random_init_progress_results["success"][k]) > 0]

    w_random_successes_mean = [np.mean(w_random_init_progress_results["success"][task_idx]) for task_idx in w_random_init_progress_results["success"].keys() if len(w_random_init_progress_results["success"][task_idx]) > 0]
    w_random_successes_std = [np.std(w_random_init_progress_results["success"][task_idx]) for task_idx in w_random_init_progress_results["success"].keys() if len(w_random_init_progress_results["success"][task_idx]) > 0]

    wo_random_successes_mean = [np.mean(wo_random_init_progress_results["success"][task_idx]) for task_idx in wo_random_init_progress_results["success"].keys() if len(wo_random_init_progress_results["success"][task_idx]) > 0]
    wo_random_successes_std = [np.std(wo_random_init_progress_results["success"][task_idx]) for task_idx in wo_random_init_progress_results["success"].keys() if len(wo_random_init_progress_results["success"][task_idx]) > 0]
    
    N = len(task_indices)
    ind = np.arange(N)
    width = 0.25
    rects1= ax.bar(ind, w_random_successes_mean, width, yerr=w_random_successes_std)
    rects2 = ax.bar(ind + width, wo_random_successes_mean, width, yerr=wo_random_successes_std)
    ax.set_xticks(ind+width)
    ax.set_xticklabels(task_indices)
    ax.legend( (rects1[0], rects2[0]), ('with random_init_progress', 'w/o random_init_progress') )

    plt.xlabel("Task index")
    plt.ylabel("Success rate")

    plt.title("Success rate for each task")
    file_name = os.path.basename(__file__)
    if not os.path.exists(f"debug/{file_name}"):
        os.makedirs(f"debug/{file_name}")
    plt.savefig(f"debug/{file_name}/success_rate.png")
    plt.close()

    # barplot for the returns
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    w_random_returns_mean = [np.mean(w_random_init_progress_results["return"][task_idx]) for task_idx in w_random_init_progress_results["return"].keys() if len(w_random_init_progress_results["return"][task_idx]) > 0]
    w_random_returns_std = [np.std(w_random_init_progress_results["return"][task_idx]) for task_idx in w_random_init_progress_results["return"].keys() if len(w_random_init_progress_results["return"][task_idx]) > 0]

    wo_random_returns_mean = [np.mean(wo_random_init_progress_results["return"][task_idx]) for task_idx in wo_random_init_progress_results["return"].keys() if len(wo_random_init_progress_results["return"][task_idx]) > 0]
    wo_random_returns_std = [np.std(wo_random_init_progress_results["return"][task_idx]) for task_idx in wo_random_init_progress_results["return"].keys() if len(wo_random_init_progress_results["return"][task_idx]) > 0]

    N = len(task_indices)
    ind = np.arange(N)
    width = 0.25
    rects1= ax.bar(ind, w_random_returns_mean, width, yerr=w_random_returns_std)
    rects2 = ax.bar(ind + width, wo_random_returns_mean, width, yerr=wo_random_returns_std)
    ax.set_xticks(ind+width)
    ax.set_xticklabels(task_indices)
    ax.legend( (rects1[0], rects2[0]), ('with random_init_progress', 'w/o random_init_progress') )

    plt.xlabel("Task index")
    plt.ylabel("Episodic Reward")
    plt.title("Episodic reward for each task")
    file_name = os.path.basename(__file__)
    if not os.path.exists(f"debug/{file_name}"):
        os.makedirs(f"debug/{file_name}")
    plt.savefig(f"debug/{file_name}/reward.png")
    plt.close()

    print("Overall success rate:")
    print("With random initial progress:", np.mean([np.mean(w_random_init_progress_results["success"][task_idx]) for task_idx in w_random_init_progress_results["success"].keys() if len(w_random_init_progress_results["success"][task_idx]) > 0]))
    print("Without random initial progress:", np.mean([np.mean(wo_random_init_progress_results["success"][task_idx]) for task_idx in wo_random_init_progress_results["success"].keys() if len(wo_random_init_progress_results["success"][task_idx]) > 0]))

    print("Overall episodic reward:")
    print("With random initial progress:", np.mean([np.mean(w_random_init_progress_results["return"][task_idx]) for task_idx in w_random_init_progress_results["return"].keys() if len(w_random_init_progress_results["return"][task_idx]) > 0]))
    print("Without random initial progress:", np.mean([np.mean(wo_random_init_progress_results["return"][task_idx]) for task_idx in wo_random_init_progress_results["return"].keys() if len(wo_random_init_progress_results["return"][task_idx]) > 0]))