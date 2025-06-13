import os
from collections import defaultdict
import yaml

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


if __name__ == "__main__":

    log_dir = "runs/"
    exp_folders = [f for f in os.listdir(log_dir) if f.startswith("0417_meta_world_pcgrad")]

    famo_with_norm_results = defaultdict(lambda: defaultdict(list))
    famo_wo_norm_results = defaultdict(lambda: defaultdict(list))

    for i, exp_folder in enumerate(exp_folders):
        sp = os.path.join(log_dir, exp_folder, "summaries")
        cp = os.path.join(log_dir, exp_folder, "config.yaml")
        if not os.path.exists(sp):
            print(f"Skipping {exp_folder}")
            continue

        try:
            config = yaml.load(open(cp, "r"), Loader=yaml.FullLoader)
            algo = config["train"]["params"]["algo"]["name"]
            task_ids = config["task_id"]

            if algo == "pcgrad_a2c":

                ef = [f for f in os.listdir(sp) if f.startswith("events.out.tfevents")][0]
                p = os.path.join(sp, ef)
                ea = event_accumulator.EventAccumulator(p)
                ea.Reload()
                tags = ea.Tags()["scalars"]
                data = {k: ea.Scalars(k) for k in tags if k.startswith("Episode/task")}
                min_len = min([len(v) for v in data.values()])
                print(exp_folder, min_len)
                df = pd.DataFrame({k: [x.value for x in v][:min_len] for k, v in data.items()})

                if "project" in exp_folder:
                    results_logger = famo_with_norm_results
                else:
                    results_logger = famo_wo_norm_results

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
    task_indices = [str(k) for k in list(famo_with_norm_results["success"].keys()) if len(famo_with_norm_results["success"][k]) > 0]
    with_norm_successes_mean = [np.mean(famo_with_norm_results["success"][task_idx]) for task_idx in famo_with_norm_results["success"].keys() if len(famo_with_norm_results["success"][task_idx]) > 0]
    with_norm_successes_std = [np.std(famo_with_norm_results["success"][task_idx]) for task_idx in famo_with_norm_results["success"].keys() if len(famo_with_norm_results["success"][task_idx]) > 0]

    wo_norm_successes_mean = [np.mean(famo_wo_norm_results["success"][task_idx]) for task_idx in famo_wo_norm_results["success"].keys() if len(famo_wo_norm_results["success"][task_idx]) > 0]
    wo_norm_successes_std = [np.std(famo_wo_norm_results["success"][task_idx]) for task_idx in famo_wo_norm_results["success"].keys() if len(famo_wo_norm_results["success"][task_idx]) > 0]
    
    N = len(task_indices)
    ind = np.arange(N)
    width = 0.25
    rects1 = ax.bar(ind, with_norm_successes_mean, width, yerr=with_norm_successes_std)
    rects2 = ax.bar(ind + width, wo_norm_successes_mean, width, yerr=wo_norm_successes_std)
    ax.set_xticks(ind+width)
    ax.set_xticklabels(task_indices)
    ax.legend( (rects1[0], rects2[0]), ('PCGrad with actor projection', 'PCGrad wo actor projection') )

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
    mt_ppo_results = famo_with_norm_results
    pcgrad_ppo_results = famo_wo_norm_results
    mt_returns_mean = [np.mean(mt_ppo_results["return"][task_idx]) for task_idx in mt_ppo_results["return"].keys() if len(mt_ppo_results["return"][task_idx]) > 0]
    mt_returns_std = [np.std(mt_ppo_results["return"][task_idx]) for task_idx in mt_ppo_results["return"].keys() if len(mt_ppo_results["return"][task_idx]) > 0]

    pcgrad_returns_mean = [np.mean(pcgrad_ppo_results["return"][task_idx]) for task_idx in pcgrad_ppo_results["return"].keys() if len(pcgrad_ppo_results["return"][task_idx]) > 0]
    pcgrad_returns_std = [np.std(pcgrad_ppo_results["return"][task_idx]) for task_idx in pcgrad_ppo_results["return"].keys() if len(pcgrad_ppo_results["return"][task_idx]) > 0]

    N = len(task_indices)
    ind = np.arange(N)
    width = 0.25
    rects1= ax.bar(ind, mt_returns_mean, width, yerr=mt_returns_std)
    rects2 = ax.bar(ind + width, pcgrad_returns_mean, width, yerr=pcgrad_returns_std)
    ax.set_xticks(ind+width)
    ax.set_xticklabels(task_indices)
    ax.legend( (rects1[0], rects2[0]), ('PCGrad with actor projection', 'PCGrad wo actor projection') )

    plt.xlabel("Task index")
    plt.ylabel("Episodic Reward")
    plt.title("Episodic reward for each task")
    file_name = os.path.basename(__file__)
    if not os.path.exists(f"debug/{file_name}"):
        os.makedirs(f"debug/{file_name}")
    plt.savefig(f"debug/{file_name}/reward.png")
    plt.close()

    print("Overall success rate:")
    print("FAMO with w norm:", np.mean([np.mean(mt_ppo_results["success"][task_idx]) for task_idx in mt_ppo_results["success"].keys() if len(mt_ppo_results["success"][task_idx]) > 0]))
    print("FAMO wo w norm:", np.mean([np.mean(pcgrad_ppo_results["success"][task_idx]) for task_idx in pcgrad_ppo_results["success"].keys() if len(pcgrad_ppo_results["success"][task_idx]) > 0]))

    print("Overall episodic reward:")
    print("FAMO with w norm:", np.mean([np.mean(mt_ppo_results["return"][task_idx]) for task_idx in mt_ppo_results["return"].keys() if len(mt_ppo_results["return"][task_idx]) > 0]))
    print("FAMO wo w norm:", np.mean([np.mean(pcgrad_ppo_results["return"][task_idx]) for task_idx in pcgrad_ppo_results["return"].keys() if len(pcgrad_ppo_results["return"][task_idx]) > 0]))