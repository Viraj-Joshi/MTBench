import os
from collections import defaultdict
import yaml

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


if __name__ == "__main__":

    log_dir = "runs/"
    dict_run_names = {
        "CAGrad": [
            "1221_ppo_CAGrad_larger_entropy_21-20-08-36",
            "1221_ppo_CAGrad_larger_entropy_21-18-30-07",
            "1221_ppo_CAGrad_larger_entropy_21-16-55-40",
            "1221_ppo_CAGrad_larger_entropy_21-15-19-49",
            "1221_ppo_CAGrad_larger_entropy_21-13-39-10",
        ],
        "FAMO": [
            "test_ppo_famo_23-20-53-00",
            "test_ppo_famo_23-18-39-43",
            "test_ppo_famo_23-16-26-04",
            "test_ppo_famo_23-14-10-28",
            "test_ppo_famo_23-11-59-49",
        ],
        "MT-PPO": [
            "1222_ppo_vanilla_23-16-08-17",
            "1222_ppo_vanilla_23-14-52-48",
            "1222_ppo_vanilla_23-13-34-57",
            "1222_ppo_vanilla_23-12-20-07",
            "1222_ppo_vanilla_23-11-08-54",
        ],
        "PCGrad": [
            "1222_ppo_pcgrad_23-20-01-55",
            "1222_ppo_pcgrad_23-17-48-09",
            "1222_ppo_pcgrad_23-15-35-27",
            "1222_ppo_pcgrad_23-13-20-52",
            "1222_ppo_pcgrad_23-11-07-05",
        ],
        "Multihead-PPO": [
            "1222_ppo_multihead_23-21-23-56",
            "1222_ppo_multihead_23-19-39-18",
            "1222_ppo_multihead_23-17-55-45",
            "1222_ppo_multihead_23-16-11-20",
            "1222_ppo_multihead_23-14-29-31",
        ]
    }

    results = defaultdict(lambda : defaultdict(lambda: defaultdict(list)))

    for exp_name, run_names in dict_run_names.items():
        for run_name in run_names:
            sp = os.path.join(log_dir, run_name, "summaries")
            cp = os.path.join(log_dir, run_name, "config.yaml")
            assert os.path.exists(sp), f"summaries folder not found for {run_name}"

            try:
                config = yaml.load(open(cp, "r"), Loader=yaml.FullLoader)
                algo = config["train"]["params"]["algo"]["name"]
                task_ids = config["task_id"]

                ef = [f for f in os.listdir(sp) if f.startswith("events.out.tfevents")][0]
                p = os.path.join(sp, ef)
                ea = event_accumulator.EventAccumulator(p)
                ea.Reload()
                tags = ea.Tags()["scalars"]
                data = {k: ea.Scalars(k) for k in tags if k.startswith("Episode/task")}
                df = pd.DataFrame({k: [x.value for x in v] for k, v in data.items()})

                results_logger = results[exp_name]

                for tid in task_ids:
                    results_logger["success"][tid].append(np.mean(df[f"Episode/task_{tid}_success"].iloc[-5:]))
                    results_logger["return"][tid].append(np.mean(df[f"Episode/task_{tid}_reward"].iloc[-5:]))

            except Exception as e:
                print(e)
                print(f"Running in error in {run_name}")

            # print(f"Processing {run_name}: {results_logger['success'][tid][-1]}", end="\r")

    # barplot for success rate
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    task_indices = [str(k) for k in list(results[exp_name]["success"].keys()) if len(results[exp_name]["success"][k]) > 0]
    task_indices = sorted(task_indices, key=lambda x: int(x))
    N = len(task_indices)
    ind = np.arange(N)
    exp_names = list(results.keys())
    width = 0.8 / len(exp_names)
    rects = []

    for i, exp_name in enumerate(exp_names):
        successes_mean = [np.mean(results[exp_name]["success"][int(task_idx)]) for task_idx in task_indices]
        successes_std = [np.std(results[exp_name]["success"][int(task_idx)]) for task_idx in task_indices]
        indi = np.arange(N) + i * width
        rect = ax.bar(indi, successes_mean, width * 0.9, yerr=successes_std)
        rects.append(rect)
        print("Average success rate for %s: %.4f" % (exp_name, np.mean(successes_mean)))
    ax.set_xticks(ind + width * (len(exp_names) - 1) / 2)
    ax.set_xticklabels(task_indices)
    ax.legend( rects, exp_names )
    
    plt.xlabel("Task index")
    plt.ylabel("Success rate")

    plt.title("Success rates - MT10")
    file_name = os.path.basename(__file__)
    if not os.path.exists(f"debug/{file_name}"):
        os.makedirs(f"debug/{file_name}")
    plt.savefig(f"debug/{file_name}/success_rates.png")
    plt.close()