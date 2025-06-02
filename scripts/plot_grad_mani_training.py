import os
from collections import defaultdict
import yaml

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

TASK_IDX_TO_NAME = {
    0:  "assemble",
    1:  "basketball",
    2:  "bin_picking",
    3:  "box_close",
    4:  "button_press_topdown",
    5:  "button_press_topdown_wall",
    6:  "button_press",
    7:  "button_press_wall",
    8:  "coffee_button",
    9:  "coffee_pull",
    10: "coffee_push",
    11: "dial_turn",
    12: "disassemble",
    13: "door_close",
    14: "door_lock",
    15: "door_unlock",
    16: "door_open",
    17: "drawer_close",
    18: "drawer_open",
    19: "faucet_close",
    20: "faucet_open",
    21: "hammer",
    22: "hand_insert",
    23: "handle_press_side",
    24: "handle_press",
    25: "handle_pull_side",
    26: "handle_pull",
    27: "lever_pull",
    28: "peg_insert_side",
    29: "peg_unplug_side",
    30: "pick_out_of_hole",
    31: "pick_place",
    32: "pick_place_wall",
    33: "plate_slide_back_side",
    34: "plate_slide_back",
    35: "plate_slide_side",
    36: "plate_slide",
    37: "push_back",
    38: "push",
    39: "push_wall",
    40: "reach",
    41: "reach_wall",
    42: "shelf_place",
    43: "soccer",
    44: "stick_pull",
    45: "stick_push",
    46: "sweep_into_goal",
    47: "sweep",
    48: "window_close",
    49: "window_open",
}

if __name__ == "__main__":
    log_dir = "runs/"
    runname_to_exps = {
        "Vanilla": [f for f in os.listdir(log_dir) if "0123_MT50_ppo_vanilla" in f],
        "Cagrad": [f for f in os.listdir(log_dir) if "0123_MT50_ppo_cagrad" in f],
        "PCGrad": [f for f in os.listdir(log_dir) if "0123_MT50_ppo_pcgrad" in f],
        "Multihead": [f for f in os.listdir(log_dir) if "0123_MT50_ppo_multihead" in f],
        "FAMO": [f for f in os.listdir(log_dir) if  "0123_MT50_ppo_famo" in f]
    }

    results = defaultdict(lambda : defaultdict(lambda: defaultdict(list)))

    for exp_name, run_names in sorted(runname_to_exps.items()):
        for run_name in run_names:
            sp = os.path.join(log_dir, run_name, "summaries")
            cp = os.path.join(log_dir, run_name, "config.yaml")
            if not os.path.exists(sp):
                print(f"skipping summaries folder not found for {run_name}. wandb probably split the run or failed")
                continue

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
    fig, ax = plt.subplots(1, 1, figsize=(18, 36))
    task_indices = [str(k) for k in list(results[exp_name]["success"].keys()) if len(results[exp_name]["success"][k]) > 0]
    N = len(task_indices)
    ind = np.arange(N + 1)  # Add one more position for averages
    exp_names = list(results.keys())
    width = 0.8 / len(exp_names)
    bar_colors = []  # Store colors for each method
    rects = []

    average_successes = []

    # Plot regular bars first
    for i, exp_name in enumerate(exp_names):
        # Calculate success rates for individual tasks
        successes_mean = [np.mean(results[exp_name]["success"][task_idx]) for task_idx in results[exp_name]["success"].keys() 
                        if len(results[exp_name]["success"][task_idx]) > 0]
        
        # Calculate average across all tasks for each method 'exp_name'
        average_successes.append(np.mean(successes_mean))
        
        # Plot individual task bars
        indi = ind[:-1] - i * width
        rect = plt.barh(indi, successes_mean, width * 0.9, label=exp_name)
        bar_colors.append(rect[0].get_facecolor())  # Store the color

    # Plot average bars after the separator line
    for i, exp_name in enumerate(exp_names):
        # Plot average bar (positioned at the bottom)
        avg_pos = ind[-1] - i * width
        plt.barh([avg_pos], [average_successes[i]], width * 0.9, 
                color=bar_colors[i],
                hatch='///')

    # Calculate the center of each group of bars
    group_centers = ind[:-1] - (len(exp_names) - 1) * width / 2
    avg_center = ind[-1] - (len(exp_names) - 1) * width / 2

    # Create labels including the average group
    task_labels = [TASK_IDX_TO_NAME[int(idx)] for idx in task_indices]
    all_labels = task_labels + ['Average']

    # Set y-axis ticks and labels
    plt.yticks(np.concatenate([group_centers, [avg_center]]), labels=all_labels)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel("Task")
    plt.xlabel("Success rate")

    plt.gca().invert_yaxis()
    plt.title("Success rates - MT10")

    file_name = os.path.basename(__file__)
    if not os.path.exists(f"debug/{file_name}"):
        os.makedirs(f"debug/{file_name}")
    plt.tight_layout()
    plt.savefig(f"debug/{file_name}/success_rates.png")
    plt.close()
    
    
    
    print("Saved success rates plot to debug/")



    # barplot for the returns
    # fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    # N = len(task_indices)
    # ind = np.arange(N)
    # exp_names = list(results.keys())
    # width = 0.8 / len(exp_names)
    # rects = []

    # for i, exp_name in enumerate(exp_names):
    #     returns_mean = [np.mean(results[exp_name]["return"][task_idx]) for task_idx in results[exp_name]["return"].keys() if len(results[exp_name]["return"][task_idx]) > 0]
    #     returns_std = [np.std(results[exp_name]["return"][task_idx]) for task_idx in results[exp_name]["return"].keys() if len(results[exp_name]["return"][task_idx]) > 0]
    #     N = len(task_indices)
    #     indi = np.arange(N) + i * width
    #     rect = ax.bar(indi, returns_mean, width * 0.9, yerr=returns_std)
    #     rects.append(rect)
    #     print("Average return for %s: %.4f" % (exp_name, np.mean(returns_mean)))
    
    # ax.set_xticks(ind + width * (len(exp_names) - 1) / 2)
    # ax.set_xticklabels(task_indices)
    # ax.legend( rects, exp_names )

    # plt.xlabel("Task index")
    # plt.ylabel("Episodic Reward")

    # plt.title("Episodic rewards - MT10")
    # file_name = os.path.basename(__file__)
    # if not os.path.exists(f"debug/{file_name}"):
    #     os.makedirs(f"debug/{file_name}")
    # plt.savefig(f"debug/{file_name}/rewards.png")
    # plt.close()