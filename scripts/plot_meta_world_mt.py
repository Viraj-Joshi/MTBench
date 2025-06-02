import os
from collections import defaultdict
import yaml

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

TASK_IDX_TO_NAME = {
    0:  "assembly",
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

def plot_results(runname_to_exps,all=True,rand=True):
    results = defaultdict(lambda : defaultdict(lambda: defaultdict(list)))

    for exp_name, run_names in sorted(runname_to_exps.items()):
        for run_name in run_names:
            current_run_path = os.path.join(log_dir, run_name)
            if [f for f in os.listdir(current_run_path) if f.startswith('config')]:
                cp = os.path.join(log_dir, run_name, "config.yaml")
            if [f for f in os.listdir(current_run_path) if f.startswith('summaries')]:
                sp = os.path.join(log_dir, run_name, "summaries")
            else:
                # sometimes wandb splits the run and we don't have the summaries folder in the same directory as the config.yaml
                # so skip extracting the data until we have the summaries folder
                continue 
            
            if not os.path.exists(sp):
                raise RuntimeError(f"summaries folder not found for {run_name}.")

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

                avg_task_success = []
                avg_task_return = []
                for tid in task_ids:
                    results_logger["success"][tid].append(np.mean(df[f"Episode/task_{tid}_success"].iloc[-5:]))
                    results_logger["return"][tid].append(np.mean(df[f"Episode/task_{tid}_reward"].iloc[-5:]))
                    avg_task_success.append(np.mean(df[f"Episode/task_{tid}_success"].iloc[-5:]))
                    avg_task_return.append(np.mean(df[f"Episode/task_{tid}_reward"].iloc[-5:]))
                results_logger["success_avg"][0].append(np.mean(avg_task_success))
                results_logger["return_avg"][0].append(np.mean(avg_task_return))

            except Exception as e:
                print(e)
                print(f"Running in error in {run_name}")

            print(f"Processed {run_name}")

    # barplot for success rate
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    task_indices = [str(k) for k in list(results[exp_name]["success"].keys()) if len(results[exp_name]["success"][k]) > 0]
    N = len(task_indices)
    ind = np.arange(N + 1)  # Add one more position for averages
    exp_names = list(results.keys())
    width = 0.8 / len(exp_names)
    bar_colors = []  # Store colors for each method
    rects = []

    average_successes = []
    average_returns = []

    std_successes = []
    std_returns = []

    # Plot regular bars first
    for i, exp_name in enumerate(exp_names):
        # Calculate success rates for individual tasks
        successes_mean = [np.mean(results[exp_name]["success"][task_idx]) for task_idx in results[exp_name]["success"].keys() 
                        if len(results[exp_name]["success"][task_idx]) > 0]
        returns_mean = [np.mean(results[exp_name]["return"][task_idx]) for task_idx in results[exp_name]["return"].keys() 
                        if len(results[exp_name]["return"][task_idx]) > 0]
        
        # Calculate average across all tasks for each method 'exp_name'
        average_successes.append(np.mean(successes_mean))
        std_successes.append(np.std(successes_mean))
        average_returns.append(np.mean(returns_mean))
        std_returns.append(np.std(returns_mean))
        
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
        
    print("Average success rates:")
    print_names = runname_to_exps.keys()
    print(print_names)
    for i, exp_name in enumerate(print_names):
        # print(f"{exp_name}: {average_successes[i]}")
        idx = exp_names.index(exp_name)
        print("%.2f +- %.2f"%(np.mean(results[exp_name]["success_avg"][0]) * 100, np.std(results[exp_name]["success_avg"][0]) * 100), end=" & ")
    print("\n")

    print("Average returns:")
    print(print_names)
    for i, exp_name in enumerate(print_names):
        # print(f"{exp_name}: {average_returns[i]}")
        idx = exp_names.index(exp_name)
        print("%.2f +- %.2f"%(np.mean(results[exp_name]["return_avg"][0]), np.std(results[exp_name]["return_avg"][0])), end=" & ")

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
    plt.tight_layout()
    title = "Success rates"
    if all:
        title += " - MT50"
    else:
        title += " - MT10"
    if rand:
        title += " - Rand"
    else:
        title += " - Fixed"
    plt.title(title)

    file_name = os.path.basename(__file__)
    if not os.path.exists(f"figures/{file_name}"):
        os.makedirs(f"figures/{file_name}")
    plt.savefig(f"figures/{file_name}/{title}.png")
    plt.close()
    
    
    
    print("Saved success rates plot to figures/")

if __name__ == "__main__":
    log_dir = "runs/"
    
    ### MT50
    # plot for fixed setting
    # runname_to_exps = {
    #     "PaCO": [f for f in os.listdir(log_dir) if "shppo_paco_mt50_fixed" in f],
    #     "CARE": [f for f in os.listdir(log_dir) if "mhppo_care_mt50_fixed" in f],
    #     "MHMOORE": [f for f in os.listdir(log_dir) if "mhppo_moore_mt50_fixed" in f],
    #     "Soft-Modularization": [f for f in os.listdir(log_dir) if "soft_modularization_mt50_fixed" in f],
    # }

    # plot_results(runname_to_exps)

    # plot for rand setting
    runname_to_exps = {
        "PaCO": [f for f in os.listdir(log_dir) if "shppo_paco_mt50_rand" in f],
        "MHCARE": [f for f in os.listdir(log_dir) if "mhppo_care_mt50_rand" in f],
        "SHCARE": [f for f in os.listdir(log_dir) if "shppo_care_mt50_rand" in f],
        "SHMOORE": [f for f in os.listdir(log_dir) if "shppo_moore_mt50_rand" in f],
        "MHMOORE": [f for f in os.listdir(log_dir) if "mhppo_moore_mt50_rand" in f],
        "Soft-Modularization": [f for f in os.listdir(log_dir) if "soft_modularization_mt50_rand" in f],
    }

    plot_results(runname_to_exps, all=True, rand=True)


    ### MT10
    # runname_to_exps = {
    #     "PaCO": [f for f in os.listdir(log_dir) if "shppo_paco_mt10_rand" in f],
    #     "MHCARE": [f for f in os.listdir(log_dir) if "mhppo_care_mt10_rand" in f],
    #     "SHCARE": [f for f in os.listdir(log_dir) if "shppo_care_mt10_rand" in f],
    #     "Soft-Modularization": [f for f in os.listdir(log_dir) if "ppo_soft_modularization_mt10_rand" in f],
    #     "MHMOORE": [f for f in os.listdir(log_dir) if "mhppo_moore_mt10_rand" in f],
    #     "SHMOORE": [f for f in os.listdir(log_dir) if "shppo_moore_mt10_rand" in f],
    # }
    # plot_results(runname_to_exps, all=False, rand=True)     

    ### GRPO
    # MT10
    # runname_to_exps = {
    #     "GRPO": [f for f in os.listdir(log_dir) if "vanilla_grpo_mt10_rand" in f],
    # }   
    # plot_results(runname_to_exps, all=False, rand=True)
    # ### MT50
    # runname_to_exps = {
    #     "GRPO": [f for f in os.listdir(log_dir) if "vanilla_grpo_mt50_rand" in f],
    # }
    # plot_results(runname_to_exps, all=True, rand=True)

    