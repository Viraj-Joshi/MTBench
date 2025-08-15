import os
from collections import defaultdict
import glob
import multiprocessing

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm

# --- Constants ---
BOOTSTRAP_ITERATIONS = 1000
CI_PERCENT = 95.0

import os
from collections import defaultdict
import glob
import multiprocessing

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm

# --- Your existing constants and utility functions go here ---
# (BOOTSTRAP_ITERATIONS, CI_PERCENT, get_percentile_ci, bootstrap_confidence_interval)

TASK_IDX_TO_NAME = {
    0:  "assembly", 1:  "basketball", 2:  "bin_picking", 3:  "box_close", 4:  "button_press_topdown",
    5:  "button_press_topdown_wall", 6:  "button_press", 7:  "button_press_wall", 8:  "coffee_button",
    9:  "coffee_pull", 10: "coffee_push", 11: "dial_turn", 12: "disassemble", 13: "door_close",
    14: "door_lock", 15: "door_unlock", 16: "door_open", 17: "drawer_close", 18: "drawer_open",
    19: "faucet_close", 20: "faucet_open", 21: "hammer", 22: "hand_insert", 23: "handle_press_side",
    24: "handle_press", 25: "handle_pull_side", 26: "handle_pull", 27: "lever_pull", 28: "peg_insert_side",
    29: "peg_unplug_side", 30: "pick_out_of_hole", 31: "pick_place", 32: "pick_place_wall",
    33: "plate_slide_back_side", 34: "plate_slide_back", 35: "plate_slide_side", 36: "plate_slide",
    37: "push_back", 38: "push", 39: "push_wall", 40: "reach", 41: "reach_wall", 42: "shelf_place",
    43: "soccer", 44: "stick_pull", 45: "stick_push", 46: "sweep_into_goal", 47: "sweep",
    48: "window_close", 49: "window_open",
}

def _fetch_performance_worker(job):
    """
    Worker function for multiprocessing. Loads a single event file.
    Args:
        job (tuple): A tuple containing (task_id_str, run_dir).
    Returns:
        tuple: A tuple containing (task_id_str, final_performance_value).
    """
    task_id_str, run_dir = job
    success_tag = "Episode/average_task_success_rate"
    try:
        event_files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
        if not event_files:
            return task_id_str, None

        ea = event_accumulator.EventAccumulator(sorted(event_files)[-1], size_guidance={'scalars': 0})
        ea.Reload()
        
        if success_tag in ea.Tags()['scalars']:
            scalar_events = ea.Scalars(success_tag)
            if scalar_events:
                return task_id_str, scalar_events[-1].value
    except Exception:
        return task_id_str, None # Return None on failure
    return task_id_str, None

def generate_single_task_latex_table(log_dir, experiment_config):
    """
    Generates a LaTeX table for single-task performance, using multiprocessing to speed up data loading.
    """
    print("\n" + "="*80)
    print("Generating LaTeX Table with Bootstrapped CI (using multiprocessing)...")

    # --- 1. Find and group runs by task ID (this part is fast) ---
    single_task_config = experiment_config.get("Single-Task Average (1M)")
    if not single_task_config:
        print("Could not find 'Single-Task Average (1M)' in experiment configuration.")
        return

    pattern = single_task_config['pattern']
    run_dirs = sorted(glob.glob(pattern))
    if not run_dirs:
        print(f"Warning: No single-task runs found using pattern: {pattern}")
        return

    grouped_runs = defaultdict(list)
    for run_dir in run_dirs:
        try:
            task_id_str = run_dir.split('_rand_envs')[0]
            grouped_runs[task_id_str].append(run_dir)
        except IndexError:
            print(f"Warning: Could not parse task ID from directory name: {run_dir}")

    # --- 2. Create a flat list of jobs and process them in parallel ---
    jobs = [(task_id, rd) for task_id, rds in grouped_runs.items() for rd in rds]
    print(f"Found {len(jobs)} total runs across {len(grouped_runs)} tasks. Loading in parallel...")

    regrouped_performances = defaultdict(list)
    with multiprocessing.Pool() as pool:
        pbar = tqdm(pool.imap_unordered(_fetch_performance_worker, jobs), total=len(jobs))
        for task_id_str, performance in pbar:
            if performance is not None:
                regrouped_performances[task_id_str].append(performance)

    # --- 3. Calculate CI for each task from the loaded data ---
    task_final_results = []
    for task_id_str, seed_performances in regrouped_performances.items():
        try:
            task_index = int(task_id_str.split('_')[-1])
            task_name = TASK_IDX_TO_NAME.get(task_index, f"Task {task_index}").replace('_', ' ').title()
        except (ValueError, IndexError):
            task_name = task_id_str

        mean_perf, lower_ci, upper_ci = bootstrap_confidence_interval(seed_performances)
        margin_of_error = (upper_ci - lower_ci) / 2.0 if not np.isnan(upper_ci) else np.nan
        
        task_final_results.append({
            'task': task_name,
            'mean': mean_perf,
            'moe': margin_of_error
        })

    if not task_final_results:
        print("No valid performance data could be extracted.")
        return

    # --- 4. Calculate global average and print the LaTeX table ---
    per_task_means = [item['mean'] for item in task_final_results if not np.isnan(item['mean'])]
    global_mean, global_lower_ci, global_upper_ci = bootstrap_confidence_interval(per_task_means)
    global_margin_of_error = (global_upper_ci - global_lower_ci) / 2.0 if not np.isnan(global_upper_ci) else np.nan

    print("\n--- Generated LaTeX Code ---")
    print("(Requires LaTeX packages: booktabs, amsmath)")
    print("\\begin{table}[h!]")
    print("\\centering")
    print("\\caption{Final success rate for single-task runs. Each task's performance is the mean across seeds, reported with its 95\\% CI margin of error. The global average is the mean of these per-task averages, also with a 95\\% CI.}")
    print("\\label{tab:single_task_performance}")
    print("\\begin{tabular}{lc}")
    print("\\toprule")
    print("\\textbf{Task} & \\textbf{Final Success Rate (Mean $\\pm$ 95\\% CI)} \\\\")
    print("\\midrule")
    
    for item in sorted(task_final_results, key=lambda x: x['task']):
        if not np.isnan(item['mean']):
            print(f"{item['task']} & ${item['mean']:.3f} \\pm {item['moe']:.3f}$ \\\\")
        else:
            print(f"{item['task']} & --- \\\\")
        
    print("\\midrule")
    if not np.isnan(global_mean):
        print(f"\\textbf{{Average (all tasks)}} & \\textbf{{${global_mean:.3f} \\pm {global_margin_of_error:.3f}$}} \\\\")
    else:
        print(f"\\textbf{{Average (all tasks)}} & --- \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("="*80 + "\n")
# ======================================================================================
# MAIN EXECUTION
# ======================================================================================
if __name__ == "__main__":
    LOG_DIR = "runs/"
    OUTPUT_DIR = "scripts/"
    SUCCESS_TAG = "Episode/average_task_success_rate"

    K = 5
    m = 32768
    n = 4096      # Default number of environments
    T = 32

    # --- 2. Define Experiments to Plot ---
    EXPERIMENTS_TO_PLOT = {
        "Single-Task Average (1M)": {
            'type': 'single-task',
            'n': 4096,
            'pattern': os.path.join(LOG_DIR, "single-task", "uid_ppo_vanilla_task_*"),
            'style': {'color': 'firebrick', 'marker': 's', 'label': 'Single-Task Average (1M)'}
        }
    }

    # --- 6. NEW: Generate and Print the LaTeX Table ---
    generate_single_task_latex_table(LOG_DIR, EXPERIMENTS_TO_PLOT)