# Filename: generate_final_performance_plot.py

import os
import re
from collections import defaultdict
import multiprocessing

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm

# ==============================================================================
# SCRIPT CONFIGURATION
# ==============================================================================
LOG_DIR = 'runs/batch_scaling/'
SUCCESS_TAG = "Episode/average_task_success_rate"
FILENAME_PATTERN = re.compile(r'_e_(\d+)_n_(\d+)_')

# ==============================================================================
# DATA LOADING & PARSING FUNCTIONS
# ==============================================================================
def find_experiments(log_dir):
    """
    Finds all experiment runs and parses hyperparameters from their directory names.
    """
    experiments = []
    print(f"🔍 Scanning for experiments in '{log_dir}'...")
    if not os.path.isdir(log_dir):
        print(f"Error: Directory not found at '{os.path.abspath(log_dir)}'")
        return []
        
    for dir_name in os.listdir(log_dir):
        full_path = os.path.join(log_dir, dir_name)
        if os.path.isdir(full_path):
            match = FILENAME_PATTERN.search(dir_name)
            if match:
                num_envs = int(match.group(1))
                horizon = int(match.group(2))
                experiments.append({
                    'num_envs': num_envs,
                    'horizon': horizon,
                    'run_dir': full_path
                })
                    
    print(f"✅ Found {len(experiments)} valid experiment runs.")
    return experiments

def load_final_point_from_dir(run_dir, tag):
    """
    Loads final success and total run time, correctly handling resumed runs
    by summing the duration of all event files in the directory.
    """
    try:
        event_files = [os.path.join(run_dir, f) for f in os.listdir(run_dir) if 'tfevents' in f]
        if not event_files:
            return np.nan, np.nan
        
        all_events_for_tag = []
        total_duration_sec = 0

        for file_path in event_files:
            ea = event_accumulator.EventAccumulator(file_path, size_guidance={'scalars': 0})
            ea.Reload()
            if tag in ea.Tags()['scalars']:
                events = ea.Scalars(tag)
                if len(events) > 1:
                    all_events_for_tag.extend(events)
                    total_duration_sec += (events[-1].wall_time - events[0].wall_time)
        
        if not all_events_for_tag:
            return np.nan, np.nan
        
        latest_event = max(all_events_for_tag, key=lambda e: e.wall_time)
        final_success = latest_event.value
        
        total_time_hours = total_duration_sec / 3600.0
        
        return total_time_hours, final_success

    except Exception:
        return np.nan, np.nan

def worker_load_data(args_tuple):
    """
    A picklable wrapper that takes experiment parameters, loads the data,
    and returns the parameters along with the result.
    """
    num_envs, horizon, run_dir, tag = args_tuple
    time_h, success = load_final_point_from_dir(run_dir, tag)
    return num_envs, horizon, time_h, success

# ==============================================================================
# PLOTTING FUNCTION
# ==============================================================================
def plot_scaling_curves(processed_data, output_dir="."):
    """
    Generates a line plot of final success rate vs. training time.
    Lines connect points with the same batch size.
    """
    AXIS_LABEL_FONTSIZE = 20
    TICK_LABEL_FONTSIZE = 20
    LEGEND_FONTSIZE = 14
    
    fig, ax = plt.subplots(figsize=(12, 8))

    all_batch_sizes = sorted(list(set(k[0] * k[1] for k in processed_data.keys())))
    all_envs = sorted(list(set(k[0] for k in processed_data.keys())))

    colors = plt.cm.viridis(np.linspace(0, 1, len(all_batch_sizes)))
    color_map = {bs: colors[i] for i, bs in enumerate(all_batch_sizes)}

    markers = ['o', 's', '^', 'X', 'P', 'D', 'v', '<', '>']
    marker_map = {env: markers[i % len(markers)] for i, env in enumerate(all_envs)}
    
    lines_to_plot = defaultdict(list)
    for (num_envs, horizon), (mean_time, mean_success) in processed_data.items():
        batch_size = num_envs * horizon
        lines_to_plot[batch_size].append({
            'time': mean_time,
            'success': mean_success,
            'num_envs': num_envs
        })

    for batch_size, points in sorted(lines_to_plot.items()):
        points.sort(key=lambda p: p['time'])
        if len(points) >= 2:
            times = [p['time'] for p in points]
            successes = [p['success'] for p in points]
            line_color = color_map.get(batch_size)
            ax.plot(times, successes, color=line_color, linestyle=':', linewidth=2, zorder=1)

    for batch_size, points in sorted(lines_to_plot.items()):
        line_color = color_map.get(batch_size)
        for p in points:
            ax.plot(p['time'], p['success'],
                    color=line_color,
                    marker=marker_map.get(p['num_envs'], '*'),
                    linestyle='none',
                    markersize=12,
                    markeredgecolor='white',
                    markeredgewidth=1.5,
                    zorder=2)

    # --- Formatting ---
    ax.set_xscale('log')
    
    tick_locations = [2, 3, 5, 7, 10, 15]
    ax.set_xticks(tick_locations)
    ax.set_xticklabels([str(t) for t in tick_locations])

    ax.set_xlabel("Training time (hours)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Success Rate", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    ax.set_title("")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    legend_elements = [Line2D([0], [0], marker=marker_map[env], color='gray', linestyle='None',
                              markersize=10, label=str(env)) for env in all_envs]
    
    legend = ax.legend(handles=legend_elements, title="Environments", 
                       fontsize=LEGEND_FONTSIZE, loc='best')
    plt.setp(legend.get_title(), fontsize=LEGEND_FONTSIZE)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # <<< MODIFIED: Save both PNG and PDF versions of the plot >>>
    output_filename_png = os.path.join(output_dir, "final_success_rate_vs_time.png")
    plt.savefig(output_filename_png, dpi=300)
    print(f"\n📈 Plot successfully saved as '{output_filename_png}'")

    output_filename_pdf = os.path.join(output_dir, "final_success_rate_vs_time.pdf")
    plt.savefig(output_filename_pdf, bbox_inches='tight')
    print(f"📈 Plot also saved as '{output_filename_pdf}'")

    plt.show()

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    experiments = find_experiments(LOG_DIR)
    OUTPUT_DIR="scripts/batch_scaling/"
    
    tasks = [
        (exp['num_envs'], exp['horizon'], exp['run_dir'], SUCCESS_TAG)
        for exp in experiments
    ]

    print(f"\nLoading final performance from {len(experiments)} runs...")
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap_unordered(worker_load_data, tasks), total=len(tasks), desc="📊 Loading data"))

    raw_data_groups = defaultdict(list)
    for num_envs, horizon, time_h, success in results:
        if not (np.isnan(time_h) or np.isnan(success)):
            key = (num_envs, horizon)
            raw_data_groups[key].append((time_h, success))
            
    processed_data = {}
    print("\nCalculating mean final performance...")
    for key, values_list in raw_data_groups.items():
        if values_list:
            times, successes = zip(*values_list)
            processed_data[key] = (np.mean(times), np.mean(successes))

    if not processed_data:
        print("\nCould not load or process any data. Please check configuration.")
    else:
        table_data = []
        for (num_envs, horizon), (mean_time, mean_success) in processed_data.items():
            table_data.append({
                'num_envs': num_envs,
                'horizon': horizon,
                'PPO_batch_size': num_envs * horizon,
                'mean_training_time_hours': round(mean_time, 2),
                'mean_final_success_rate': round(mean_success, 4)
            })
        
        summary_df = pd.DataFrame(table_data)
        summary_df = summary_df.sort_values(by=['PPO_batch_size', 'num_envs'])
        
        latex_df = summary_df.copy()
        latex_df.rename(columns={
            'num_envs': 'Environments',
            'horizon': 'Horizon',
            'PPO_batch_size': 'Batch Size',
            'mean_training_time_hours': 'Mean Time (h)',
            'mean_final_success_rate': 'Mean Success Rate'
        }, inplace=True)
        
        latex_string = latex_df.to_latex(
            index=False,
            caption='Summary of mean final performance across different scaling configurations.',
            label='tab:scaling_summary',
            position='htbp'
        )
        
        print("\n" + "="*85)
        print("--- LaTeX Summary Table ---")
        print(latex_string)
        print("="*85 + "\n")

        plot_scaling_curves(processed_data, output_dir=OUTPUT_DIR)