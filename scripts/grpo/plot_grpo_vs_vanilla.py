# pip install pandas numpy matplotlib tensorboard PyYAML scipy tqdm
import os
import re
import argparse
from collections import defaultdict
import yaml
import warnings
import traceback

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from scipy.stats import scoreatpercentile
from tqdm import tqdm

# --- Constants for Statistical Analysis ---
BOOTSTRAP_ITERATIONS = 2000 # Number of bootstrap samples for CI calculation
CI_PERCENT = 95.0 # Confidence Interval percentage

# --- Helper Function for Bootstrapped Confidence Interval ---
def bootstrap_confidence_interval(data, metric_func, n_iterations=BOOTSTRAP_ITERATIONS, ci_level=CI_PERCENT):
    """Calculates the bootstrapped confidence interval for a given metric on data."""
    data = np.asarray(data)
    valid_data = data[~np.isnan(data)]
    if valid_data.size < 2:
        return np.nan, np.nan

    bootstrap_stats = []
    for _ in range(n_iterations):
        sample = np.random.choice(valid_data, size=len(valid_data), replace=True)
        bootstrap_stats.append(metric_func(sample))

    alpha = (100.0 - ci_level) / 2.0
    lower_bound = np.percentile(bootstrap_stats, alpha)
    upper_bound = np.percentile(bootstrap_stats, 100.0 - alpha)
    return lower_bound, upper_bound

# --- Helper Function to Adjust Data ---
def adjust_data(df, value_col_name='success'):
    df_copy = df.copy()
    if df_copy.empty or 0 not in df_copy['frame'].values:
        new_row = pd.DataFrame({value_col_name: [0.0], 'frame': [0]})
        df_copy = pd.concat([new_row, df_copy], ignore_index=True)
    df_copy.loc[df_copy['frame'] == 0, value_col_name] = 0.0
    return df_copy.drop_duplicates(subset=['frame'], keep='last')

# --- Helper to Get Final Stat with CI ---
def get_final_stat_ci(list_of_run_dataframes, point_metric_func, value_col_name, data_adjuster_func, target_frame):
    if not list_of_run_dataframes:
        return np.nan, np.nan, np.nan

    final_point_values_across_runs = []
    for df in list_of_run_dataframes:
        df_adj = data_adjuster_func(df.copy(), value_col_name=value_col_name)
        if df_adj.empty:
            final_point_values_across_runs.append(np.nan)
            continue
        
        interp_val = np.interp(target_frame, df_adj['frame'].values, df_adj[value_col_name].values,
                               left=df_adj[value_col_name].iloc[0], right=df_adj[value_col_name].iloc[-1])
        final_point_values_across_runs.append(interp_val)

    valid_final_points = np.array([v for v in final_point_values_across_runs if not np.isnan(v)])
    if valid_final_points.size == 0:
        return np.nan, np.nan, np.nan

    metric_final = point_metric_func(valid_final_points)
    ci_low, ci_high = bootstrap_confidence_interval(valid_final_points, point_metric_func)
    
    error = (ci_high - ci_low) / 2.0
    
    return np.clip(metric_final, 0.0, 1.0), error, len(valid_final_points)

# --- Dynamic Pattern and SETUP_INFO generation ---
def get_patterns_for_setting(setting_str):
    """Defines the regex patterns to identify different experimental runs."""
    patterns = {}
    if setting_str == "mt10":
        patterns['shppo_vanilla'] = rf"05_09_ppo_vanilla_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['shgrpo_vanilla'] = rf"05_26_grpo_vanilla_{setting_str}_rand_(\d+)_seed_(\d+).*"
    elif setting_str == "mt50":
        patterns['shppo_vanilla'] = rf"05_31_ppo_vanilla_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['shgrpo_vanilla'] = rf"05_26_grpo_vanilla_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
    return patterns

def get_setup_info_for_setting(setting_str, current_patterns):
    """Maps method keys to their plot names, patterns, and colors."""
    return {
        'shppo_vanilla': {'pattern': current_patterns.get('shppo_vanilla'), 'color': '#1f77b4', 'name': 'MT-PPO'},
        'shgrpo_vanilla': {'pattern': current_patterns.get('shgrpo_vanilla'), 'color': '#2ca02c', 'name': 'MT-GRPO'},
    }

def extract_setup_info(run_name, setup_info_dict):
    """Identifies which method a given run directory belongs to."""
    for setup_type, info in setup_info_dict.items():
        if info['pattern'] and re.search(info['pattern'], run_name):
            return setup_type
    return None

# --- Plotting Function ---
def create_bar_plot(df, title):
    """Generates a grouped bar plot with the legend inside the plot area."""
    methods = sorted(df['method_name'].unique())
    categories = df['category'].unique()
    colors = df.set_index('method_name')['color'].to_dict()
    
    n_methods = len(methods)
    n_categories = len(categories)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    # Skinnier bars to make room for the legend on the left
    bar_width = 0.25
    index = np.arange(n_categories)

    for i, method_name in enumerate(methods):
        bar_positions = index + (i - (n_methods - 1) / 2) * bar_width
        method_data = df[df['method_name'] == method_name].set_index('category').reindex(categories)
        
        success_rates = method_data['success_rate'].values
        errors = method_data['ci_error'].values

        bars = ax.bar(bar_positions, success_rates, bar_width,
                      label=method_name, color=colors.get(method_name, 'gray'),
                      edgecolor='black', linewidth=1.5)

        ax.errorbar(bar_positions, success_rates, yerr=errors,
                    fmt='none', ecolor='black', capsize=7, capthick=1.5)
        
        for bar, rate, err in zip(bars, success_rates, errors):
            if not np.isnan(rate):
                yval = rate + err if not np.isnan(err) else rate
                ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02,
                        f"{rate:.2f}",
                        ha='center', va='bottom', fontsize=22, color='black')
            
    ax.set_ylabel('Success Rate', fontsize=24)
    ax.set_xticks(index)
    ax.set_xticklabels(categories, fontsize=24)
    ax.tick_params(axis='y', labelsize=22)

    ax.yaxis.grid(True, linestyle=':', alpha=0.7)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Set Y-axis limit to 1.0 (with a small margin)
    ax.set_ylim(0, 1.05)
    
    # Place legend inside the plot area
    ax.legend(fontsize=20,
              loc='upper right', frameon=True, edgecolor='black')
    
    # Adjust layout to prevent labels from being cut off
    fig.tight_layout()
    
    save_dir = "scripts/figures/"
    os.makedirs(save_dir, exist_ok=True)
    save_path_base = os.path.join(save_dir, title.replace(' ', '_'))
    
    plt.savefig(f"{save_path_base}.png", dpi=300)
    plt.savefig(f"{save_path_base}.pdf")
    print(f"\nPlot saved to '{save_dir}' as '{title.replace(' ', '_')}.png' and '.pdf'")
    plt.show()

# --- Main Execution Logic ---
if __name__ == "__main__":
    LOG_DIR_DEFAULT = "./runs/"
    parser = argparse.ArgumentParser(description="Generate summary plots from TensorBoard logs.")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR_DEFAULT, help="Root directory containing TensorBoard run folders.")
    parser.add_argument("--target_frame", type=int, default=None, help="Specific frame to evaluate metrics at. Defaults to the last frame common to all runs.")
    args = parser.parse_args()

    settings_to_process = ["MT10", "MT50"]
    all_plot_data = []

    for setting in settings_to_process:
        print(f"\n--- Processing data for setting: {setting} ---")
        
        patterns = get_patterns_for_setting(setting.lower())
        SETUP_INFO = get_setup_info_for_setting(setting.lower(), patterns)
        
        results_data = defaultdict(list)
        
        if not os.path.isdir(args.log_dir):
            print(f"Error: Log directory '{args.log_dir}' not found. Please check the path.")
            continue

        run_dirs = os.listdir(args.log_dir)
        for run_name in tqdm(run_dirs, desc=f"Scanning {setting} runs", unit="dir"):
            setup_type = extract_setup_info(run_name, SETUP_INFO)
            if not setup_type:
                continue
            
            run_path = os.path.join(args.log_dir, run_name)
            event_file_path = None
            try:
                summaries_path = os.path.join(run_path, "summaries")
                if os.path.isdir(summaries_path):
                    event_files = [f for f in os.listdir(summaries_path) if f.startswith("events.out.tfevents")]
                    if event_files:
                        event_file_path = os.path.join(summaries_path, sorted(event_files)[-1])
            except Exception as e:
                tqdm.write(f"Warning: Could not access {run_name}: {e}")
                continue
            
            if not event_file_path:
                tqdm.write(f"Warning: No event file found for '{run_name}'.")
                continue

            try:
                ea = event_accumulator.EventAccumulator(event_file_path, size_guidance={'scalars': 0})
                ea.Reload()
                
                tag = "Episode/average_task_success_rate"
                if tag in ea.Tags()['scalars']:
                    events = ea.Scalars(tag)
                    config_path = os.path.join(run_path, 'config.yaml')
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    num_envs = config.get('num_envs', 2048)
                    
                    if 'ppo' in run_name.lower():
                        horizon = 32
                    else:
                        horizon = 150
                    
                    df = pd.DataFrame({
                        'success': [e.value for e in events],
                        'frame': [e.step * num_envs * horizon for e in events]
                    })
                    if not df.empty:
                        results_data[setup_type].append(df)
            except Exception as e:
                tqdm.write(f"Error processing '{run_name}': {e}\n{traceback.format_exc()}")
        
        if not results_data:
            print(f"No data found for setting {setting}. Skipping.")
            continue

        effective_target_frame = args.target_frame
        if effective_target_frame is None:
            all_max_frames = []
            for run_dfs in results_data.values():
                for df in run_dfs:
                    if not df.empty:
                        all_max_frames.append(df['frame'].max())
            if all_max_frames:
                effective_target_frame = min(all_max_frames)
            else:
                effective_target_frame = 0
        
        print(f"\nCalculating final statistics for {setting} at frame: {effective_target_frame:,}")
        for setup_type, run_dfs in results_data.items():
            mean, ci_error, run_count = get_final_stat_ci(
                run_dfs,
                np.mean,
                'success',
                adjust_data,
                effective_target_frame
            )
            
            if not np.isnan(mean):
                method_info = SETUP_INFO[setup_type]
                all_plot_data.append({
                    'category': setting,
                    'method_name': method_info['name'],
                    'success_rate': mean,
                    'ci_error': ci_error,
                    'color': method_info['color'],
                    'run_count': run_count
                })

    if not all_plot_data:
        print("\nNo data was successfully processed. Cannot generate plot.")
    else:
        df_final = pd.DataFrame(all_plot_data)
        print("\n--- Final Data for Plotting ---")
        print(df_final)
        create_bar_plot(df_final, title="grpo_vs_vanilla")