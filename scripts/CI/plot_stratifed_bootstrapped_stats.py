# pip install pandas numpy matplotlib tensorboard PyYAML scipy tqdm rliable
import os
from collections import defaultdict
import re
import math # For ceiling function for grid dimensions
import yaml # For reading config files
import warnings # Added for LaTeX table helper
import traceback # For detailed error printing

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import argparse
from scipy.stats import scoreatpercentile
from tqdm import tqdm

# Attempt to import rliable for aggregate plots
try:
    from rliable import plot_utils as rly_plot_utils
    RLIABLE_AVAILABLE = True
except ImportError:
    RLIABLE_AVAILABLE = False
    print("WARNING: `rliable` library not found. Combined aggregate interval plots will be skipped.")
    print("To install: pip install rliable")


# --- Constants for Statistical Analysis ---
BOOTSTRAP_ITERATIONS = 2000 # Number of bootstrap samples for CI calculation
CI_PERCENT = 95.0 # Confidence Interval percentage


# --- Helper Function to Calculate IQM ---
def calculate_iqm(data):
    """Calculates the Interquartile Mean (IQM) of the data."""
    data = np.asarray(data)
    if data.size == 0: # Handle empty data case
        return np.nan
    lower_quartile = scoreatpercentile(data, 25)
    upper_quartile = scoreatpercentile(data, 75)
    if lower_quartile == upper_quartile: # Handles cases where all data in IQR is the same or only one unique value
        # Check if data contains NaNs, median of all NaNs is NaN
        if np.all(np.isnan(data)): return np.nan
        return np.median(data[~np.isnan(data)]) if np.any(~np.isnan(data)) else np.nan


    interquartile_data = data[(data >= lower_quartile) & (data <= upper_quartile)]
    if interquartile_data.size == 0: # Handle cases where IQR is empty (e.g. very few points)
        if np.all(np.isnan(data)): return np.nan
        return np.median(data[~np.isnan(data)]) if np.any(~np.isnan(data)) else np.nan
    return np.mean(interquartile_data[~np.isnan(interquartile_data)]) if np.any(~np.isnan(interquartile_data)) else np.nan


# --- Helper Function for Bootstrapped Confidence Interval (from a distribution of stats) ---
def get_percentile_ci(bootstrap_stats_distribution, ci_level=CI_PERCENT):
    """Calculates CI from a pre-computed distribution of bootstrap statistics."""
    bootstrap_stats_distribution = np.asarray(bootstrap_stats_distribution)
    # Filter out NaNs before calculating percentiles
    valid_stats = bootstrap_stats_distribution[~np.isnan(bootstrap_stats_distribution)]
    if valid_stats.size < 2: # Meaningful CI requires multiple non-NaN points
        return np.nan, np.nan

    alpha = (100.0 - ci_level) / 2.0
    lower_bound = np.percentile(valid_stats, alpha)
    upper_bound = np.percentile(valid_stats, 100.0 - alpha)
    return lower_bound, upper_bound

# --- Helper Function to Generate Comparison String ---
def get_comparison_string(setup_keys):
    if not setup_keys: return "comparison"
    # Ensure SETUP_INFO is accessible, ideally passed as an argument or checked for existence
    if 'SETUP_INFO' not in globals() or not isinstance(SETUP_INFO, dict):
        return "comparison_setup_info_missing"
    names = [SETUP_INFO[key]['name'] for key in setup_keys if key in SETUP_INFO and 'name' in SETUP_INFO[key]]
    if not names: return "comparison_unknown_setups"
    names.sort()
    sanitized_names = [re.sub(r'\W+', '', name) for name in names]
    return "_vs_".join(sanitized_names)


# --- Helper Function to Extract Setup Info from Filename ---
def extract_setup_info(run_name):
    # Ensure SETUP_INFO is accessible
    if 'SETUP_INFO' not in globals() or not isinstance(SETUP_INFO, dict):
        return None, None, None
    for setup_type, info in SETUP_INFO.items():
        match = re.search(info['pattern'], run_name)
        if match:
            num_envs = int(match.group(1))
            seed_number = int(match.group(2))
            return setup_type, num_envs, seed_number
    return None, None, None

# --- Helper Function to Adjust Success Data ---
def adjust_data(df, value_col_name='success'): # Primarily for success rate
    df_copy = df.copy()
    if df_copy.empty: # Handle empty DataFrame input
        return pd.DataFrame({value_col_name: [0.0], 'frame': [0]})

    if 0 not in df_copy['frame'].values:
        # Create a new row for frame 0. Ensure value_col_name exists or handle.
        new_row_data = {value_col_name: [0.0], 'frame': [0]}
        if value_col_name not in df_copy.columns: # If somehow column is missing
            for col in df_copy.columns: # Add other columns as NaN for concat
                if col != 'frame': new_row_data[col] = [np.nan]

        new_row = pd.DataFrame(new_row_data)
        result_df = pd.concat([new_row, df_copy], ignore_index=True)
    elif df_copy.loc[df_copy['frame'] == 0, value_col_name].iloc[0] != 0.0:
        result_df = df_copy.copy() # Ensure modification on a copy
        result_df.loc[result_df['frame'] == 0, value_col_name] = 0.0
    else:
        result_df = df_copy
    
    result_df.drop_duplicates(subset=['frame'], keep='last', inplace=True)
    return result_df

# --- Helper Function to Adjust Generic Scalar Data (like Reward) ---
def adjust_generic_scalar_data(df, value_col_name='value'): # For rewards or other scalars
    df_copy = df.copy()
    if df_copy.empty: # Handle empty DataFrame input
        return pd.DataFrame(columns=[value_col_name, 'frame']) # Return empty with expected columns


    df_copy = df_copy.sort_values('frame').reset_index(drop=True)
    df_copy.drop_duplicates(subset=['frame'], keep='last', inplace=True)
    return df_copy


# --- Helper Function to Calculate Final Metric & CI for LaTeX Table (Stratified Bootstrap Only) ---
def get_final_stat_ci_metric_for_table_stratified(
    task_runs_scores_at_target_frame, # List of np.arrays (scores per task, already at target frame)
    point_metric_func, # IQM, median, or mean to apply WITHIN each task's resample AND to the final bootstrap distribution
    N_original_runs_per_task, # Number of original runs/seeds for each task
    desc_val_name='metric', # For tqdm description, e.g., 'success' or 'reward'
    is_success_metric=False # To apply clipping for success rate (0-1)
    ):
    """
    Calculates the final point metric and its bootstrapped confidence interval using stratified bootstrapping.
    Within each bootstrap iteration:
    1. For each task, a 'task-level statistic' (using point_metric_func) is computed from its resampled scores.
    2. The mean of these 'task-level statistics' is taken as the value for this bootstrap iteration.

    This collection of means is the bootstrap distribution then used to compute the confidence interval
    """
    if N_original_runs_per_task == 0 or not task_runs_scores_at_target_frame:
        return np.nan, np.nan, np.nan

    # This list will store the "mean of task-level statistics" for each bootstrap iteration
    bootstrapped_iteration_values = [] 
    
    for _ in tqdm(range(BOOTSTRAP_ITERATIONS), desc=f"Strat. BS ({desc_val_name})", leave=False, ncols=100, position=1, disable=True):
        # This list will store the point_metric_func result for each task in the current bootstrap iteration
        current_bootstrap_task_level_stats = []
        
        for task_scores_single_task in task_runs_scores_at_target_frame: 
            valid_scores = task_scores_single_task[~np.isnan(task_scores_single_task)]
            if len(valid_scores) == 0:
                # Optionally, append np.nan if a task has no data, or skip.
                # Skipping means the mean below is over fewer task-stats.
                # Appending nan means if any task has no data, the mean_of_task_level_stats might be nan.
                # For robustness, let's ensure we only average valid task stats.
                continue 
            
            resampled_task_scores = np.random.choice(valid_scores, size=N_original_runs_per_task, replace=True)
            # Calculate the statistic (IQM, median, mean) for the current task's resample
            task_level_stat = point_metric_func(resampled_task_scores)
            if not np.isnan(task_level_stat): # Only consider valid task-level stats
                 current_bootstrap_task_level_stats.append(task_level_stat)
        
        if current_bootstrap_task_level_stats: # If at least one task provided a valid statistic
            # Calculate the mean of these task-level statistics for this bootstrap iteration
            mean_of_task_level_stats_this_iteration = np.mean(current_bootstrap_task_level_stats)
            
            if is_success_metric:
                mean_of_task_level_stats_this_iteration = np.clip(mean_of_task_level_stats_this_iteration, 0.0, 1.0)
            bootstrapped_iteration_values.append(mean_of_task_level_stats_this_iteration)

    valid_bootstrapped_iteration_values = [m for m in bootstrapped_iteration_values if not np.isnan(m)]
    if not valid_bootstrapped_iteration_values:
        return np.nan, np.nan, np.nan

    # The final point_metric_func is applied to the distribution of 
    # "mean of task-level statistics"
    metric_final = point_metric_func(np.array(valid_bootstrapped_iteration_values))
    
    if len(valid_bootstrapped_iteration_values) < 2: # Not enough data for a CI
        return metric_final, np.nan, np.nan
            
    ci_low, ci_high = get_percentile_ci(np.array(valid_bootstrapped_iteration_values), CI_PERCENT)

    if is_success_metric: # Clipping applies to success-like metrics
        if not np.isnan(metric_final): metric_final = np.clip(metric_final, 0.0, 1.0)
        if not np.isnan(ci_low): ci_low = np.clip(ci_low, 0.0, 1.0)
        if not np.isnan(ci_high): ci_high = np.clip(ci_high, 0.0, 1.0)
        
    return metric_final, ci_low, ci_high

# --- LaTeX Table Generation Function ---
def print_latex_summary_table(
    all_setups_data,
    setup_run_counts,
    stat_func, 
    stat_name_short, 
    stat_name_display
    ):
    print(f"\n\n--- LaTeX Summary Table ({stat_name_display} & CI, Stratified Overall Success & Reward if per-task data available) ---")
    header = [
        "Setup Name", "Envs~(K)", "Seeds",
        f"Final Overall Success (%) ({stat_name_display} [{CI_PERCENT:.0f}\% CI])",
        f"Avg Reward ({stat_name_display} [{CI_PERCENT:.0f}\% CI])"
    ]
    latex_table_rows = []
    numerical_summary_list = [] 

    global SETUP_INFO, NUM_TASKS, HORIZON # Declare use of global for clarity

    gsf_s, mfps_s, upsmf_s, haptd_s = get_target_frame_for_metric('per_task_success', 'overall_avg', 'success', adjust_data, all_setups_data, setup_run_counts)
    gsf_r, mfps_r, upsmf_r, haptd_r = get_target_frame_for_metric('per_task_reward', 'avg_reward', 'value', adjust_generic_scalar_data, all_setups_data, setup_run_counts)
    
    stratified_success_possible_globally = haptd_s
    stratified_reward_possible_globally = haptd_r


    for setup_type, num_runs_for_setup in tqdm(setup_run_counts.items(), desc=f"Generating Table Rows ({stat_name_display})", ncols=100, position=0, disable=False):
        setup_config = SETUP_INFO.get(setup_type, {})
        name = setup_config.get('name', setup_type); required_envs = setup_config.get('required_envs')
        env_k_str = "N/A";
        if required_envs is not None:
            try: env_k_str = f"{int(required_envs) // 1024}k"
            except: env_k_str = "Err"

        current_summary_entry = {
            'setup_type': setup_type, 'name': name, 'env_k_str': env_k_str, 'num_runs': num_runs_for_setup,
            f'success_{stat_name_short}': np.nan, f'success_ci_low_{stat_name_short}': np.nan, f'success_ci_high_{stat_name_short}': np.nan,
            'success_str': "No Per-Task Data", 
            f'reward_{stat_name_short}': np.nan, f'reward_ci_low_{stat_name_short}': np.nan, f'reward_ci_high_{stat_name_short}': np.nan,
            'reward_str': "No Per-Task Data"
        }
        
        final_overall_success_str_temp = "No Per-Task Data" 
        N_original_runs_success = 0; attempt_stratified_success = False
        if stratified_success_possible_globally and 'per_task_success' in all_setups_data.get(setup_type, {}):
            per_task_success_data_for_setup = all_setups_data[setup_type]['per_task_success']
            if any(per_task_success_data_for_setup.get(tid) for tid in range(NUM_TASKS)): 
                for tid_check in range(NUM_TASKS): 
                    task_data_for_check = per_task_success_data_for_setup.get(tid_check, [])
                    if task_data_for_check: # Check if list is not empty
                        N_original_runs_success = len(task_data_for_check) # Number of runs for this task
                        if N_original_runs_success > 0: attempt_stratified_success = True; break
        
        target_frame_s_current_setup = gsf_s if not upsmf_s else mfps_s.get(setup_type, gsf_s)

        if attempt_stratified_success and N_original_runs_success > 0 :
            per_task_s_at_final = []; valid_tasks_s = 0 
            for task_num in range(NUM_TASKS):
                run_data_list = all_setups_data[setup_type]['per_task_success'].get(task_num, [])
                adj_dfs = [adjust_data(df.copy(), 'success') for df in run_data_list if isinstance(df, pd.DataFrame) and not df.empty]
                
                # Scores for this task across all its original runs, at the target frame
                task_s_scores_for_this_task = np.full((N_original_runs_success,), np.nan) 
                
                for seed_idx, df_s_task in enumerate(adj_dfs):
                    if seed_idx >= N_original_runs_success: continue 
                    if df_s_task.empty: continue
                    left_interp = df_s_task['success'].iloc[0] if 0 in df_s_task['frame'].values else np.nan
                    right_interp = df_s_task['success'].iloc[-1]
                    interp_val = np.interp(target_frame_s_current_setup, df_s_task['frame'].values, df_s_task['success'].values,
                                           left=left_interp, right=right_interp)
                    task_s_scores_for_this_task[seed_idx] = interp_val
                
                if not np.all(np.isnan(task_s_scores_for_this_task)): 
                    per_task_s_at_final.append(task_s_scores_for_this_task); valid_tasks_s +=1
            
            if valid_tasks_s > 0: 
                metric_s, ci_l_s, ci_h_s = get_final_stat_ci_metric_for_table_stratified(
                    task_runs_scores_at_target_frame=per_task_s_at_final,
                    point_metric_func=stat_func,
                    N_original_runs_per_task=N_original_runs_success,
                    desc_val_name='success',
                    is_success_metric=True
                )
                if not np.isnan(metric_s):
                    current_summary_entry.update({
                        f'success_{stat_name_short}': metric_s * 100, 
                        f'success_ci_low_{stat_name_short}': ci_l_s * 100 if not np.isnan(ci_l_s) else np.nan, 
                        f'success_ci_high_{stat_name_short}': ci_h_s * 100 if not np.isnan(ci_h_s) else np.nan
                    })
                else: final_overall_success_str_temp = "N/A (calc error)" # Should not happen if valid_tasks_s > 0
        # else: Non-stratified path removed. If no per-task data, it remains "No Per-Task Data".
        
        if not np.isnan(current_summary_entry[f'success_{stat_name_short}']): 
            s_metric_val = current_summary_entry[f'success_{stat_name_short}']
            s_cil = current_summary_entry[f'success_ci_low_{stat_name_short}']
            s_cih = current_summary_entry[f'success_ci_high_{stat_name_short}']
            if np.isnan(s_cil) or np.isnan(s_cih):
                current_summary_entry['success_str'] = f"${s_metric_val:.2f}$"
            else:
                current_summary_entry['success_str'] = f"${s_metric_val:.2f} [{s_cil:.2f} - {s_cih:.2f}]$"
        else: current_summary_entry['success_str'] = final_overall_success_str_temp


        final_avg_reward_str_temp = "No Per-Task Data"
        N_original_runs_reward = 0; attempt_stratified_reward = False
        if stratified_reward_possible_globally and 'per_task_reward' in all_setups_data.get(setup_type, {}):
            per_task_reward_data_for_setup = all_setups_data[setup_type]['per_task_reward']
            if any(per_task_reward_data_for_setup.get(tid) for tid in range(NUM_TASKS)):
                for tid_check in range(NUM_TASKS):
                    task_data_for_check_r = per_task_reward_data_for_setup.get(tid_check, [])
                    if task_data_for_check_r:
                        N_original_runs_reward = len(task_data_for_check_r)
                        if N_original_runs_reward > 0: attempt_stratified_reward = True; break
        
        target_frame_r_current_setup = gsf_r if not upsmf_r else mfps_r.get(setup_type, gsf_r)

        if attempt_stratified_reward and N_original_runs_reward > 0:
            per_task_r_at_final = []; valid_tasks_r = 0
            for task_num in range(NUM_TASKS):
                run_data_list = all_setups_data[setup_type]['per_task_reward'].get(task_num, [])
                adj_dfs = [adjust_generic_scalar_data(df.copy(), 'value') for df in run_data_list if isinstance(df, pd.DataFrame) and not df.empty]
                task_r_scores_for_this_task = np.full((N_original_runs_reward,), np.nan)

                for seed_idx, df_r_task in enumerate(adj_dfs):
                    if seed_idx >= N_original_runs_reward: continue
                    if df_r_task.empty: continue
                    
                    df_frames = df_r_task['frame'].values
                    df_values = df_r_task['value'].values
                    left_val_interp = df_values[0] if len(df_values)>0 else np.nan
                    right_val_interp = df_values[-1] if len(df_values)>0 else np.nan

                    interp_val = np.interp(target_frame_r_current_setup, df_frames, df_values,
                                           left=left_val_interp, right=right_val_interp)
                    task_r_scores_for_this_task[seed_idx] = interp_val
                if not np.all(np.isnan(task_r_scores_for_this_task)):
                    per_task_r_at_final.append(task_r_scores_for_this_task); valid_tasks_r +=1
            
            if valid_tasks_r > 0:
                metric_r, ci_l_r, ci_h_r = get_final_stat_ci_metric_for_table_stratified(
                    task_runs_scores_at_target_frame=per_task_r_at_final,
                    point_metric_func=stat_func,
                    N_original_runs_per_task=N_original_runs_reward,
                    desc_val_name='reward',
                    is_success_metric=False
                )
                if not np.isnan(metric_r):
                    current_summary_entry.update({
                        f'reward_{stat_name_short}': metric_r, 
                        f'reward_ci_low_{stat_name_short}': ci_l_r if not np.isnan(ci_l_r) else np.nan, 
                        f'reward_ci_high_{stat_name_short}': ci_h_r if not np.isnan(ci_h_r) else np.nan
                    })
                else: final_avg_reward_str_temp = "N/A (calc error)"
        # else: Non-stratified path removed.

        if not np.isnan(current_summary_entry[f'reward_{stat_name_short}']): 
            r_metric_val = current_summary_entry[f'reward_{stat_name_short}']
            r_cil = current_summary_entry[f'reward_ci_low_{stat_name_short}']
            r_cih = current_summary_entry[f'reward_ci_high_{stat_name_short}']
            if np.isnan(r_cil) or np.isnan(r_cih):
                current_summary_entry['reward_str'] = f"${r_metric_val:.2f}$"
            else:
                current_summary_entry['reward_str'] = f"${r_metric_val:.2f} [{r_cil:.2f} - {r_cih:.2f}]$"
        else: current_summary_entry['reward_str'] = final_avg_reward_str_temp
        
        numerical_summary_list.append(current_summary_entry)
        latex_table_rows.append([name, env_k_str, str(num_runs_for_setup), current_summary_entry['success_str'], current_summary_entry['reward_str']])

    print("\\begin{table}[H] % Requires 'float' package")
    print("\\centering")
    caption_note = f"Frames = step * num\\_envs * {HORIZON}."
    common_frame_s_text = ""
    if not upsmf_s and gsf_s > 0 : common_frame_s_text = f"$\\approx {gsf_s:,.0f}$"
    elif upsmf_s : common_frame_s_text = "per-setup max" 
    else : common_frame_s_text = "max available" 
    if common_frame_s_text: caption_note += f" Final Success at {common_frame_s_text} frames."
    
    common_frame_r_text = ""
    if not upsmf_r and gsf_r > 0 : common_frame_r_text = f"$\\approx {gsf_r:,.0f}$"
    elif upsmf_r : common_frame_r_text = "per-setup max" 
    else : common_frame_r_text = "max available" 
    if common_frame_r_text: caption_note += f" Avg Reward at {common_frame_r_text} frames."
    
    print(f"\\caption{{Algorithm Performance Summary ({stat_name_display} [{CI_PERCENT:.0f}\% CI]). {caption_note}}}")
    print(f"\\label{{tab:perf_summary_{stat_name_short}_stratified}}") 
    col_format = "l r c c c"; print(f"\\begin{{tabular}}{{@{{}} {col_format} @{{}}}}"); print("\\toprule")
    print(" & ".join(header) + " \\\\"); print("\\midrule")
    for row_idx, row_content in enumerate(latex_table_rows):
        processed_row = [str(item).replace("_", "\\_") if not (str(item).startswith("$") and str(item).endswith("$")) else str(item) for item in row_content]
        print(" & ".join(processed_row) + (" \\\\" if row_idx < len(latex_table_rows) -1 else " \\\\ \\bottomrule"))
    print("\\end{tabular}"); print("\\end{table}")
    print(f"\n% Note: Values are '{stat_name_display} [{CI_PERCENT:.0f}% CI]'. Stratified bootstrap (mean of pooled resamples per iteration) for Overall metrics if per-task data available.")
    print("% If per-task data is unavailable, 'No Per-Task Data' is shown.")
    print("% Target frame for metrics is the minimum of per-setup maximum frames, or per-setup max if global minimum is 0, or max available if no common frame.")

    return numerical_summary_list

# Helper function to determine target frames, needs to be defined before use in print_latex_summary_table
def get_target_frame_for_metric(metric_key_per_task, metric_key_aggregate, value_col, adjuster_func, all_setups_data_local, setup_run_counts_local):
    target_frame = 0; max_frames_per_setup_metric = {}; has_any_per_task_data_for_metric = False
    all_max_frames_across_setups_for_metric = []
    global NUM_TASKS 

    for setup_type_iter, num_runs_iter in setup_run_counts_local.items():
        if num_runs_iter == 0 or setup_type_iter not in all_setups_data_local: continue
        current_setup_max_frame_metric = 0; setup_has_per_task_data = False
        
        setup_data = all_setups_data_local.get(setup_type_iter, {})
        if metric_key_per_task and metric_key_per_task in setup_data:
            per_task_data_for_setup = setup_data[metric_key_per_task]
            if isinstance(per_task_data_for_setup, dict) and any(per_task_data_for_setup.get(tid) for tid in range(NUM_TASKS)): 
                setup_has_per_task_data = True; has_any_per_task_data_for_metric = True
                for task_num in range(NUM_TASKS):
                    run_data_list = per_task_data_for_setup.get(task_num, [])
                    adjusted_dfs = [adjuster_func(df.copy(), value_col) for df in run_data_list if isinstance(df, pd.DataFrame) and not df.empty]
                    for df_adj in adjusted_dfs:
                        if not df_adj.empty and 'frame' in df_adj.columns: 
                            current_setup_max_frame_metric = max(current_setup_max_frame_metric, df_adj['frame'].max())
        
        # Fallback to aggregate tags if per-task is not available for this setup for frame determination
        if not setup_has_per_task_data and metric_key_aggregate and metric_key_aggregate in setup_data:
            aggregate_data_list = setup_data[metric_key_aggregate]
            adjusted_dfs_agg = [adjuster_func(df.copy(), value_col) for df in aggregate_data_list if isinstance(df, pd.DataFrame) and not df.empty]
            for df_agg in adjusted_dfs_agg:
                if not df_agg.empty and 'frame' in df_agg.columns:
                    current_setup_max_frame_metric = max(current_setup_max_frame_metric, df_agg['frame'].max())
        
        max_frames_per_setup_metric[setup_type_iter] = current_setup_max_frame_metric
        if current_setup_max_frame_metric > 0 :
            all_max_frames_across_setups_for_metric.append(current_setup_max_frame_metric)

    if all_max_frames_across_setups_for_metric:
        target_frame = min(all_max_frames_across_setups_for_metric) 
    else: target_frame = 0 
        
    use_per_setup_max = (target_frame == 0 and any(val > 0 for val in max_frames_per_setup_metric.values()))
    # has_any_per_task_data_for_metric indicates if *any* setup has per-task data for this metric type (e.g. success)
    return target_frame, max_frames_per_setup_metric, use_per_setup_max, has_any_per_task_data_for_metric

# --- Main Execution Logic ---
if __name__ == "__main__":
    LOG_DIR_DEFAULT = "/work/08962/vjoshi12/ls6/IsaacGymEnvs/runs/" 
    if not os.path.exists(LOG_DIR_DEFAULT): 
        LOG_DIR_DEFAULT = "./runs_default/" 
        print(f"Warning: Default log directory not found, using {LOG_DIR_DEFAULT}")
        os.makedirs(LOG_DIR_DEFAULT, exist_ok=True)

    parser = argparse.ArgumentParser(description="Generate summary table and bar charts from TensorBoard logs using IQM and CIs (Stratified for Overall metrics).")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR_DEFAULT, help="Root directory containing TensorBoard run folders.")
    parser.add_argument("--is_mt10", action="store_true", help="MT10 specific setup (NUM_TASKS=10).")
    parser.add_argument("--horizon", type=int, default=32, help="Horizon value for frame calculation.")
    args = parser.parse_args()

    LOG_DIR = args.log_dir
    HORIZON = args.horizon; print(f"Args: {args}")
    if args.is_mt10: setting = "mt10"; NUM_TASKS = 10
    else: setting = "mt50"; NUM_TASKS = 50
    print(f"Using setting: {setting}, Number of tasks: {NUM_TASKS}, Horizon: {HORIZON}, Log Dir: {LOG_DIR}")

    if args.is_mt10:
        PATTERN_SHPPO_VANILLA = rf"05_09_ppo_vanilla_{setting}_rand_envs_(\d+)_seed_(\d+).*"
        PATTERN_MHPPO_VANILLA = rf"mhppo_vanilla_{setting}_rand_envs_(\d+)_seed_(\d+).*"
        PATTERN_FAMO = rf"05_09_ppo_famo_{setting}_rand_envs_(\d+)_seed_(\d+).*"
        PATTERN_PCGRAD = rf"05_11_ppo_pcgrad_{setting}_rand_envs_(\d+)_seed_(\d+).*"
        PATTERN_CAGRAD = rf"ppo_cagrad_{setting}_rand_envs_(\d+)_seed_(\d+).*"
        PATTERN_SHPPO_PACO = rf"shppo_paco_{setting}_rand_envs_(\d+)_seed_(\d+).*"
        PATTERN_MHPPO_MOORE = rf"mhppo_moore_{setting}_rand_envs_(\d+)_seed_(\d+).*"
        PATTERN_SHPPO_MOORE = rf"shppo_moore_{setting}_rand_envs_(\d+)_seed_(\d+).*"
        PATTERN_SOFT_MODULARIZATION = rf"ppo_soft_modularization_{setting}_rand_envs_(\d+)_seed_(\d+).*"
        PATTERN_MHPPO_CARE = rf"05_11_mhppo_care_{setting}_rand_envs_(\d+)_seed_(\d+).*"
        PATTERN_SHPPO_CARE = rf"shppo_care_{setting}_rand_envs_(\d+)_seed_(\d+).*"
    else:
        PATTERN_SHPPO_VANILLA = rf"^ppo_vanilla_{setting}_rand_envs_(\d+)_seed_(\d+).*"
        PATTERN_MHPPO_VANILLA = rf"05_07_mhppo_vanilla_{setting}_rand_envs_(\d+)_seed_(\d+).*"
        PATTERN_FAMO = rf"ppo_famo_{setting}_rand_envs_(\d+)_seed_(\d+).*"
        PATTERN_PCGRAD = rf"ppo_pcgrad_{setting}_rand_envs_(\d+)_seed_(\d+).*"
        PATTERN_CAGRAD = rf"ppo_cagrad_{setting}_rand_envs_(\d+)_seed_(\d+).*"
        PATTERN_SHPPO_PACO = rf"shppo_paco_{setting}_rand_envs_(\d+)_seed_(\d+).*"
        PATTERN_MHPPO_MOORE = rf"mhppo_moore_{setting}_rand_envs_(\d+)_seed_(\d+).*"
        PATTERN_SHPPO_MOORE = rf"shppo_moore_{setting}_rand_envs_(\d+)_seed_(\d+).*"
        PATTERN_MHPPO_CARE = rf"mhppo_care_{setting}_rand_envs_(\d+)_seed_(\d+).*"
        PATTERN_SHPPO_CARE = rf"shppo_care_{setting}_rand_envs_(\d+)_seed_(\d+).*"
        PATTERN_SOFT_MODULARIZATION = rf"ppo_soft_modularization_{setting}_rand_envs_(\d+)_seed_(\d+).*"

    SETUP_INFO = {
        ### VANILLA ###
        'shppo_vanilla': {'pattern': PATTERN_SHPPO_VANILLA, 'color': '#1A85FF', 'name': 'SH-Vanilla', 'required_envs': 24576},
        'mhppo_vanilla': {'pattern': PATTERN_MHPPO_VANILLA, 'color': '#2ca02c', 'name': 'MH-Vanilla', 'required_envs': 24576},
        ### Grad Mani ###
        'pcgrad': {'pattern': PATTERN_PCGRAD, 'color': '#ff7f0e', 'name': 'PCGrad', 'required_envs': 24576},
        'cagrad': {'pattern': PATTERN_CAGRAD, 'color': '#1f77b4', 'name': 'CAGrad', 'required_envs': 24576},
        'famo': {'pattern': PATTERN_FAMO, 'color': '#3690ff', 'name': 'FAMO', 'required_envs': 24576},
        ### Neural Architecture ###
        'soft_modularization': {'pattern': PATTERN_SOFT_MODULARIZATION, 'color': '#FF4081', 'name': 'Soft-Modularization', 'required_envs': 24576},
        'shppo_care': {'pattern': PATTERN_SHPPO_CARE, 'color': '#FF5722', 'name': 'SH-CARE', 'required_envs': 24576},
        'mhppo_care': {'pattern': PATTERN_MHPPO_CARE, 'color': '#FF9800', 'name': 'MH-CARE', 'required_envs': 24576},
        'shppo_paco': {'pattern': PATTERN_SHPPO_PACO, 'color': '#E91E63', 'name': 'SH-PaCo', 'required_envs': 24576},
        'shppo_moore': {'pattern': PATTERN_SHPPO_MOORE, 'color': '#FF4081', 'name': 'SH-MOORE', 'required_envs': 24576}, 
        'mhppo_moore': {'pattern': PATTERN_MHPPO_MOORE, 'color': '#F50057', 'name': 'MH-MOORE', 'required_envs': 24576}, 
    }
    
    CONFIG_FILENAME = "config.yaml"
    results_data = defaultdict(lambda: {
        'per_task_success': defaultdict(list), 'per_task_reward': defaultdict(list),
        'overall_avg': list(), 'avg_reward': list() # overall_avg and avg_reward might be less used if only stratified is primary
    })
    setup_run_counts = defaultdict(int)

    if not os.path.isdir(LOG_DIR): print(f"Error: Log directory '{LOG_DIR}' not found."); exit()
    
    directory_contents = os.listdir(LOG_DIR)
    directories_to_process = [name for name in directory_contents if os.path.isdir(os.path.join(LOG_DIR, name))]
    print(f"Found {len(directories_to_process)} potential run directories to scan in '{LOG_DIR}'.")
    
    processed_runs_total = 0; skipped_runs_count = 0
    for item_name in tqdm(directories_to_process, desc="Scanning Log Dirs", unit="dir", ncols=100, position=0):
        item_path = os.path.join(LOG_DIR, item_name)
        setup_type, num_envs, seed_number = extract_setup_info(item_name)
        if setup_type is None: skipped_runs_count +=1; continue
        
        setup_config_main = SETUP_INFO.get(setup_type) 
        if not setup_config_main: skipped_runs_count +=1; continue

        required_env_count = setup_config_main.get('required_envs')
        if required_env_count is not None and num_envs != required_env_count : 
            tqdm.write(f"Skipping {item_name}: Env count mismatch (found {num_envs}, need {required_env_count}).")
            skipped_runs_count +=1; continue
        
        config_path = os.path.join(item_path, CONFIG_FILENAME)
        task_ids_from_config = [] 
        if setting == "mt10":
            task_ids_from_config = [4,16,17,18,28,31,38,40,48,49] 
        else: 
            task_ids_from_config = list(range(NUM_TASKS)) 

        if os.path.isfile(config_path):
            try:
                with open(config_path, "r") as f: config_yaml = yaml.safe_load(f)
                loaded_task_ids_raw = config_yaml.get('task_id') 
                if loaded_task_ids_raw is not None: 
                    if isinstance(loaded_task_ids_raw, list) and all(isinstance(tid, int) for tid in loaded_task_ids_raw):
                        if NUM_TASKS > 0 and not loaded_task_ids_raw: 
                            tqdm.write(f" W: Empty task_ids in '{config_path}' for {item_name} (NUM_TASKS={NUM_TASKS}). Using default or skipping if NUM_TASKS=0.");
                            if NUM_TASKS == 0: task_ids_from_config = [] 
                        else:
                            task_ids_from_config = loaded_task_ids_raw
                    elif NUM_TASKS > 0 : 
                        tqdm.write(f" W: Invalid task_ids format in '{config_path}' for {item_name}. Using default task IDs.");
            except Exception as e: tqdm.write(f" E: reading config '{config_path}': {e}. Using default task IDs.");
        elif NUM_TASKS > 0 and not task_ids_from_config and setting !="mt10": 
            tqdm.write(f" W: Config missing for {item_name} and NUM_TASKS={NUM_TASKS}. Defaulting to range(NUM_TASKS).");
            task_ids_from_config = list(range(NUM_TASKS))

        event_file_path = None
        for p_dir_opt in [os.path.join(item_path, "summaries"), item_path]: 
            if os.path.isdir(p_dir_opt):
                try:
                    event_files = [f for f in os.listdir(p_dir_opt) if f.startswith("events.out.tfevents")]
                    if event_files: event_file_path = os.path.join(p_dir_opt, sorted(event_files)[-1]); break 
                except Exception as e: tqdm.write(f" W: Accessing '{p_dir_opt}' for {item_name}: {e}.")
        if event_file_path is None: tqdm.write(f" W: No event file for '{item_name}'. Skipping."); skipped_runs_count += 1; continue

        run_data_loaded_flag = False
        try:
            ea = event_accumulator.EventAccumulator(event_file_path, size_guidance={'scalars': 0}, purge_orphaned_data=True)
            ea.Reload(); all_scalar_tags = ea.Tags().get("scalars", [])
        except Exception as e:
            tqdm.write(f" E: Loading EventAccumulator for '{item_name}': {e}. Skipping run.");
            skipped_runs_count +=1; continue
            
        if not all_scalar_tags: tqdm.write(f" W: No scalars in event file for {item_name}."); skipped_runs_count +=1; continue
        
        # Load per-task data for stratified bootstrapping
        if NUM_TASKS > 0 and task_ids_from_config: 
            for task_num_idx, actual_task_id in enumerate(task_ids_from_config): 
                s_tag = f"Episode/task_{actual_task_id}_success"
                if s_tag in all_scalar_tags:
                    scalar_events_s = ea.Scalars(s_tag)
                    if scalar_events_s:
                        df_s = pd.DataFrame({'success': [x.value for x in scalar_events_s], 'frame': [x.step * num_envs * HORIZON for x in scalar_events_s]})
                        if not df_s.empty: 
                            results_data[setup_type]['per_task_success'][task_num_idx].append(df_s); run_data_loaded_flag = True
                
                r_tag = f"Episode/task_{actual_task_id}_reward"
                if r_tag in all_scalar_tags:
                    scalar_events_r = ea.Scalars(r_tag)
                    if scalar_events_r:
                        df_r = pd.DataFrame({'value': [x.value for x in scalar_events_r], 'frame': [x.step * num_envs * HORIZON for x in scalar_events_r]})
                        if not df_r.empty: 
                            results_data[setup_type]['per_task_reward'][task_num_idx].append(df_r); run_data_loaded_flag = True
        
        # Still load aggregate tags if needed for other purposes or simple checks,
        # but they won't be used for the primary CI calculation in the table if per-task is available.
        overall_s_tag = "Episode/average_task_success_rate" 
        if overall_s_tag in all_scalar_tags:
            scalar_events = ea.Scalars(overall_s_tag)
            if scalar_events:
                df_overall_s = pd.DataFrame({'success': [x.value for x in scalar_events], 'frame': [x.step * num_envs * HORIZON for x in scalar_events]})
                if not df_overall_s.empty:
                    results_data[setup_type]['overall_avg'].append(df_overall_s) 
                    # run_data_loaded_flag might already be true from per-task, this is fine.
                    if not run_data_loaded_flag : run_data_loaded_flag = True


        agg_r_tag = "rewards/step" 
        if agg_r_tag in all_scalar_tags:
            scalar_events_agg_r = ea.Scalars(agg_r_tag)
            if scalar_events_agg_r:
                df_agg_r = pd.DataFrame({'value': [x.value for x in scalar_events_agg_r], 'frame': [x.step * num_envs * HORIZON for x in scalar_events_agg_r]})
                if not df_agg_r.empty: 
                    results_data[setup_type]['avg_reward'].append(df_agg_r)
                    if not run_data_loaded_flag : run_data_loaded_flag = True
        
        if run_data_loaded_flag:
            processed_runs_total += 1; setup_run_counts[setup_type] += 1
        else: tqdm.write(f" W: No primary (per-task or aggregate) data extracted from '{item_name}'."); skipped_runs_count += 1
    
    print("\n--- Processing Summary ---")
    if processed_runs_total == 0 :
        print("No valid data loaded from any run. Exiting.")
        exit()

    print(f"Successfully processed data from {processed_runs_total} runs.")
    if skipped_runs_count > 0: print(f"Skipped {skipped_runs_count} directories/runs due to config/setup/data issues.")
    for st, count in setup_run_counts.items():
        name = SETUP_INFO.get(st, {}).get('name', st); envs = SETUP_INFO.get(st, {}).get('required_envs', "N/A")
        print(f"       Setup '{name}' (Envs: {envs}): {count} runs processed.")
            
    STATISTICS_TO_REPORT = [
        {"func": calculate_iqm, "short_name": "iqm", "display_name": "IQM"},
        {"func": np.median, "short_name": "median", "display_name": "Median"},
        {"func": np.mean, "short_name": "mean", "display_name": "Mean"},
    ]
    STAT_ORDER = [s['short_name'] for s in STATISTICS_TO_REPORT]
    METRIC_NAMES_DISPLAY_RLIABLE = [s['display_name'] for s in STATISTICS_TO_REPORT]

    master_numerical_summary = {} 
    for stat_info in STATISTICS_TO_REPORT:
        numerical_summary_for_stat = print_latex_summary_table(
            results_data, 
            setup_run_counts,
            stat_func=stat_info["func"],
            stat_name_short=stat_info["short_name"],
            stat_name_display=stat_info["display_name"]
        )
        master_numerical_summary[stat_info["short_name"]] = numerical_summary_for_stat

    if RLIABLE_AVAILABLE:
        print("\n--- Generating Combined Aggregate Interval Plots (using rliable) ---")

        ordered_setup_types_for_plot = [
            stype for stype in SETUP_INFO.keys()
            if stype in setup_run_counts and setup_run_counts[stype] > 0
        ]

        if not ordered_setup_types_for_plot:
            print("No algorithms found with processed data for rliable plots. Skipping.")
        else:
            vanilla_display_names = {
                SETUP_INFO[stype]['name'] for stype in SETUP_INFO
                if 'vanilla' in stype.lower() and 'name' in SETUP_INFO[stype]
            }
            grad_manip_display_names = {
                SETUP_INFO[stype]['name'] for stype in SETUP_INFO
                if stype in ['famo', 'pcgrad', 'cagrad'] and 'name' in SETUP_INFO[stype]
            }

            # SUCCESS Metric
            valid_algos_success = []
            for setup_type in ordered_setup_types_for_plot:
                algo_display_name = SETUP_INFO[setup_type]['name']
                has_any_success_data_for_algo = False
                for stat_short_name in STAT_ORDER:
                    if stat_short_name in master_numerical_summary and master_numerical_summary[stat_short_name]:
                        for summary_item in master_numerical_summary[stat_short_name]:
                            if summary_item['setup_type'] == setup_type:
                                if not np.isnan(summary_item.get(f'success_{stat_short_name}', np.nan)):
                                    has_any_success_data_for_algo = True
                                    break
                    if has_any_success_data_for_algo: break
                if has_any_success_data_for_algo:
                    valid_algos_success.append(algo_display_name)
            
            if not valid_algos_success:
                print("No algorithms with plottable success data for rliable plots. Skipping success plot.")
            else:
                aggregate_scores_success = {}
                aggregate_score_cis_success = {}
                for algo_display_name in valid_algos_success:
                    current_setup_type = next(stype for stype, sinfo in SETUP_INFO.items() if sinfo['name'] == algo_display_name)
                    success_points, success_ci_lows, success_ci_highs = [], [], []
                    for stat_short_name in STAT_ORDER:
                        point_val, low_val, high_val = np.nan, np.nan, np.nan
                        if stat_short_name in master_numerical_summary and master_numerical_summary[stat_short_name]:
                            for summary_item in master_numerical_summary[stat_short_name]:
                                if summary_item['setup_type'] == current_setup_type:
                                    point_val = summary_item.get(f'success_{stat_short_name}', np.nan)
                                    low_val = summary_item.get(f'success_ci_low_{stat_short_name}', np.nan)
                                    high_val = summary_item.get(f'success_ci_high_{stat_short_name}', np.nan)
                                    break
                        success_points.append(point_val)
                        success_ci_lows.append(low_val)
                        success_ci_highs.append(high_val)
                    aggregate_scores_success[algo_display_name] = np.array(success_points)
                    aggregate_score_cis_success[algo_display_name] = np.array([success_ci_lows, success_ci_highs])

                if aggregate_scores_success:
                    algos_for_plot_s = valid_algos_success[::-1]
                    scores_for_plot_s = {name: aggregate_scores_success[name] for name in algos_for_plot_s}
                    cis_for_plot_s = {name: aggregate_score_cis_success[name] for name in algos_for_plot_s}
                    try:
                        fig_agg_s, axes_agg_s_array = rly_plot_utils.plot_interval_estimates(
                            scores_for_plot_s, cis_for_plot_s,
                            metric_names=METRIC_NAMES_DISPLAY_RLIABLE,
                            algorithms=algos_for_plot_s, 
                            xlabel_y_coordinate=0, xlabel='Success Rate (%)',
                            max_ticks_y=max(5, len(algos_for_plot_s))
                        )
                        plt.subplots_adjust(left=0.25, bottom=0.2)
                        plotted_algorithms_list_s = valid_algos_success 
                        num_algos_s = len(plotted_algorithms_list_s)
                        last_vanilla_idx_s = -1
                        for i, algo_name in enumerate(plotted_algorithms_list_s):
                            if algo_name in vanilla_display_names: last_vanilla_idx_s = i
                        last_grad_manip_idx_s = -1
                        for i, algo_name in enumerate(plotted_algorithms_list_s):
                            if algo_name in grad_manip_display_names: last_grad_manip_idx_s = i
                        
                        for ax_s in (axes_agg_s_array if isinstance(axes_agg_s_array, np.ndarray) else [axes_agg_s_array]):
                            if last_vanilla_idx_s != -1 and last_vanilla_idx_s < num_algos_s - 1:
                                if plotted_algorithms_list_s[last_vanilla_idx_s + 1] not in vanilla_display_names:
                                    line_y_pos1_s = (num_algos_s - 1 - last_vanilla_idx_s) - 0.5
                                    ax_s.axhline(y=line_y_pos1_s, color='dimgray', linestyle=':', linewidth=1.2, zorder=0.5)
                            if last_grad_manip_idx_s != -1 and last_grad_manip_idx_s < num_algos_s - 1:
                                if plotted_algorithms_list_s[last_grad_manip_idx_s + 1] not in grad_manip_display_names and \
                                   plotted_algorithms_list_s[last_grad_manip_idx_s + 1] not in vanilla_display_names:
                                    line_y_pos2_s = (num_algos_s - 1 - last_grad_manip_idx_s) - 0.5
                                    ax_s.axhline(y=line_y_pos2_s, color='dimgray', linestyle=':', linewidth=1.2, zorder=0.5)

                        agg_save_path_s_png = f"scripts/figures/iqm_rliable/aggregate_scores_plot_success_{setting}.png"
                        agg_save_path_s_pdf = f"scripts/figures/iqm_rliable/aggregate_scores_plot_success_{setting}.pdf"
                        os.makedirs(os.path.dirname(agg_save_path_s_png), exist_ok=True)
                        fig_agg_s.savefig(agg_save_path_s_png, dpi=300, bbox_inches='tight')
                        print(f"Combined aggregate scores plot for Success Rate saved to {agg_save_path_s_png}")
                        fig_agg_s.savefig(agg_save_path_s_pdf, bbox_inches='tight')
                        print(f"Combined aggregate scores plot for Success Rate saved to {agg_save_path_s_pdf}")
                        plt.close(fig_agg_s)
                    except Exception as e:
                        print(f"ERROR generating rliable plot for Success Rate: {e}"); traceback.print_exc()
                else: print("No data to plot for Success Rate (rliable).")

            # REWARD Metric
            valid_algos_reward = []
            for setup_type in ordered_setup_types_for_plot:
                algo_display_name = SETUP_INFO[setup_type]['name']
                has_any_reward_data_for_algo = False
                for stat_short_name in STAT_ORDER:
                    if stat_short_name in master_numerical_summary and master_numerical_summary[stat_short_name]:
                        for summary_item in master_numerical_summary[stat_short_name]:
                            if summary_item['setup_type'] == setup_type:
                                if not np.isnan(summary_item.get(f'reward_{stat_short_name}', np.nan)):
                                    has_any_reward_data_for_algo = True; break
                    if has_any_reward_data_for_algo: break
                if has_any_reward_data_for_algo: valid_algos_reward.append(algo_display_name)
            
            if not valid_algos_reward:
                print("No algorithms with plottable reward data for rliable plots. Skipping reward plot.")
            else:
                aggregate_scores_reward, aggregate_score_cis_reward = {}, {}
                for algo_display_name in valid_algos_reward:
                    current_setup_type = next(stype for stype, sinfo in SETUP_INFO.items() if sinfo['name'] == algo_display_name)
                    reward_points, reward_ci_lows, reward_ci_highs = [], [], []
                    for stat_short_name in STAT_ORDER:
                        point_val, low_val, high_val = np.nan, np.nan, np.nan
                        if stat_short_name in master_numerical_summary and master_numerical_summary[stat_short_name]:
                            for summary_item in master_numerical_summary[stat_short_name]:
                                if summary_item['setup_type'] == current_setup_type:
                                    point_val = summary_item.get(f'reward_{stat_short_name}', np.nan)
                                    low_val = summary_item.get(f'reward_ci_low_{stat_short_name}', np.nan)
                                    high_val = summary_item.get(f'reward_ci_high_{stat_short_name}', np.nan); break
                        reward_points.append(point_val); reward_ci_lows.append(low_val); reward_ci_highs.append(high_val)
                    aggregate_scores_reward[algo_display_name] = np.array(reward_points)
                    aggregate_score_cis_reward[algo_display_name] = np.array([reward_ci_lows, reward_ci_highs])

                if aggregate_scores_reward:
                    algos_for_plot_r = valid_algos_reward[::-1]
                    scores_for_plot_r = {name: aggregate_scores_reward[name] for name in algos_for_plot_r}
                    cis_for_plot_r = {name: aggregate_score_cis_reward[name] for name in algos_for_plot_r}
                    try:
                        fig_agg_r, axes_agg_r_array = rly_plot_utils.plot_interval_estimates(
                            scores_for_plot_r, cis_for_plot_r,
                            metric_names=METRIC_NAMES_DISPLAY_RLIABLE, algorithms=algos_for_plot_r,
                            xlabel_y_coordinate=0, xlabel='Average Reward',
                            max_ticks_y=max(5, len(algos_for_plot_r))
                        )
                        plt.subplots_adjust(left=0.25, bottom=0.2)
                        plotted_algorithms_list_r = valid_algos_reward
                        num_algos_r = len(plotted_algorithms_list_r)
                        last_vanilla_idx_r = -1
                        for i, algo_name in enumerate(plotted_algorithms_list_r):
                            if algo_name in vanilla_display_names: last_vanilla_idx_r = i
                        last_grad_manip_idx_r = -1
                        for i, algo_name in enumerate(plotted_algorithms_list_r):
                            if algo_name in grad_manip_display_names: last_grad_manip_idx_r = i
                        
                        for ax_r in (axes_agg_r_array if isinstance(axes_agg_r_array, np.ndarray) else [axes_agg_r_array]):
                            if last_vanilla_idx_r != -1 and last_vanilla_idx_r < num_algos_r - 1:
                                if plotted_algorithms_list_r[last_vanilla_idx_r + 1] not in vanilla_display_names:
                                    line_y_pos1_r = (num_algos_r - 1 - last_vanilla_idx_r) - 0.5
                                    ax_r.axhline(y=line_y_pos1_r, color='dimgray', linestyle=':', linewidth=1.2, zorder=0.5)
                            if last_grad_manip_idx_r != -1 and last_grad_manip_idx_r < num_algos_r - 1:
                                if plotted_algorithms_list_r[last_grad_manip_idx_r + 1] not in grad_manip_display_names and \
                                   plotted_algorithms_list_r[last_grad_manip_idx_r + 1] not in vanilla_display_names:
                                    line_y_pos2_r = (num_algos_r - 1 - last_grad_manip_idx_r) - 0.5
                                    ax_r.axhline(y=line_y_pos2_r, color='dimgray', linestyle=':', linewidth=1.2, zorder=0.5)

                        agg_save_path_r_png = f"scripts/figures/iqm_rliable/aggregate_scores_plot_reward_{setting}.png"
                        agg_save_path_r_pdf = f"scripts/figures/iqm_rliable/aggregate_scores_plot_reward_{setting}.pdf"
                        os.makedirs(os.path.dirname(agg_save_path_r_png), exist_ok=True)
                        fig_agg_r.savefig(agg_save_path_r_png, dpi=300, bbox_inches='tight')
                        print(f"Combined aggregate scores plot for Average Reward saved to {agg_save_path_r_png}")
                        fig_agg_r.savefig(agg_save_path_r_pdf, bbox_inches='tight')
                        print(f"Combined aggregate scores plot for Average Reward saved to {agg_save_path_r_pdf}")
                        plt.close(fig_agg_r)
                    except Exception as e:
                        print(f"ERROR generating rliable plot for Average Reward: {e}"); traceback.print_exc()
                else: print("No data to plot for Average Reward (rliable).")
    else:
        print("\nSkipping plotting stratified bootstrapped CI plots as `rliable` library is not available.")

    print("\nAll requested processing complete!")