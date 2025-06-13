# pip install pandas numpy matplotlib tensorboard PyYAML scipy tqdm seaborn
import os
from collections import defaultdict
import re
import math # For ceiling function for grid dimensions
import yaml # For reading config files
import warnings # Added for LaTeX table helper
import traceback # For detailed error printing
import multiprocessing
from functools import partial

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns # Added for rliable's plot_interval_estimates
from tensorboard.backend.event_processing import event_accumulator
import argparse
from scipy.stats import scoreatpercentile
from tqdm import tqdm

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
    valid_stats = bootstrap_stats_distribution[~np.isnan(bootstrap_stats_distribution)]
    if valid_stats.size < 2:
        return np.nan, np.nan

    alpha = (100.0 - ci_level) / 2.0
    lower_bound = np.percentile(valid_stats, alpha)
    upper_bound = np.percentile(valid_stats, 100.0 - alpha)
    return lower_bound, upper_bound

# --- Helper Function for Bootstrapped Confidence Interval (generic, on a sample of data points) ---
def bootstrap_confidence_interval(data, metric_func, n_iterations=BOOTSTRAP_ITERATIONS, ci_level=CI_PERCENT, desc="Bootstrapping CI", disable_tqdm=True):
    """Calculates the bootstrapped confidence interval for a given metric on data."""
    data = np.asarray(data)
    valid_data = data[~np.isnan(data)]
    if valid_data.size < 2: # Meaningful CI requires at least 2 non-NaN points for bootstrapping
        return np.nan, np.nan

    bootstrap_stats = []
    for _ in tqdm(range(n_iterations), desc=desc, leave=False, ncols=80, disable=disable_tqdm):
        sample = np.random.choice(valid_data, size=len(valid_data), replace=True)
        if sample.size == 0:
            bootstrap_stats.append(np.nan)
            continue
        stat = metric_func(sample)
        bootstrap_stats.append(stat)

    valid_bootstrap_stats = [s for s in bootstrap_stats if not np.isnan(s)]
    if not valid_bootstrap_stats or len(valid_bootstrap_stats) < 2: # Ensure enough bootstrap stats for CI
        return np.nan, np.nan

    return get_percentile_ci(np.array(valid_bootstrap_stats), ci_level)


# --- Helper Function to Generate Comparison String ---
def get_comparison_string(setup_keys):
    if not setup_keys: return "comparison"
    if 'SETUP_INFO' not in globals() or not isinstance(SETUP_INFO, dict):
        return "comparison_setup_info_missing"
    names = [SETUP_INFO[key]['name'] for key in setup_keys if key in SETUP_INFO and 'name' in SETUP_INFO[key]]
    if not names: return "comparison_unknown_setups"
    names.sort()
    sanitized_names = [re.sub(r'\W+', '', name) for name in names]
    return "_vs_".join(sanitized_names)


# --- Helper Function to Extract Setup Info from Filename (Refactored for Multiprocessing) ---
def extract_setup_info(run_name, setup_info_dict):
    """Extracts setup type, envs, and seed from a run name using provided patterns."""
    if not isinstance(setup_info_dict, dict):
        return None, None, None
    for setup_type, info in setup_info_dict.items():
        match = re.search(info['pattern'], run_name)
        if match:
            try:
                num_envs = int(match.group(1))
                seed_number = int(match.group(2))
                return setup_type, num_envs, seed_number
            except (IndexError, ValueError):
                # Pattern might not have 2 groups, or groups are not ints.
                # Consider logging this if it's unexpected.
                continue
    return None, None, None

# --- Helper Function to Adjust Success Data ---
def adjust_data(df, value_col_name='success'):
    df_copy = df.copy()
    if df_copy.empty:
        return pd.DataFrame({value_col_name: [0.0], 'frame': [0]})
    if 0 not in df_copy['frame'].values:
        new_row_data = {value_col_name: [0.0], 'frame': [0]}
        if value_col_name not in df_copy.columns:
            for col in df_copy.columns:
                if col != 'frame': new_row_data[col] = [np.nan]
        new_row = pd.DataFrame(new_row_data)
        result_df = pd.concat([new_row, df_copy], ignore_index=True)
    elif df_copy.loc[df_copy['frame'] == 0, value_col_name].iloc[0] != 0.0:
        result_df = df_copy.copy()
        result_df.loc[result_df['frame'] == 0, value_col_name] = 0.0
    else:
        result_df = df_copy
    result_df.drop_duplicates(subset=['frame'], keep='last', inplace=True)
    return result_df

# --- Helper Function to Adjust Generic Scalar Data (like Reward) ---
def adjust_generic_scalar_data(df, value_col_name='value'):
    df_copy = df.copy()
    if df_copy.empty:
        return pd.DataFrame(columns=[value_col_name, 'frame'])
    df_copy = df_copy.sort_values('frame').reset_index(drop=True)
    df_copy.drop_duplicates(subset=['frame'], keep='last', inplace=True)
    return df_copy


# --- Multiprocessing Worker Function ---
def process_run_directory(item_name, log_dir, setup_info_for_worker, num_tasks_for_worker, horizon_for_worker, current_setting_name_for_worker):
    """
    Processes a single run directory to extract TensorBoard data.
    Designed to be called by a multiprocessing pool.
    """
    item_path = os.path.join(log_dir, item_name)
    CONFIG_FILENAME = "config.yaml"

    setup_type, num_envs, seed_number = extract_setup_info(item_name, setup_info_for_worker)
    if setup_type is None:
        return {'status': 'skipped_no_match', 'item_name': item_name}

    setup_config_main = setup_info_for_worker.get(setup_type)
    if not setup_config_main:
        return {'status': 'skipped_no_config', 'item_name': item_name}

    required_env_count = setup_config_main.get('required_envs')
    if required_env_count is not None and num_envs != required_env_count:
        return {'status': 'skipped_env_mismatch', 'item_name': item_name, 'num_envs': num_envs, 'required': required_env_count}

    config_path = os.path.join(item_path, CONFIG_FILENAME)
    if current_setting_name_for_worker == "mt10":
        task_ids_from_config = [4, 16, 17, 18, 28, 31, 38, 40, 48, 49]
    elif num_tasks_for_worker > 0:
        task_ids_from_config = list(range(num_tasks_for_worker))
    else:
        task_ids_from_config = []

    if os.path.isfile(config_path):
        try:
            with open(config_path, "r") as f:
                config_yaml = yaml.safe_load(f)
            loaded_task_ids_raw = config_yaml.get('task_id')
            if loaded_task_ids_raw is not None and isinstance(loaded_task_ids_raw, list) and all(isinstance(tid, int) for tid in loaded_task_ids_raw):
                task_ids_from_config = loaded_task_ids_raw if loaded_task_ids_raw else list(range(num_tasks_for_worker))
        except Exception:
            pass # Silently fail and use default task IDs

    current_num_tasks_for_run = len(task_ids_from_config)
    event_file_path = None
    for p_dir_opt in [os.path.join(item_path, "summaries"), item_path]:
        if os.path.isdir(p_dir_opt):
            try:
                event_files = [f for f in os.listdir(p_dir_opt) if f.startswith("events.out.tfevents")]
                if event_files:
                    event_file_path = os.path.join(p_dir_opt, sorted(event_files)[-1])
                    break
            except Exception:
                continue
    if event_file_path is None:
        return {'status': 'skipped_no_event_file', 'item_name': item_name}

    try:
        ea = event_accumulator.EventAccumulator(event_file_path, size_guidance={'scalars': 0}, purge_orphaned_data=True)
        ea.Reload()
        all_scalar_tags = ea.Tags().get("scalars", [])
    except Exception:
        return {'status': 'error_loading_ea', 'item_name': item_name}

    if not all_scalar_tags:
        return {'status': 'skipped_no_scalars', 'item_name': item_name}

    # Per-run HORIZON override if needed
    run_horizon = 150 if 'grpo' in item_name.lower() else horizon_for_worker

    run_local_data = defaultdict(lambda: {'per_task_success': defaultdict(list), 'per_task_reward': defaultdict(list), 'overall_avg': list(), 'avg_reward': list()})
    run_data_loaded_flag = False

    overall_s_tag = "Episode/average_task_success_rate"
    if overall_s_tag in all_scalar_tags:
        scalar_events = ea.Scalars(overall_s_tag)
        if scalar_events:
            df_overall_s = pd.DataFrame({'success': [x.value for x in scalar_events], 'frame': [x.step * num_envs * run_horizon for x in scalar_events]})
            if not df_overall_s.empty:
                run_local_data[setup_type]['overall_avg'].append(df_overall_s)
                run_data_loaded_flag = True

    agg_r_tag = "rewards/step"
    if agg_r_tag in all_scalar_tags:
        scalar_events_agg_r = ea.Scalars(agg_r_tag)
        if scalar_events_agg_r:
            df_agg_r = pd.DataFrame({'value': [x.value for x in scalar_events_agg_r], 'frame': [x.step for x in scalar_events_agg_r]})
            if not df_agg_r.empty:
                run_local_data[setup_type]['avg_reward'].append(df_agg_r)
                run_data_loaded_flag = True

    if current_num_tasks_for_run > 0 and task_ids_from_config:
        for task_num_idx, actual_task_id in enumerate(task_ids_from_config):
            s_tag = f"Episode/task_{actual_task_id}_success"
            if s_tag in all_scalar_tags and (s_events := ea.Scalars(s_tag)):
                df_s = pd.DataFrame({'success': [x.value for x in s_events], 'frame': [x.step * num_envs * run_horizon for x in s_events]})
                if not df_s.empty: run_local_data[setup_type]['per_task_success'][task_num_idx].append(df_s)

            r_tag = f"Episode/task_{actual_task_id}_reward"
            if r_tag in all_scalar_tags and (r_events := ea.Scalars(r_tag)):
                df_r = pd.DataFrame({'value': [x.value for x in r_events], 'frame': [x.step for x in r_events]})
                if not df_r.empty: run_local_data[setup_type]['per_task_reward'][task_num_idx].append(df_r)

    if run_data_loaded_flag:
        return {'status': 'processed', 'setup_type': setup_type, 'data': run_local_data[setup_type], 'item_name': item_name}
    else:
        return {'status': 'skipped_no_data', 'item_name': item_name}

def process_run_directory_wrapper(args):
    """Helper to unpack arguments for pool.imap_unordered."""
    return process_run_directory(*args)


# --- Helper Function for Non-Stratified Bootstrapped CI on Overall/Aggregate Metrics ---
def get_final_stat_ci_metric_for_table(
    list_of_run_dataframes,
    point_metric_func,
    value_col_name,
    data_adjuster_func,
    target_frame,
    desc_for_tqdm="OverallMetric",
    is_success_like_metric=False
):
    if not list_of_run_dataframes: return np.nan, np.nan, np.nan
    adjusted_dfs = [data_adjuster_func(df.copy(), value_col_name=value_col_name) for df in list_of_run_dataframes if isinstance(df, pd.DataFrame) and not df.empty]
    adjusted_dfs = [df for df in adjusted_dfs if not df.empty]
    if not adjusted_dfs: return np.nan, np.nan, np.nan

    final_point_values_across_runs = []
    for df_adj in adjusted_dfs:
        if df_adj.empty or value_col_name not in df_adj.columns or 'frame' not in df_adj.columns or len(df_adj) == 0:
            final_point_values_across_runs.append(np.nan); continue

        interp_left_arg, interp_right_arg = np.nan, np.nan
        if data_adjuster_func == adjust_data:
            interp_left_arg = 0.0
        elif not df_adj.empty:
            interp_left_arg = df_adj[value_col_name].iloc[0]
        if not df_adj.empty:
            interp_right_arg = df_adj[value_col_name].iloc[-1]

        interp_val = np.interp(target_frame, df_adj['frame'].values, df_adj[value_col_name].values,
                                 left=interp_left_arg, right=interp_right_arg)
        final_point_values_across_runs.append(interp_val)

    final_point_values_array = np.array(final_point_values_across_runs)
    valid_final_points = final_point_values_array[~np.isnan(final_point_values_array)]

    if valid_final_points.size == 0: return np.nan, np.nan, np.nan

    metric_final = point_metric_func(valid_final_points)

    if valid_final_points.size < 2 : return metric_final, np.nan, np.nan

    ci_low, ci_high = bootstrap_confidence_interval(
        valid_final_points, point_metric_func,
        desc=f"BS CI ({desc_for_tqdm})", disable_tqdm=True
    )
    if is_success_like_metric:
        if not np.isnan(metric_final): metric_final = np.clip(metric_final, 0.0, 1.0)
        if not np.isnan(ci_low): ci_low = np.clip(ci_low, 0.0, 1.0)
        if not np.isnan(ci_high): ci_high = np.clip(ci_high, 0.0, 1.0)
    return metric_final, ci_low, ci_high

# --- Function to Calculate Numerical Summary Data (Replaces print_latex_summary_table's core logic) ---
def calculate_summary_data_for_setting(
    all_setups_data_for_current_setting,
    setup_run_counts_for_current_setting,
    stat_func,
    stat_name_short,
    current_setup_info_for_setting,
    cmd_line_target_frame_overall,
    current_num_tasks
):
    numerical_summary_list = []

    # Determine per-setup max frames if cmd_line_target_frame_overall is not specified
    gsf_s_auto, mfps_s_auto, upsmf_s_auto, _ = get_target_frame_for_metric(
        'per_task_success', 'overall_avg', 'success', adjust_data,
        all_setups_data_for_current_setting, setup_run_counts_for_current_setting, current_num_tasks
    )
    gsf_r_auto, mfps_r_auto, upsmf_r_auto, _ = get_target_frame_for_metric(
        'per_task_reward', 'avg_reward', 'value', adjust_generic_scalar_data,
        all_setups_data_for_current_setting, setup_run_counts_for_current_setting, current_num_tasks
    )

    for setup_type, num_runs_for_setup in tqdm(setup_run_counts_for_current_setting.items(), desc=f"Calculating Summary ({stat_name_short}, {CURRENT_SETTING_NAME})", ncols=100, position=0, disable=False):
        setup_config = current_setup_info_for_setting.get(setup_type, {})
        name = setup_config.get('name', setup_type)
        required_envs = setup_config.get('required_envs')
        env_k_str = "N/A"
        if required_envs is not None:
            try: env_k_str = f"{int(required_envs) // 1024}k"
            except: env_k_str = "Err"

        current_summary_entry = {
            'setup_type': setup_type, 'name': name, 'env_k_str': env_k_str, 'num_runs': num_runs_for_setup,
            f'success_{stat_name_short}': np.nan, f'success_ci_low_{stat_name_short}': np.nan, f'success_ci_high_{stat_name_short}': np.nan,
            f'reward_{stat_name_short}': np.nan, f'reward_ci_low_{stat_name_short}': np.nan, f'reward_ci_high_{stat_name_short}': np.nan,
        }

        target_frame_s_current_setup = cmd_line_target_frame_overall if cmd_line_target_frame_overall is not None and cmd_line_target_frame_overall >=0 else mfps_s_auto.get(setup_type, 0)
        target_frame_r_current_setup = cmd_line_target_frame_overall if cmd_line_target_frame_overall is not None and cmd_line_target_frame_overall >=0 else mfps_r_auto.get(setup_type, 0)
        
        metric_s, ci_l_s, ci_h_s = np.nan, np.nan, np.nan
        overall_avg_data_list = all_setups_data_for_current_setting.get(setup_type, {}).get('overall_avg', [])
        valid_overall_avg_dfs = [df for df in overall_avg_data_list if isinstance(df, pd.DataFrame) and not df.empty]
        if valid_overall_avg_dfs:
            metric_s, ci_l_s, ci_h_s = get_final_stat_ci_metric_for_table(
                valid_overall_avg_dfs, stat_func, 'success', adjust_data, target_frame_s_current_setup,
                desc_for_tqdm=f'{name[:5]}_Suc_Calc', is_success_like_metric=True
            )
        
        if not np.isnan(metric_s):
            current_summary_entry.update({
                f'success_{stat_name_short}': metric_s * 100, # Scaled here
                f'success_ci_low_{stat_name_short}': ci_l_s * 100 if not np.isnan(ci_l_s) else np.nan,
                f'success_ci_high_{stat_name_short}': ci_h_s * 100 if not np.isnan(ci_h_s) else np.nan
            })

        metric_r, ci_l_r, ci_h_r = np.nan, np.nan, np.nan
        avg_reward_data_list = all_setups_data_for_current_setting.get(setup_type, {}).get('avg_reward', [])
        valid_avg_reward_dfs = [df for df in avg_reward_data_list if isinstance(df, pd.DataFrame) and not df.empty]
        if valid_avg_reward_dfs:
            metric_r, ci_l_r, ci_h_r = get_final_stat_ci_metric_for_table(
                valid_avg_reward_dfs, stat_func, 'value', adjust_generic_scalar_data, target_frame_r_current_setup,
                desc_for_tqdm=f'{name[:5]}_Rew_Calc', is_success_like_metric=False
            )

        if not np.isnan(metric_r):
            current_summary_entry.update({
                f'reward_{stat_name_short}': metric_r, # Not scaled by 100
                f'reward_ci_low_{stat_name_short}': ci_l_r if not np.isnan(ci_l_r) else np.nan,
                f'reward_ci_high_{stat_name_short}': ci_h_r if not np.isnan(ci_h_r) else np.nan
            })
        numerical_summary_list.append(current_summary_entry)
    return numerical_summary_list

# (Ensure script_setup_type matches keys in SETUP_INFO for different settings)
ALGO_DISPLAY_MAPPING_AND_ORDER = {
    "group1": [ # Vanilla Methods
        {"table_name": "Vanilla", "script_setup_type": "shppo_vanilla"},
        {"table_name": "Multihead", "script_setup_type": "mhppo_vanilla"},
        {"table_name": "GRPO-Vanilla", "script_setup_type": "grpo_vanilla"},
    ],
    "group2": [ # Gradient Manipulation Methods
        {"table_name": "PCGrad", "script_setup_type": "pcgrad"},
        {"table_name": "CAGrad", "script_setup_type": "cagrad"},
        {"table_name": "FAMO", "script_setup_type": "famo"},
    ],
    "group3": [ # Other Advanced Methods
        {"table_name": "PaCo", "script_setup_type": "shppo_paco"},
        {"table_name": "SH-MOORE", "script_setup_type": "shppo_moore"},
        {"table_name": "MH-MOORE", "script_setup_type": "mhppo_moore"},
        {"table_name": "SH-CARE", "script_setup_type": "shppo_care"},
        {"table_name": "MH-CARE", "script_setup_type": "mhppo_care"},
        {"table_name": "Soft-Modularization", "script_setup_type": "soft_modularization"},
    ]
}

# --- Helper to format metric string for the new table ---
def format_metric_value_for_final_table(point, ci_low, ci_high):
    if np.isnan(point):
        return "x"
    # Assumes point, ci_low, ci_high are already correctly scaled before being passed.
    if np.isnan(ci_low) or np.isnan(ci_high): # If CI is incomplete, just show point estimate
        return f"${point:.2f}$"
    else:
        return f"${point:.2f} [{ci_low:.2f}, {ci_high:.2f}]$"

# --- New LaTeX Table Generation Function (Final Format) ---
def generate_latex_table_final_format(
    all_settings_numerical_summaries_data, # Dict: setting_name -> stat_short_name -> numerical_summary_list
    all_settings_details_for_table_gen,    # Dict: setting_name -> {details like 'setup_info', 'num_tasks', etc.}
    active_settings_for_table,             # List of setting names to include in this table, e.g., ['mt10', 'mt50']
    stat_info_for_table,                   # Dict for the current statistic, e.g., {"short_name": "mean", "display_name": "Mean"}
    cmd_target_frame_for_caption,          # Parsed args.target_frame
    ci_level_for_caption                   # Global CI_PERCENT
):
    stat_short = stat_info_for_table["short_name"]
    stat_display_name = stat_info_for_table["display_name"]

    print(f"\n\n--- Generating LaTeX Table ({stat_display_name}) for Settings: {', '.join(active_settings_for_table)} ---")
    print("\\begin{table}[h!]\n\\centering")
    
    resize_width = ".8\\columnwidth" if len(active_settings_for_table) > 1 else ".6\\columnwidth"
    print(f"\\resizebox{{{resize_width}}}{{!}}{{%\n\\begin{{tabular}}{{l{''.join(['cc' for _ in active_settings_for_table])}}}")
    print("\\toprule")

    # Header Row 1: \diagbox and \multicolumn for setting names
    print("\\multirow{2}{*}{\\diagbox{Methods}{Tasks}}", end="")
    for setting_name in active_settings_for_table:
        display_setting_name = f"{setting_name.upper()}-rand"
        print(f" & \\multicolumn{{2}}{{c}}{{{display_setting_name}}}", end="")
    print(" \\\\")

    # Header Row 2: \cmidrule and SR/R
    # \cmidrule requires careful indexing. For 'l c c c c', cols are 1, 2-3, 4-5.
    # So for active_settings_for_table[i], cols are (2+2*i)-(3+2*i)
    for i in range(len(active_settings_for_table)):
        start_col = 2 + 2 * i
        end_col = start_col + 1
        print(f"\\cmidrule(lr){{{start_col}-{end_col}}}", end=" ")
    print("\n & ", end="") # For the first column (Methods diagonal)
    
    metrics_headers = []
    for _ in active_settings_for_table:
        metrics_headers.extend(["SR $\\uparrow$", "R $\\uparrow$"])
    print(" & ".join(metrics_headers) + " \\\\")
    print("\\midrule")

    # Data Rows
    num_groups = len(ALGO_DISPLAY_MAPPING_AND_ORDER)
    current_group_idx = 0
    for group_name, algos_in_group_list in ALGO_DISPLAY_MAPPING_AND_ORDER.items():
        for algo_map_entry in algos_in_group_list:
            table_method_name = algo_map_entry["table_name"]
            script_setup_key = algo_map_entry["script_setup_type"]
            
            print(table_method_name.replace("_", "\\_"), end="") # Method name

            for setting_name_current_col in active_settings_for_table:
                sr_val_str, r_val_str = "x", "x" # Defaults if no data

                # Check if data exists for this setting and stat
                if setting_name_current_col in all_settings_numerical_summaries_data and \
                   stat_short in all_settings_numerical_summaries_data[setting_name_current_col]:
                    
                    summaries_for_current_setting_stat = all_settings_numerical_summaries_data[setting_name_current_col][stat_short]
                    
                    algo_summary_data_found = None
                    for summary_item in summaries_for_current_setting_stat:
                        if summary_item['setup_type'] == script_setup_key:
                            algo_summary_data_found = summary_item
                            break
                    
                    if algo_summary_data_found:
                        # Success Rate (SR) - already scaled by 100 in calculate_summary_data_for_setting
                        s_point = algo_summary_data_found.get(f'success_{stat_short}', np.nan)
                        s_low   = algo_summary_data_found.get(f'success_ci_low_{stat_short}', np.nan)
                        s_high  = algo_summary_data_found.get(f'success_ci_high_{stat_short}', np.nan)
                        sr_val_str = format_metric_value_for_final_table(s_point, s_low, s_high)

                        # Reward (R) - not scaled by 100
                        r_point = algo_summary_data_found.get(f'reward_{stat_short}', np.nan)
                        r_low   = algo_summary_data_found.get(f'reward_ci_low_{stat_short}', np.nan)
                        r_high  = algo_summary_data_found.get(f'reward_ci_high_{stat_short}', np.nan)
                        r_val_str = format_metric_value_for_final_table(r_point, r_low, r_high)
                        
                print(f" & {sr_val_str} & {r_val_str}", end="")
            print(" \\\\")
        
        current_group_idx += 1
        if current_group_idx < num_groups:
            print("\\midrule")

    print("\\bottomrule\n\\end{tabular}\n}") # End resizebox

    # Caption and Label - simplified as per user's example format
    print(f"\\caption{{{stat_display_name} [{ci_level_for_caption:.0f}\% CI] reported. SR is Success Rate (\\%), R is Average Reward.}}")
    active_settings_label_part = "_".join(sorted(active_settings_for_table))
    print(f"\\label{{table:main_table_metaworld_results_{stat_short}_{active_settings_label_part}}}")
    print("\\end{table}")


# Helper function to determine target frames
def get_target_frame_for_metric(metric_key_per_task, metric_key_aggregate, value_col, adjuster_func, all_setups_data_local, setup_run_counts_local, current_num_tasks_for_frame):
    target_frame = 0; max_frames_per_setup_metric = {};
    has_any_per_task_data_for_metric = False
    all_max_frames_across_setups_for_metric = []

    for setup_type_iter, num_runs_iter in setup_run_counts_local.items():
        if num_runs_iter == 0 or setup_type_iter not in all_setups_data_local:
            max_frames_per_setup_metric[setup_type_iter] = 0
            continue
        current_setup_max_frame_metric = 0; setup_has_any_data_for_frame = False
        setup_data = all_setups_data_local.get(setup_type_iter, {})

        if metric_key_aggregate and metric_key_aggregate in setup_data:
            aggregate_data_list = setup_data[metric_key_aggregate]
            adjusted_dfs_agg = [adjuster_func(df.copy(), value_col) for df in aggregate_data_list if isinstance(df, pd.DataFrame) and not df.empty]
            for df_agg in adjusted_dfs_agg:
                if not df_agg.empty and 'frame' in df_agg.columns:
                    current_setup_max_frame_metric = max(current_setup_max_frame_metric, df_agg['frame'].max())
                    setup_has_any_data_for_frame = True

        if not setup_has_any_data_for_frame and metric_key_per_task and metric_key_per_task in setup_data:
            per_task_data_for_setup = setup_data[metric_key_per_task]
            if isinstance(per_task_data_for_setup, dict) and current_num_tasks_for_frame > 0 and \
                any(per_task_data_for_setup.get(tid) for tid in range(current_num_tasks_for_frame)):
                has_any_per_task_data_for_metric = True
                for task_num in range(current_num_tasks_for_frame):
                    run_data_list = per_task_data_for_setup.get(task_num, [])
                    adjusted_dfs = [adjuster_func(df.copy(), value_col) for df in run_data_list if isinstance(df, pd.DataFrame) and not df.empty]
                    for df_adj in adjusted_dfs:
                        if not df_adj.empty and 'frame' in df_adj.columns:
                            current_setup_max_frame_metric = max(current_setup_max_frame_metric, df_adj['frame'].max())
                            setup_has_any_data_for_frame = True

        max_frames_per_setup_metric[setup_type_iter] = current_setup_max_frame_metric
        if current_setup_max_frame_metric > 0 :
            all_max_frames_across_setups_for_metric.append(current_setup_max_frame_metric)

    if all_max_frames_across_setups_for_metric:
        target_frame = min(all_max_frames_across_setups_for_metric)
    else: target_frame = 0

    use_per_setup_max = (target_frame == 0 and any(val > 0 for val in max_frames_per_setup_metric.values()))
    return target_frame, max_frames_per_setup_metric, use_per_setup_max, has_any_per_task_data_for_metric

# --- Dynamic Pattern and SETUP_INFO generation ---
def get_patterns_for_setting(setting_str):
    patterns = {}
    if setting_str == "mt10":
        patterns['PATTERN_SHPPO_VANILLA'] = rf"05_09_ppo_vanilla_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_MHPPO_VANILLA'] = rf"mhppo_vanilla_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_FAMO'] = rf"05_09_ppo_famo_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_PCGRAD'] = rf"05_11_ppo_pcgrad_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_CAGRAD'] = rf"ppo_cagrad_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_SHPPO_PACO'] = rf"shppo_paco_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_MHPPO_MOORE'] = rf"mhppo_moore_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_SHPPO_MOORE'] = rf"shppo_moore_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_SOFT_MODULARIZATION'] = rf"ppo_soft_modularization_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_MHPPO_CARE'] = rf"05_11_mhppo_care_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_SHPPO_CARE'] = rf"shppo_care_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_GRPO'] = rf"05_26_grpo_vanilla_{setting_str}_rand_(\d+)_seed_(\d+).*"
    else: # Default MT50 patterns
        patterns['PATTERN_SHPPO_VANILLA'] = rf"^ppo_vanilla_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_MHPPO_VANILLA'] = rf"05_07_mhppo_vanilla_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_FAMO'] = rf"ppo_famo_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_PCGRAD'] = rf"ppo_pcgrad_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_CAGRAD'] = rf"ppo_cagrad_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_SHPPO_PACO'] = rf"shppo_paco_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_MHPPO_MOORE'] = rf"mhppo_moore_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_SHPPO_MOORE'] = rf"shppo_moore_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_MHPPO_CARE'] = rf"mhppo_care_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_SHPPO_CARE'] = rf"shppo_care_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_SOFT_MODULARIZATION'] = rf"ppo_soft_modularization_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_GRPO'] = rf"05_26_grpo_vanilla_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
    return patterns

def get_setup_info_for_setting(setting_str, current_patterns):
    required_envs_val = 24576
    if setting_str == "mt10":
        pc_grad_required_envs = 24576
        grpo_required_envs = 4096
    else: # mt50
        pc_grad_required_envs = 8192
        grpo_required_envs = 24576

    return {
        'shppo_vanilla': {'pattern': current_patterns['PATTERN_SHPPO_VANILLA'], 'color': '#1A85FF', 'name': 'SH-Vanilla', 'required_envs': required_envs_val},
        'mhppo_vanilla': {'pattern': current_patterns['PATTERN_MHPPO_VANILLA'], 'color': '#2ca02c', 'name': 'MH-Vanilla', 'required_envs': required_envs_val},
        'grpo_vanilla': {'pattern': current_patterns['PATTERN_GRPO'], 'color': "#3309f0", 'name': 'GRPO-Vanilla', 'required_envs': grpo_required_envs},
        'pcgrad': {'pattern': current_patterns['PATTERN_PCGRAD'], 'color': '#ff7f0e', 'name': 'PCGrad', 'required_envs': pc_grad_required_envs},
        'cagrad': {'pattern': current_patterns['PATTERN_CAGRAD'], 'color': '#1f77b4', 'name': 'CAGrad', 'required_envs': required_envs_val},
        'famo': {'pattern': current_patterns['PATTERN_FAMO'], 'color': '#3690ff', 'name': 'FAMO', 'required_envs': required_envs_val},
        'soft_modularization': {'pattern': current_patterns['PATTERN_SOFT_MODULARIZATION'], 'color': '#FF4081', 'name': 'Soft-Modularization', 'required_envs': required_envs_val},
        'shppo_care': {'pattern': current_patterns['PATTERN_SHPPO_CARE'], 'color': '#FF5722', 'name': 'SH-CARE', 'required_envs': required_envs_val},
        'mhppo_care': {'pattern': current_patterns['PATTERN_MHPPO_CARE'], 'color': '#FF9800', 'name': 'MH-CARE', 'required_envs': required_envs_val},
        'shppo_paco': {'pattern': current_patterns['PATTERN_SHPPO_PACO'], 'color': '#E91E63', 'name': 'SH-PaCo', 'required_envs': required_envs_val},
        'shppo_moore': {'pattern': current_patterns['PATTERN_SHPPO_MOORE'], 'color': '#C2185B', 'name': 'SH-MOORE', 'required_envs': required_envs_val},
        'mhppo_moore': {'pattern': current_patterns['PATTERN_MHPPO_MOORE'], 'color': '#F50057', 'name': 'MH-MOORE', 'required_envs': required_envs_val},
    }

STATISTICS_TO_REPORT = [
    {"func": np.mean, "short_name": "mean", "display_name": "Mean"},
]
STAT_ORDER = [s['short_name'] for s in STATISTICS_TO_REPORT]
METRIC_NAMES_DISPLAY_RLIABLE = [s['display_name'] for s in STATISTICS_TO_REPORT]


# --- Integrated rliable.plot_utils._decorate_axis (simplified) ---
def _decorate_axis(ax, ticklabelsize='x-large', wrect=10):
  """Helper function for decorating plots."""
  ax.tick_params(axis='both', which='major', labelsize=ticklabelsize)
  ax.tick_params(axis='both', which='minor', labelsize=ticklabelsize)
  ax.spines['bottom'].set_visible(False)
  ax.spines['top'].set_visible(False)

# --- Helper function to add padding to x-axis limits ---
def adjust_plot_xlimits(ax, all_lowers, all_uppers, padding_fraction=0.05):
    """Adjusts x-axis limits to add padding."""
    if not all_lowers and not all_uppers:
        current_xlim = ax.get_xlim()
        if current_xlim != (0.0, 1.0):
                return
        return

    valid_lowers = [x for x in all_lowers if not np.isnan(x)]
    valid_uppers = [x for x in all_uppers if not np.isnan(x)]

    if not valid_lowers and not valid_uppers:
        return

    min_val = min(valid_lowers) if valid_lowers else (min(valid_uppers) if valid_uppers else 0)
    max_val = max(valid_uppers) if valid_uppers else (max(valid_lowers) if valid_lowers else 1)
    
    if not valid_lowers: min_val = max_val - abs(max_val * 0.2) if max_val !=0 else -0.2
    if not valid_uppers: max_val = min_val + abs(min_val * 0.2) if min_val !=0 else 0.2

    data_range = max_val - min_val
    if data_range == 0:
        padding = abs(min_val) * padding_fraction if min_val != 0 else 0.1
    else:
        padding = data_range * padding_fraction
    
    ax.set_xlim(min_val - padding, max_val + padding)


# --- Integrated rliable.plot_utils.plot_interval_estimates (from user) ---
def plot_interval_estimates(point_estimates,
                            interval_estimates,
                            metric_names,
                            algorithms=None,
                            colors=None,
                            color_palette='colorblind',
                            max_ticks=4,
                            subfigure_width=3.4,
                            row_height=0.37,
                            xlabel_y_coordinate=-0.1,
                            xlabel='Normalized Score',
                            **kwargs):
    """Plots various metrics with confidence intervals. (User Provided Code)"""

    if algorithms is None:
        algorithms = list(point_estimates.keys())
    if not algorithms:
        fig, ax = plt.subplots(figsize=(subfigure_width, row_height if row_height > 0.1 else 1.0))
        ax.text(0.5, 0.5, "No data/algorithms", ha='center', va='center', transform=ax.transAxes)
        return fig, ax

    if not point_estimates or algorithms[0] not in point_estimates:
        fig, ax = plt.subplots(figsize=(subfigure_width, row_height * len(algorithms) if algorithms else 1.0))
        ax.text(0.5, 0.5, f"Data missing for {algorithms[0] if algorithms else 'algorithms'}", ha='center', va='center', transform=ax.transAxes)
        return fig, ax

    num_metrics = len(point_estimates[algorithms[0]])
    if num_metrics == 0:
        fig, ax = plt.subplots(figsize=(subfigure_width, row_height * len(algorithms)))
        ax.text(0.5, 0.5, f"No metrics for {algorithms[0]}", ha='center', va='center', transform=ax.transAxes)
        return fig, ax

    figsize = (subfigure_width * num_metrics, row_height * len(algorithms))
    fig, axes = plt.subplots(nrows=1, ncols=num_metrics, figsize=figsize)
    if colors is None:
        color_palette_obj = sns.color_palette(color_palette, n_colors=len(algorithms))
        colors = dict(zip(algorithms, color_palette_obj))
    h = kwargs.pop('interval_height', 0.6)

    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx] if num_metrics > 1 else axes
        all_lowers_metric, all_uppers_metric = [], []

        for alg_idx, algorithm in enumerate(algorithms):
            if algorithm in interval_estimates and algorithm in point_estimates:
                current_interval = interval_estimates[algorithm]
                current_point = point_estimates[algorithm]
                if current_interval.ndim == 2 and current_interval.shape[1] > idx and \
                   current_point.ndim == 1 and len(current_point) > idx:
                    lower, upper = current_interval[:, idx]
                    if not (np.isnan(lower) or np.isnan(upper)):
                        ax.barh(
                            y=alg_idx,
                            width=upper - lower,
                            height=h,
                            left=lower,
                            color=colors.get(algorithm, '#000000'),
                            alpha=0.75,
                            label=algorithm if alg_idx == 0 and idx == 0 else None)
                        all_lowers_metric.append(lower)
                        all_uppers_metric.append(upper)
                    point_val = current_point[idx]
                    if not np.isnan(point_val):
                        ax.vlines(
                            x=point_val,
                            ymin=alg_idx - (7.5 * h / 16),
                            ymax=alg_idx + (6 * h / 16),
                            color='k',
                            alpha=0.5)
                        if not (np.isnan(lower) or np.isnan(upper)):
                            all_lowers_metric.append(min(lower, point_val))
                            all_uppers_metric.append(max(upper, point_val))
                        else:
                            all_lowers_metric.append(point_val)
                            all_uppers_metric.append(point_val)

        ax.set_yticks(list(range(len(algorithms))))
        ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
        
        adjust_plot_xlimits(ax, all_lowers_metric, all_uppers_metric)

        if idx != 0:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', length=0)
        else:
            ax.set_yticklabels(algorithms, fontsize='x-large')
        ax.set_title(metric_name, fontsize='xx-large')
        ax.tick_params(axis='both', which='major')
        _decorate_axis(ax, ticklabelsize='xx-large', wrect=5)
        ax.spines['left'].set_visible(False)
        ax.grid(True, axis='x', alpha=0.25)

    fig_text_y = xlabel_y_coordinate if xlabel else 0.01
    if xlabel:
        fig.text(0.5, fig_text_y, xlabel, ha='center', va='bottom', fontsize='xx-large')

    bottom_padding = 0.20 if xlabel else 0.05
    top_padding = 0.85 if any(str(m).strip() for m in metric_names) else 0.95
    
    left_padding = 0.05
    if num_metrics == 1 and algorithms and len(algorithms) > 0:
        max_label_len = max(len(name) for name in algorithms) if algorithms else 10
        left_padding = min(0.4, max(0.1, 0.01 * max_label_len + 0.05 * (len(algorithms) / 10)))


    plt.subplots_adjust(wspace=kwargs.pop('wspace', 0.15), left=left_padding, right=0.95, bottom=bottom_padding, top=top_padding)
    return fig, axes


# --- Function to extract data for RLiable plots ---
def extract_data_for_rliable_plot(master_numerical_summary_dict, setup_run_counts_dict, current_setup_info_dict):
    output_data = {}
    ordered_setup_types_for_plot = [
        stype for stype in current_setup_info_dict.keys()
        if stype in setup_run_counts_dict and setup_run_counts_dict[stype] > 0
    ]

    for metric_type in ["success", "reward"]:
        scores_for_current_metric = {}
        cis_for_current_metric = {}

        valid_algos_display_names_metric = []
        for setup_type_iter in ordered_setup_types_for_plot:
            if setup_type_iter not in current_setup_info_dict:
                continue
            algo_display_name = current_setup_info_dict[setup_type_iter]['name']
            has_data = any(not np.isnan(summary_item.get(f'{metric_type}_{stat_short_name}', np.nan))
                           for stat_short_name in STAT_ORDER
                           if stat_short_name in master_numerical_summary_dict and master_numerical_summary_dict[stat_short_name]
                           for summary_item in master_numerical_summary_dict[stat_short_name]
                           if summary_item['setup_type'] == setup_type_iter)
            if has_data:
                valid_algos_display_names_metric.append(algo_display_name)

        if not valid_algos_display_names_metric:
            output_data[metric_type] = {"scores": {}, "cis": {}, "algos_ordered": []}
            continue

        for algo_display_name in valid_algos_display_names_metric:
            current_setup_type = next((stype for stype, sinfo in current_setup_info_dict.items() if sinfo['name'] == algo_display_name), None)
            if current_setup_type is None: continue

            points, ci_lows, ci_highs = [], [], []
            for stat_short_name in STAT_ORDER:
                p, l, h = np.nan, np.nan, np.nan
                if stat_short_name in master_numerical_summary_dict and master_numerical_summary_dict[stat_short_name]:
                    for item in master_numerical_summary_dict[stat_short_name]:
                        if item['setup_type'] == current_setup_type:
                            p = item.get(f'{metric_type}_{stat_short_name}', np.nan)
                            l = item.get(f'{metric_type}_ci_low_{stat_short_name}', np.nan)
                            h = item.get(f'{metric_type}_ci_high_{stat_short_name}', np.nan)
                            break
                points.append(p); ci_lows.append(l); ci_highs.append(h)
            scores_for_current_metric[algo_display_name] = np.array(points)
            cis_for_current_metric[algo_display_name] = np.array([ci_lows, ci_highs])

        algos_ordered_for_plot = valid_algos_display_names_metric[::-1]

        output_data[metric_type] = {
            "scores": {name: scores_for_current_metric[name] for name in algos_ordered_for_plot if name in scores_for_current_metric},
            "cis": {name: cis_for_current_metric[name] for name in algos_ordered_for_plot if name in cis_for_current_metric},
            "algos_ordered": algos_ordered_for_plot
        }
    return output_data

# --- Function for Individual RLiable Plots (using local plot_interval_estimates) ---
def generate_individual_rliable_plots(master_summary, run_counts, s_info, setting_name, cmd_args): # Removed specified_target_frame from direct params
    print(f"\n--- Generating Individual Aggregate Interval Plots for {setting_name.upper()} (using local plotting function) ---")

    rliable_data_for_setting = extract_data_for_rliable_plot(master_summary, run_counts, s_info)
    
    # Use cmd_args.target_frame to determine label suffix, as per user's provided code structure
    specified_target_frame = cmd_args.target_frame
    frame_label_plot = f" (at {specified_target_frame:,} frames)" if specified_target_frame is not None else ""


    for metric_type, xlabel_text_base in [("success", "Success Rate (%)"), ("reward", "Average Reward")]:
        metric_data = rliable_data_for_setting.get(metric_type)
        if not metric_data or not metric_data['algos_ordered']:
            print(f"No algorithms with plottable {metric_type} data for rliable plots ({setting_name.upper()}). Skipping.")
            continue

        scores_for_plot = metric_data['scores']
        cis_for_plot = metric_data['cis']
        algos_for_plot = metric_data['algos_ordered']

        metric_names_for_plot = ["" if name == "Mean" else name for name in METRIC_NAMES_DISPLAY_RLIABLE]

        algo_to_color = {details['name']: details['color'] for _, details in s_info.items()}
        default_palette_size = len(algos_for_plot) if algos_for_plot else 1
        color_palette_default = sns.color_palette('colorblind', n_colors=default_palette_size)
        colors_for_plot_fn = {
            algo_name: algo_to_color.get(algo_name, color_palette_default[i % len(color_palette_default)])
            for i, algo_name in enumerate(algos_for_plot)
        }

        try:
            num_actual_metrics = len(metric_names_for_plot)
            row_h = 0.37
            if algos_for_plot: row_h = max(0.25, 0.37 * min(15, len(algos_for_plot)) / (len(algos_for_plot)/10.0 if len(algos_for_plot)>10 else 1.0) )


            dynamic_subfigure_width = 3.4
            if num_actual_metrics == 1 and algos_for_plot:
                max_label_len = max(len(name) for name in algos_for_plot) if algos_for_plot else 10
                dynamic_subfigure_width = max(3.4, 1.5 + max_label_len * 0.08)


            fig_agg, axes_agg_array = plot_interval_estimates(
                point_estimates=scores_for_plot,
                interval_estimates=cis_for_plot,
                metric_names=metric_names_for_plot,
                algorithms=algos_for_plot,
                xlabel=f"{setting_name.upper()} - {xlabel_text_base}{frame_label_plot}", # X-axis label kept as per user's code
                colors=colors_for_plot_fn,
                max_ticks=4,
                subfigure_width=dynamic_subfigure_width,
                row_height=row_h,
                xlabel_y_coordinate=0.08,
            )

            active_axes = axes_agg_array
            if not isinstance(axes_agg_array, np.ndarray):
                active_axes = [axes_agg_array]

            vanilla_display_names = {s_info[stype]['name'] for stype in s_info if 'vanilla' in stype.lower() and 'name' in s_info[stype]}
            grad_manip_display_names = {s_info[stype]['name'] for stype in s_info if stype in ['famo', 'pcgrad', 'cagrad'] and 'name' in s_info[stype]}

            plotted_algos_list_ordered = algos_for_plot[::-1]
            num_algos = len(plotted_algos_list_ordered)
            last_vanilla_idx = -1; last_grad_idx = -1
            for i, name in enumerate(plotted_algos_list_ordered):
                if name in vanilla_display_names: last_vanilla_idx = i
                if name in grad_manip_display_names: last_grad_idx = i

            for ax_k in active_axes:
                pass # Removed axhline calls for dotted lines
            
            # --- Updated save paths for individual plots ---
            save_path_png = f"scripts/figures/CI/plot_{metric_type}_CI_{setting_name}.png"
            save_path_pdf = f"scripts/figures/CI/plot_{metric_type}_CI_{setting_name}.pdf"
            # --- End of updated save paths ---

            os.makedirs(os.path.dirname(save_path_png), exist_ok=True)
            fig_agg.savefig(save_path_png, dpi=300, bbox_inches='tight')
            print(f"Individual aggregate scores plot for {xlabel_text_base} ({setting_name.upper()}) saved to {save_path_png}")
            fig_agg.savefig(save_path_pdf, bbox_inches='tight')
            plt.close(fig_agg)
        except Exception as e:
            print(f"ERROR generating individual rliable plot for {xlabel_text_base} ({setting_name.upper()}): {e}"); traceback.print_exc()


# --- Function for Combined RLiable Plots (Modified to use local plotting logic) ---
def generate_combined_rliable_plots(all_rliable_data, all_s_info, cmd_args): # Removed specified_target_frame from direct params
    print("\n--- Generating Combined Side-by-Side Aggregate Interval Plots (using local plotting logic) ---")

    metric_name_orig = METRIC_NAMES_DISPLAY_RLIABLE[0]
    subplot_title_text = "" if metric_name_orig == "Mean" else metric_name_orig
    metric_idx_to_plot = 0
    
    # Use cmd_args.target_frame to determine label suffix, as per user's provided code structure
    specified_target_frame = cmd_args.target_frame


    for metric_type, xlabel_base_text in [("success", "Success Rate (%)"), ("reward", "Average Reward")]:

        data_mt10 = all_rliable_data.get('mt10', {}).get(metric_type)
        data_mt50 = all_rliable_data.get('mt50', {}).get(metric_type)

        y_algorithms_ordered = []
        if data_mt10 and data_mt10.get('algos_ordered'):
            y_algorithms_ordered = data_mt10['algos_ordered']
        elif data_mt50 and data_mt50.get('algos_ordered'):
            y_algorithms_ordered = data_mt50['algos_ordered']
        
        if not y_algorithms_ordered:
            all_algo_names = set()
            if 'mt10' in all_s_info:
                all_algo_names.update(s_info_item['name'] for s_info_item in all_s_info['mt10'].values() if 'name' in s_info_item)
            if 'mt50' in all_s_info:
                all_algo_names.update(s_info_item['name'] for s_info_item in all_s_info['mt50'].values() if 'name' in s_info_item)
            
            if all_algo_names:
                ref_s_info = all_s_info.get('mt10', all_s_info.get('mt50', {}))
                
                ordered_ref_names_from_setup_info = []
                temp_all_algo_names = all_algo_names.copy() 

                for _, details in ref_s_info.items(): 
                    if details['name'] in temp_all_algo_names:
                        ordered_ref_names_from_setup_info.append(details['name'])
                        temp_all_algo_names.remove(details['name'])
                
                if temp_all_algo_names:
                    ordered_ref_names_from_setup_info.extend(sorted(list(temp_all_algo_names)))

                y_algorithms_ordered = ordered_ref_names_from_setup_info[::-1] 

        if not y_algorithms_ordered:
            print(f"No algorithms found for Y-axis in combined {metric_type} plot. Skipping.")
            continue

        # --- Logic to determine positions for group separator lines ---
        # This assumes y_algorithms_ordered has G3 algos at top, then G2, then G1 at bottom.
        script_to_display_name_map = {}
        for setting_key_for_map in ['mt50', 'mt10']: 
            if setting_key_for_map in all_s_info:
                for setup_type, info_dict in all_s_info[setting_key_for_map].items():
                    script_to_display_name_map[setup_type] = info_dict.get('name', setup_type)
        
        group1_actual_display_names = {
            script_to_display_name_map.get(algo_map["script_setup_type"])
            for algo_map in ALGO_DISPLAY_MAPPING_AND_ORDER.get("group1", [])
            if script_to_display_name_map.get(algo_map["script_setup_type"]) is not None
        }
        group2_actual_display_names = {
            script_to_display_name_map.get(algo_map["script_setup_type"])
            for algo_map in ALGO_DISPLAY_MAPPING_AND_ORDER.get("group2", [])
            if script_to_display_name_map.get(algo_map["script_setup_type"]) is not None
        }
        group3_actual_display_names = {
            script_to_display_name_map.get(algo_map["script_setup_type"])
            for algo_map in ALGO_DISPLAY_MAPPING_AND_ORDER.get("group3", [])
            if script_to_display_name_map.get(algo_map["script_setup_type"]) is not None
        }

        last_g3_idx_on_plot = -1
        last_g2_idx_on_plot = -1

        for alg_plot_idx, algo_name_on_plot in enumerate(y_algorithms_ordered):
            if algo_name_on_plot in group3_actual_display_names:
                last_g3_idx_on_plot = alg_plot_idx
            if algo_name_on_plot in group2_actual_display_names:
                last_g2_idx_on_plot = alg_plot_idx
        
        line_y_after_g3_block = -1
        if last_g3_idx_on_plot != -1 and last_g3_idx_on_plot < len(y_algorithms_ordered) - 1:
            # Check if the algorithm following the last G3 algorithm is actually a G2 algorithm
            next_algo_name_check = y_algorithms_ordered[last_g3_idx_on_plot + 1]
            if next_algo_name_check in group2_actual_display_names:
                 line_y_after_g3_block = last_g3_idx_on_plot + 0.5
        
        line_y_after_g2_block = -1
        if last_g2_idx_on_plot != -1 and last_g2_idx_on_plot < len(y_algorithms_ordered) - 1:
            # Check if the algorithm following the last G2 algorithm is actually a G1 algorithm
            next_algo_name_check = y_algorithms_ordered[last_g2_idx_on_plot + 1]
            if next_algo_name_check in group1_actual_display_names:
                line_y_after_g2_block = last_g2_idx_on_plot + 0.5
        # --- End of logic for group separator lines ---

        num_algorithms = len(y_algorithms_ordered)
        fig_height = max(2.0, 0.40 * num_algorithms + 0.5) 
        fig_width_per_subplot = 5.0

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(fig_width_per_subplot * 2, fig_height), sharey=True)
        ax_mt10, ax_mt50 = axes[0], axes[1]

        plot_data_list = [ 
            ('mt10', data_mt10, all_s_info.get('mt10', {}), ax_mt10, f"MT10"),
            ('mt50', data_mt50, all_s_info.get('mt50', {}), ax_mt50, f"MT50")
        ]

        max_ticks_val = 4
        interval_height_val = 0.6

        for setting_name_internal, current_setting_rliable_data, current_s_info, current_ax, xlabel_suffix_for_ax in plot_data_list:
            all_lowers_subplot, all_uppers_subplot = [], []
            current_ax.set_xlabel(xlabel_suffix_for_ax, fontsize='x-large') 
            current_ax.set_title(subplot_title_text, fontsize='xx-large') 
            current_ax.set_yticks(np.arange(num_algorithms))

            if not current_setting_rliable_data or not current_setting_rliable_data.get('algos_ordered'):
                current_ax.text(0.5, 0.5, f'No {setting_name_internal.upper()} Data', ha='center', va='center', transform=current_ax.transAxes)
                if current_ax == ax_mt10: 
                    current_ax.set_yticklabels(y_algorithms_ordered, fontsize='x-large')
                else:
                    current_ax.set_yticklabels([])
            else:
                point_estimates_setting = current_setting_rliable_data['scores']
                interval_estimates_setting = current_setting_rliable_data['cis']

                algo_to_color = {details['name']: details['color'] for _, details in current_s_info.items() if 'name' in details and 'color' in details}
                default_palette_size = len(y_algorithms_ordered) if y_algorithms_ordered else 1
                color_palette_default = sns.color_palette('colorblind', n_colors=default_palette_size)

                for alg_idx, common_algo_name in enumerate(y_algorithms_ordered):
                    if common_algo_name in point_estimates_setting and \
                       common_algo_name in interval_estimates_setting:
                        
                        point_val_arr = point_estimates_setting[common_algo_name]
                        interval_val_arr = interval_estimates_setting[common_algo_name]

                        if len(point_val_arr) > metric_idx_to_plot and interval_val_arr.shape[1] > metric_idx_to_plot:
                            point_val = point_val_arr[metric_idx_to_plot]
                            lower, upper = interval_val_arr[:, metric_idx_to_plot]
                            current_color = algo_to_color.get(common_algo_name, color_palette_default[alg_idx % len(color_palette_default)])

                            if not (np.isnan(lower) or np.isnan(upper)):
                                current_ax.barh(
                                    y=alg_idx, width=upper - lower, height=interval_height_val, left=lower,
                                    color=current_color, alpha=0.75)
                                all_lowers_subplot.append(lower)
                                all_uppers_subplot.append(upper)
                            if not np.isnan(point_val):
                                current_ax.vlines(
                                    x=point_val, ymin=alg_idx - (7.5 * interval_height_val / 16),
                                    ymax=alg_idx + (6 * interval_height_val / 16), color='k', alpha=0.5)
                                if not (np.isnan(lower) or np.isnan(upper)):
                                    all_lowers_subplot.append(min(lower, point_val))
                                    all_uppers_subplot.append(max(upper, point_val))
                                else: 
                                    all_lowers_subplot.append(point_val)
                                    all_uppers_subplot.append(point_val)
                        
                if current_ax == ax_mt10:
                    current_ax.set_yticklabels(y_algorithms_ordered, fontsize='x-large')
                else:
                    current_ax.set_yticklabels([]) 
            
            current_ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks_val))
            adjust_plot_xlimits(current_ax, all_lowers_subplot, all_uppers_subplot) 
            _decorate_axis(current_ax, ticklabelsize='x-large') 
            current_ax.spines['left'].set_visible(False) 
            current_ax.grid(True, axis='x', alpha=0.25) 

            # Add the group separator lines
            if line_y_after_g3_block != -1:
                current_ax.axhline(y=line_y_after_g3_block, color='gray', linestyle=':', linewidth=1)
            if line_y_after_g2_block != -1:
                current_ax.axhline(y=line_y_after_g2_block, color='gray', linestyle=':', linewidth=1)
        
        if y_algorithms_ordered: 
            ax_mt10.set_yticklabels(y_algorithms_ordered, fontsize='x-large')
        else: 
            ax_mt10.set_yticklabels([])
            ax_mt50.set_yticklabels([])


        fig.tight_layout(rect=[0.02, 0, 0.97, 0.95], w_pad=3.0, h_pad=1.5) 

        save_path_png = f"scripts/figures/CI/plot_CI_{metric_type}.png"
        save_path_pdf = f"scripts/figures/CI/plot_CI_{metric_type}.pdf"
        os.makedirs(os.path.dirname(save_path_png), exist_ok=True)
        fig.savefig(save_path_png, dpi=300)
        print(f"Custom combined rliable plot for {metric_type} saved to {save_path_png}")
        fig.savefig(save_path_pdf)
        plt.close(fig)

# --- Main Execution Logic ---
SETUP_INFO = {}
NUM_TASKS = 0
HORIZON = 0
LOG_DIR = ""
CURRENT_SETTING_NAME = ""

if __name__ == "__main__":
    LOG_DIR_DEFAULT = "/work/08962/vjoshi12/ls6/IsaacGymEnvs/runs/"
    if not os.path.exists(LOG_DIR_DEFAULT):
        LOG_DIR_DEFAULT = "./runs_default/"
        print(f"Warning: Default log directory '{LOG_DIR_DEFAULT}' from script not found, trying local '{LOG_DIR_DEFAULT}'")

    parser = argparse.ArgumentParser(description="Generate summary table from TensorBoard logs using non-stratified bootstrap on overall run metrics.")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR_DEFAULT, help="Root directory containing TensorBoard run folders.")
    parser.add_argument("--is_mt10", action="store_true", help="MT10 specific setup (NUM_TASKS=10). If not set, MT50 is assumed.")
    parser.add_argument("--horizon", type=int, default=32, help="Horizon value for frame calculation.")
    parser.add_argument("--target_frame", type=int, default=None, help="Specific frame count (e.g., 250000000) to evaluate metrics at. Overrides default per-setup max frame. Must be non-negative.")
    parser.add_argument("--combined_mt_plot", action="store_true", help="Generate a side-by-side plot for MT10 and MT50, in addition to individual plots. Also enables combined LaTeX table.")
    args = parser.parse_args()

    if args.target_frame is not None and args.target_frame < 0:
        parser.error("--target_frame must be a non-negative integer.")

    LOG_DIR = args.log_dir
    # HORIZON is set per setting later
    
    if LOG_DIR == "./runs_default/" and not os.path.exists(LOG_DIR):
        print(f"Warning: Fallback log directory '{LOG_DIR}' does not exist. Creating it.")
        os.makedirs(LOG_DIR, exist_ok=True)


    settings_to_process = []
    if args.combined_mt_plot: # If combined plot, assume we process both mt10 and mt50 for the table too
        settings_to_process = ["mt10", "mt50"]
    elif args.is_mt10:
        settings_to_process = ["mt10"]
    else:
        settings_to_process = ["mt50"]

    # Data storage for all processed settings
    all_settings_numerical_summaries = {}
    all_settings_processing_details = {}
    all_settings_rliable_data = {} # For plots
    all_settings_specific_setup_info = {} # For plots (SETUP_INFO per setting)
    
    if args.target_frame is not None:
        print(f"INFO: Using specified target frame for metrics: {args.target_frame:,}")
    else:
        print(f"INFO: Using per-setup maximum available frames for metrics (default behavior).")

    for setting_val in settings_to_process:
        CURRENT_SETTING_NAME = setting_val
        HORIZON = args.horizon # Global HORIZON for frame calculation, can be overridden if config has per-setting horizons
        
        print(f"\n\nProcessing data for setting: {CURRENT_SETTING_NAME.upper()}")
        print(f"Using setting: {CURRENT_SETTING_NAME}, Horizon: {HORIZON}, Log Dir: {LOG_DIR}")
        if args.target_frame is not None:
            print(f"Evaluating metrics at specified target frame: {args.target_frame:,}")
        else:
            print("Evaluating metrics using per-setup maximum available frames (default).")
        print("Bootstrap Method: Non-Stratified Overall on aggregate tags.")


        NUM_TASKS = 10 if CURRENT_SETTING_NAME == "mt10" else 50
        current_patterns = get_patterns_for_setting(CURRENT_SETTING_NAME)
        SETUP_INFO = get_setup_info_for_setting(CURRENT_SETTING_NAME, current_patterns) # SETUP_INFO is now correctly scoped per setting
        all_settings_specific_setup_info[CURRENT_SETTING_NAME] = SETUP_INFO.copy()


        results_data = defaultdict(lambda: { # This will be for the current setting_val
            'per_task_success': defaultdict(list), 'per_task_reward': defaultdict(list),
            'overall_avg': list(), 'avg_reward': list()
        })
        setup_run_counts = defaultdict(int) # Also for the current setting_val

        if not os.path.isdir(LOG_DIR):
            print(f"Error: Log directory '{LOG_DIR}' not found. Please check the path.");
            if len(settings_to_process) == 1: exit()
            else:
                print(f"Skipping setting {CURRENT_SETTING_NAME.upper()} due to missing log directory.")
                continue


        directories_to_process = [name for name in os.listdir(LOG_DIR) if os.path.isdir(os.path.join(LOG_DIR, name))]
        print(f"Found {len(directories_to_process)} potential run directories to scan in '{LOG_DIR}' for {CURRENT_SETTING_NAME.upper()}.")

        processed_runs_total = 0
        skipped_runs_count = 0
        
        # Prepare arguments for the multiprocessing worker
        worker_args = (LOG_DIR, SETUP_INFO, NUM_TASKS, HORIZON, CURRENT_SETTING_NAME)
        tasks = [(d, *worker_args) for d in directories_to_process]

        # Use a multiprocessing pool to load data in parallel
        with multiprocessing.Pool() as pool:
            results_iterator = pool.imap_unordered(process_run_directory_wrapper, tasks)
            
            for result in tqdm(results_iterator, total=len(tasks), desc=f"Scanning Dirs ({CURRENT_SETTING_NAME.upper()})", unit="dir", ncols=100, position=0):
                if result['status'] == 'processed':
                    setup_type = result['setup_type']
                    run_data = result['data']
                    
                    # Aggregate the data from the worker process
                    results_data[setup_type]['overall_avg'].extend(run_data.get('overall_avg', []))
                    results_data[setup_type]['avg_reward'].extend(run_data.get('avg_reward', []))
                    for task_idx, data_list in run_data.get('per_task_success', {}).items():
                        results_data[setup_type]['per_task_success'][task_idx].extend(data_list)
                    for task_idx, data_list in run_data.get('per_task_reward', {}).items():
                        results_data[setup_type]['per_task_reward'][task_idx].extend(data_list)

                    setup_run_counts[setup_type] += 1
                    processed_runs_total += 1
                else:
                    skipped_runs_count += 1
                    if result['status'] == 'skipped_env_mismatch':
                        tqdm.write(f"Skipping {result['item_name']}: Env count {result['num_envs']} != required {result['required']}.")
                    elif result['status'] not in ['skipped_no_match', 'skipped_no_config']:
                        # Optionally log other non-critical skips
                        # tqdm.write(f"Skipping {result['item_name']}: {result['status']}")
                        pass

        print(f"\n--- Processing Summary for {CURRENT_SETTING_NAME.upper()} ---")
        if processed_runs_total == 0 :
            print(f"No valid data loaded from any run for {CURRENT_SETTING_NAME.upper()}. Skipping tables and plots for this setting.");
            all_settings_numerical_summaries[CURRENT_SETTING_NAME] = {} # Mark as processed but empty data
            all_settings_rliable_data[CURRENT_SETTING_NAME] = extract_data_for_rliable_plot({}, defaultdict(int), SETUP_INFO)
            # Store details even if no data, for completeness if table function expects it
            all_settings_processing_details[CURRENT_SETTING_NAME] = {
                'num_tasks': NUM_TASKS, 'horizon': HORIZON,
                'setup_info': SETUP_INFO.copy(),
                'results_data': results_data.copy(), # Will be mostly empty
                'run_counts': setup_run_counts.copy() # Will be zeros
            }
            continue

        print(f"Successfully processed data from {processed_runs_total} runs for {CURRENT_SETTING_NAME.upper()}.")
        if skipped_runs_count > 0: print(f"Skipped {skipped_runs_count} directories/runs for {CURRENT_SETTING_NAME.upper()}.")
        for st, count in setup_run_counts.items():
            name = SETUP_INFO.get(st, {}).get('name', st); envs = SETUP_INFO.get(st, {}).get('required_envs', "N/A")
            print(f"        Setup '{name}' (Envs: {envs}): {count} runs processed for {CURRENT_SETTING_NAME.upper()}.")

        current_setting_summaries_for_all_stats = {}
        for stat_info_iter in STATISTICS_TO_REPORT:
            numerical_list = calculate_summary_data_for_setting(
                results_data, setup_run_counts,
                stat_info_iter["func"], stat_info_iter["short_name"],
                SETUP_INFO,
                args.target_frame,
                NUM_TASKS
            )
            current_setting_summaries_for_all_stats[stat_info_iter["short_name"]] = numerical_list
        all_settings_numerical_summaries[CURRENT_SETTING_NAME] = current_setting_summaries_for_all_stats
        
        all_settings_processing_details[CURRENT_SETTING_NAME] = {
            'num_tasks': NUM_TASKS, 'horizon': HORIZON,
            'setup_info': SETUP_INFO.copy(),
            'results_data': results_data.copy(),
            'run_counts': setup_run_counts.copy()
        }

        # Generate RLiable plot data and individual plots per setting
        # This uses master_numerical_summary_for_setting which is current_setting_summaries_for_all_stats
        generate_individual_rliable_plots(current_setting_summaries_for_all_stats, setup_run_counts, SETUP_INFO, CURRENT_SETTING_NAME, args)
        all_settings_rliable_data[CURRENT_SETTING_NAME] = extract_data_for_rliable_plot(current_setting_summaries_for_all_stats, setup_run_counts, SETUP_INFO)

    # --- After processing all settings, generate LaTeX table(s) ---
    if not all_settings_numerical_summaries:
        print("No data processed for any setting. Skipping final LaTeX table generation.")
    else:
        table_settings_to_generate = []
        if args.combined_mt_plot and 'mt10' in all_settings_numerical_summaries and 'mt50' in all_settings_numerical_summaries:
            # Ensure both mt10 and mt50 were actually processed and have summary entries
            if all_settings_numerical_summaries.get('mt10') and all_settings_numerical_summaries.get('mt50'):
                table_settings_to_generate.append(['mt10', 'mt50']) # Combined table
            else: # Fallback to individual if one is missing despite combined_mt_plot flag
                if 'mt10' in all_settings_numerical_summaries and all_settings_numerical_summaries.get('mt10'): table_settings_to_generate.append(['mt10'])
                if 'mt50' in all_settings_numerical_summaries and all_settings_numerical_summaries.get('mt50'): table_settings_to_generate.append(['mt50'])
                if not table_settings_to_generate: print("WARN: --combined_mt_plot was set, but data for one or both settings (MT10, MT50) is missing. No combined table.")

        else: # Individual tables for each processed setting
            for setting_name_processed in settings_to_process:
                if setting_name_processed in all_settings_numerical_summaries and all_settings_numerical_summaries.get(setting_name_processed):
                    table_settings_to_generate.append([setting_name_processed])
        
        if not table_settings_to_generate:
            print("No settings eligible for LaTeX table generation based on processed data.")

        for active_settings_list in table_settings_to_generate:
            # Assuming we generate one table per statistic defined (currently only 'mean')
            for stat_report_info in STATISTICS_TO_REPORT:
                # Check if all active settings for this table actually have data for this stat
                data_available_for_all_active = True
                for s_name in active_settings_list:
                    if not all_settings_numerical_summaries.get(s_name, {}).get(stat_report_info["short_name"]):
                        data_available_for_all_active = False
                        print(f"WARN: Data for statistic '{stat_report_info['short_name']}' missing for setting '{s_name}'. Skipping table for this stat combination.")
                        break
                if not data_available_for_all_active:
                    continue

                generate_latex_table_final_format(
                    all_settings_numerical_summaries,
                    all_settings_processing_details,
                    active_settings_list,
                    stat_report_info,
                    args.target_frame,
                    CI_PERCENT
                )

    # --- Generate combined RLiable plot if requested and data is available ---
    if args.combined_mt_plot:
        can_generate_combined_plot = False
        # Check if 'mt10' and 'mt50' data exists for rliable plots
        if 'mt10' in all_settings_rliable_data and 'mt50' in all_settings_rliable_data:
            for metric_key in ["success", "reward"]:
                if all_settings_rliable_data['mt10'].get(metric_key, {}).get("algos_ordered") and \
                   all_settings_rliable_data['mt50'].get(metric_key, {}).get("algos_ordered"):
                    can_generate_combined_plot = True
                    break
        
        if can_generate_combined_plot:
            generate_combined_rliable_plots(all_settings_rliable_data, all_settings_specific_setup_info, args)
        else:
            print("Could not generate combined rliable plot: Plottable algorithm data missing for MT10 and/or MT50.")

    print("\nAll requested processing complete!")