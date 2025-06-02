# Keep all previous code (imports, constants, helpers, plotting) ...
import os
from collections import defaultdict
import re
import math
import yaml

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# --- Configuration Constants ---
LOG_DIR = "runs/"
FILENAME_PATTERN = r"vanilla_grpo_mt\d+_rand_envs_(\d+)_seed_(\d+)_.*"
CONFIG_FILENAME_OPTIONS = ["config.yaml", "params.yaml"]
NUM_TASKS = 50
BASE_STEP_FRAMES = 150 # VERIFY THIS!
NUM_EPOCHS_FOR_FINAL_AVG = 5
FIGURES_DIR = "figures"
# Define the expected tag formats using f-string placeholders
TAG_FORMAT_SUCCESS = "Episode/task_{task_id}_success"
TAG_FORMAT_REWARD = "Episode/task_{task_id}_reward" # Added reward tag format
# Colors for different num_envs lines within a plot
ENV_PLOT_COLORS = {4096: 'darkblue', 8192: 'red', 'default': 'green'}


# --- Helper Function to Extract Seed/NumEnvs ---
# (No changes needed)
def extract_seed_envs_from_pattern(run_name):
    match = re.match(FILENAME_PATTERN, run_name)
    if match:
        num_parallel_envs = int(match.group(1))
        seed_number = int(match.group(2))
        if num_parallel_envs > 0 : return num_parallel_envs, seed_number
        else: return None, None
    else: return None, None

# --- Helper Function to Adjust Data ---
# (No changes needed - adjusts based on 'frame' column which reward DF will also have)
def adjust_data(df):
    # Note: This expects 'frame' and 'success' columns, but will work if 'success'
    # is replaced by 'reward' as long as 'frame' exists. Let's make it safer.
    value_col = 'success' if 'success' in df.columns else ('reward' if 'reward' in df.columns else None)
    if df.empty or not value_col: return pd.DataFrame({value_col or 'value': [0.0], 'frame': [0]})

    if 0 not in df['frame'].values:
        new_row = pd.DataFrame({value_col: [0.0], 'frame': [0]})
        result_df = pd.concat([new_row, df], ignore_index=True)
    # Check if frame 0 exists and its value isn't 0.0
    elif df.loc[df['frame'] == 0, value_col].iloc[0] != 0.0:
        df.loc[df['frame'] == 0, value_col] = 0.0 # Set value at frame 0 to 0
        result_df = df
    else:
        result_df = df # Frame 0 exists with 0 value
    result_df = result_df.sort_values('frame').reset_index(drop=True)
    return result_df


# --- Calculation Function: Average of Last N ---
def calculate_avg_last_n(df, value_column, n=NUM_EPOCHS_FOR_FINAL_AVG):
    """
    Calculates the average of a specified column over the last N data points.
    Args:
        df (pd.DataFrame): DataFrame containing sorted 'frame' and value_column.
        value_column (str): The name of the column to average ('success' or 'reward').
        n (int): Number of points to average.
    Returns:
        float: The average value, or np.nan if insufficient data.
    """
    if df.empty or value_column not in df.columns: return np.nan
    # Ensure sorting just in case (though should be done before calling)
    df_sorted = df.sort_values('frame')
    if len(df_sorted) == 0: return np.nan
    num_points_to_avg = min(n, len(df_sorted))
    last_n_values = df_sorted[value_column].tail(num_points_to_avg).values
    return np.nanmean(last_n_values)


# --- Plotting Function: Grid Plot of Task Success Curves ---
# (No changes needed here, plotting is only for success)
def plot_task_success_grid(results_dict_nested):
    # ... (plotting function remains the same as previous version) ...
    print("Generating grid plot of task success curves (grouped by num_envs)...")
    ncols = 8
    nrows = math.ceil(NUM_TASKS / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3), # Increased size
                             sharex=True, sharey=True, squeeze=False)
    fig.suptitle(f'Success Rate vs Frames per Task (Grouped by Num Parallel Envs)', fontsize=16, y=0.98)
    axes_flat = axes.flatten()

    plot_task_count = 0
    for task_id in range(NUM_TASKS): # Iterate 0-49 for consistent grid layout
        ax = axes_flat[task_id]
        ax.set_title(f"Task {task_id}", fontsize=10) # Set title early

        if task_id in results_dict_nested and results_dict_nested[task_id]:
            plot_task_count += 1
            task_data = results_dict_nested[task_id] # dict {num_envs: [list of DFs]}

            # Determine common frame axis
            all_dfs_for_task = [df for df_list in task_data.values() for df in df_list]
            if not all_dfs_for_task:
                 ax.text(0.5, 0.5, 'No Data Found', ha='center', va='center', transform=ax.transAxes, color='grey', fontsize=9); ax.axis('off'); continue
            all_dfs_adjusted = [adjust_data(df.copy()) for df in all_dfs_for_task if not df.empty]
            all_dfs_adjusted = [df for df in all_dfs_adjusted if len(df)>1]
            if not all_dfs_adjusted:
                 ax.text(0.5, 0.5, 'No Valid Data', ha='center', va='center', transform=ax.transAxes, color='grey', fontsize=9); ax.axis('off'); continue

            try:
                 max_frames_per_run = [df['frame'].max() for df in all_dfs_adjusted]
                 overall_max_frame = max(max_frames_per_run) if max_frames_per_run else 0
                 longest_run_df_overall = max(all_dfs_adjusted, key=lambda df: df['frame'].max())
                 ref_frames_full, _ = np.unique(longest_run_df_overall['frame'].values, return_index=True)
                 if len(ref_frames_full) < 2:
                     ax.text(0.5, 0.5, 'Interpolation Error', ha='center', va='center', transform=ax.transAxes, color='red', fontsize=8); ax.axis('off'); continue
            except ValueError:
                 ax.text(0.5, 0.5, 'Frame Axis Error', ha='center', va='center', transform=ax.transAxes, color='red', fontsize=8); ax.axis('off'); continue


            # Plot each Num Envs Group
            num_envs_groups_plotted = 0
            sorted_num_envs_keys = sorted(task_data.keys())
            for num_envs in sorted_num_envs_keys:
                run_data = task_data[num_envs]
                run_data_adjusted = [adjust_data(data.copy()) for data in run_data]
                run_data_adjusted = [df for df in run_data_adjusted if not df.empty and len(df) > 1 and 'success' in df.columns] # Ensure success col
                if not run_data_adjusted: continue

                interpolated_successes = []
                valid_run_count = 0
                for df in run_data_adjusted:
                    unique_frames, unique_indices = np.unique(df['frame'].values, return_index=True)
                    # Ensure we get success values here
                    if 'success' not in df.columns: continue
                    unique_success = df['success'].values[unique_indices]
                    if len(unique_frames) < 2: continue

                    interp_success = np.interp(ref_frames_full, unique_frames, unique_success)
                    if not np.isnan(interp_success).all():
                        interpolated_successes.append(interp_success)
                        valid_run_count += 1

                if not interpolated_successes: continue
                frame_successes = np.array(interpolated_successes)
                frame_mean = np.nanmean(frame_successes, axis=0)
                frame_std = np.nanstd(frame_successes, axis=0)

                color = ENV_PLOT_COLORS.get(num_envs, ENV_PLOT_COLORS['default'])
                label = f'{num_envs} Envs ({valid_run_count} seeds)'
                ax.plot(ref_frames_full, frame_mean, color=color, linewidth=1.5, label=label)
                ax.fill_between(ref_frames_full, frame_mean - frame_std, frame_mean + frame_std,
                                alpha=0.15, color=color)
                num_envs_groups_plotted += 1

            # Finalize Subplot
            if num_envs_groups_plotted > 0:
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                ax.set_ylim(-0.05, 1.05)
                if num_envs_groups_plotted > 1: ax.legend(fontsize=7)

                if task_id % ncols == 0:
                     for label in ax.get_yticklabels(): label.set_fontsize(8)
                else: plt.setp(ax.get_yticklabels(), visible=False)

                is_last_row = (task_id // ncols == nrows - 1)
                has_plot_below = any((t_id + ncols < NUM_TASKS and (t_id + ncols) in results_dict_nested) for t_id in range(task_id % ncols, task_id + 1, ncols) if t_id<NUM_TASKS)
                if not is_last_row and has_plot_below :
                     plt.setp(ax.get_xticklabels(), visible=False)
                else:
                    for label in ax.get_xticklabels(): label.set_fontsize(8); label.set_rotation(30); label.set_ha('right')
                    ax.xaxis.get_offset_text().set_fontsize(8)
            else:
                 ax.text(0.5, 0.5, 'Plotting Error', ha='center', va='center', transform=ax.transAxes, color='red', fontsize=8); ax.axis('off');

        else:
            ax.text(0.5, 0.5, 'No Data Found', ha='center', va='center', transform=ax.transAxes, color='grey', fontsize=9)
            ax.axis('off')

    fig.text(0.5, 0.01, f'Frames (Step * {BASE_STEP_FRAMES} * NumEnvs)', ha='center', va='center', fontsize=12)
    fig.text(0.015, 0.5, 'Average Task Success Rate', ha='center', va='center', rotation='vertical', fontsize=12)
    plt.tight_layout(rect=[0.04, 0.03, 1, 0.96])
    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_path_base = os.path.join(FIGURES_DIR, "task_success_grouped_grid")
    plt.savefig(f"{save_path_base}.pdf"); plt.savefig(f"{save_path_base}.png", dpi=300)
    print(f"Grid plot saved to {save_path_base}.png/.pdf")
    plt.close(fig)


# --- Function to Generate LaTeX Summary Table (Success% and Reward) ---
def generate_latex_summary_table_combined(stats_by_env):
    """
    Generates a 3-column LaTeX tabular environment summarizing overall success rate
    (as percentage XX.XX ± X.XX) and overall average reward (as Mean ± SEM).

    Args:
        stats_by_env (dict): A dictionary where keys are num_envs and values are
                             dicts containing 'mean', 'sem', 'n' for success
                             and optionally 'mean_reward', 'sem_reward' for reward.

    Returns:
        str: A string containing the LaTeX table code.
    """
    if not stats_by_env:
        return "% No data available to generate summary table."

    # 'l': Num Envs (left), 'r': Combined Success (right), 'r': Combined Reward (right)
    latex_string = "\\begin{tabular}{lrr}\n"
    latex_string += "\\hline\n"
    # Updated header for the combined columns
    latex_string += "\\textbf{Num Envs} & \\textbf{Avg Success (\\%)} & \\textbf{Avg Reward} \\\\\n"
    latex_string += "\\hline\n"

    # Sort by num_envs for consistent table order
    sorted_envs = sorted(stats_by_env.keys())

    for num_envs in sorted_envs:
        stats = stats_by_env[num_envs]
        n_success = stats.get('n_success', 0) # Get count for success
        n_reward = stats.get('n_reward', 0)   # Get count for reward

        # --- Success Formatting ---
        mean_success_val = stats.get('mean_success', np.nan)
        sem_success_val = stats.get('sem_success', np.nan)
        mean_success_percent = mean_success_val * 100.0
        sem_success_percent = sem_success_val * 100.0
        mean_success_str = f"{mean_success_percent:.2f}" if not np.isnan(mean_success_percent) else "N/A"
        sem_success_str = f"{sem_success_percent:.2f}" if not np.isnan(sem_success_percent) else "N/A"
        if mean_success_str != "N/A" and sem_success_str != "N/A":
             combined_success_str = f"{mean_success_str} \\pm {sem_success_str}"
        elif mean_success_str != "N/A":
             combined_success_str = mean_success_str
        else: combined_success_str = "N/A"
        # --- End Success Formatting ---

        # --- Reward Formatting ---
        mean_reward_val = stats.get('mean_reward', np.nan)
        sem_reward_val = stats.get('sem_reward', np.nan)
        # Format reward (e.g., 1 decimal place) - adjust format as needed
        mean_reward_str = f"{mean_reward_val:.1f}" if not np.isnan(mean_reward_val) else "N/A"
        sem_reward_str = f"{sem_reward_val:.1f}" if not np.isnan(sem_reward_val) else "N/A"
        if mean_reward_str != "N/A" and sem_reward_str != "N/A":
             combined_reward_str = f"{mean_reward_str} \\pm {sem_reward_str}"
        elif mean_reward_str != "N/A":
             combined_reward_str = mean_reward_str
        else: combined_reward_str = "N/A"
        # --- End Reward Formatting ---

        num_envs_str = str(num_envs).replace("_", "\\_")

        # Add the combined strings, enclosed in $...$ for math mode (\pm)
        latex_string += f"{num_envs_str} & ${combined_success_str}$ & ${combined_reward_str}$ \\\\\n"

    latex_string += "\\hline\n"
    latex_string += "\\end{tabular}"

    # Optional: Add note about N (number of seed-task averages) if desired
    # latex_string += f"\n% N_success={ {ne: stats_by_env[ne]['n_success'] for ne in sorted_envs} }"
    # latex_string += f"\n% N_reward={ {ne: stats_by_env[ne]['n_reward'] for ne in sorted_envs} }"

    return latex_string

# --- Main Execution Logic ---
if __name__ == "__main__":
    # Success DataFrames: results_dataframes[task_id][num_envs] = [List of DFs]
    results_dataframes = defaultdict(lambda: defaultdict(list))
    # Success last N averages: task_env_seed_last_epochs_avg[task_id][num_envs] = [List of avgs]
    task_env_seed_last_epochs_avg = defaultdict(lambda: defaultdict(list))
    # REWARD last N averages: task_env_seed_last_epochs_reward_avg[task_id][num_envs] = [List of avgs]
    task_env_seed_last_epochs_reward_avg = defaultdict(lambda: defaultdict(list)) # New dict for reward


    # ... (Initial print statements, scanning dir logic) ...
    print(f"Scanning directory: {LOG_DIR}")
    print(f"Run folder pattern: '{FILENAME_PATTERN}'")
    print(f"Config filenames: {CONFIG_FILENAME_OPTIONS}")
    print(f"Base step frames (per env): {BASE_STEP_FRAMES}")
    print(f"Success Tag format: '{TAG_FORMAT_SUCCESS}'")
    print(f"Reward Tag format: '{TAG_FORMAT_REWARD}'") # Added reward tag info
    print(f"Averaging last {NUM_EPOCHS_FOR_FINAL_AVG} epochs for final success/reward report.")
    print("-" * 30)

    processed_runs_count = 0
    processed_datapoints_count = 0 # Combined count for simplicity
    found_task_ids = set()
    found_num_envs = set()


    for item_name in os.listdir(LOG_DIR):
        item_path = os.path.join(LOG_DIR, item_name)
        if not os.path.isdir(item_path): continue

        num_parallel_envs, seed_number = extract_seed_envs_from_pattern(item_name)
        if num_parallel_envs is None: continue

        found_num_envs.add(num_parallel_envs)
        current_frame_multiplier = BASE_STEP_FRAMES * num_parallel_envs

        # Find Config File & Load Task IDs
        config_file_path = None
        # ... (config finding logic - unchanged) ...
        for name in CONFIG_FILENAME_OPTIONS:
             cp_try = os.path.join(item_path, name);
             if os.path.isfile(cp_try): config_file_path = cp_try; break
        if not config_file_path: continue
        try:
            with open(config_file_path, "r") as f: config = yaml.safe_load(f)
            if config is None or "task_id" not in config or not isinstance(config["task_id"], list) or not config["task_id"]: continue
            task_ids_for_this_run = config["task_id"]
        except Exception as e: continue

        # Find Event File
        event_file_path = None
        # ... (event file finding logic - unchanged) ...
        potential_event_dirs = [item_path, os.path.join(item_path, "summaries")]
        for event_dir in potential_event_dirs:
             if os.path.isdir(event_dir):
                  try:
                      event_files = [f for f in os.listdir(event_dir) if f.startswith("events.out.tfevents")]
                      if event_files: event_file_path = os.path.join(event_dir, event_files[0]); break
                  except OSError: continue
        if not event_file_path: continue

        print(f"Processing Run: '{item_name}' (Seed: {seed_number}, NumEnvs: {num_parallel_envs}, Tasks: {task_ids_for_this_run})")
        processed_runs_count += 1

        try:
            ea = event_accumulator.EventAccumulator(event_file_path, size_guidance={'scalars': 0}, purge_orphaned_data=True)
            ea.Reload()
            available_tags = set(ea.Tags().get("scalars", []))

            for task_id in task_ids_for_this_run:
                found_task_ids.add(task_id) # Track task id found

                # --- Process SUCCESS ---
                success_tag = TAG_FORMAT_SUCCESS.format(task_id=task_id)
                if success_tag in available_tags:
                    values = ea.Scalars(success_tag)
                    if values and all(hasattr(v, 'step') and hasattr(v, 'value') for v in values):
                        df_success = pd.DataFrame({
                            'success': [v.value for v in values],
                            'step': [v.step for v in values],
                            'frame': [v.step * current_frame_multiplier for v in values]
                        })
                        df_success.dropna(subset=['success', 'frame'], inplace=True)
                        df_success = df_success[df_success['success'] >= 0]
                        df_success.drop_duplicates(subset=['frame'], keep='last', inplace=True)
                        df_success.sort_values(by='frame', inplace=True)

                        if not df_success.empty:
                            # Store success DF for plotting
                            results_dataframes[task_id][num_parallel_envs].append(df_success)
                            # Calculate and store last N success avg
                            avg_last_n_succ = calculate_avg_last_n(df_success, 'success', n=NUM_EPOCHS_FOR_FINAL_AVG)
                            if not pd.isna(avg_last_n_succ):
                                task_env_seed_last_epochs_avg[task_id][num_parallel_envs].append(avg_last_n_succ)
                            processed_datapoints_count += len(df_success)


                # --- Process REWARD ---
                reward_tag = TAG_FORMAT_REWARD.format(task_id=task_id)
                if reward_tag in available_tags:
                    values = ea.Scalars(reward_tag)
                    # Basic validation on reward values
                    if values and all(hasattr(v, 'step') and hasattr(v, 'value') for v in values):
                        df_reward = pd.DataFrame({
                            'reward': [v.value for v in values],
                            'step': [v.step for v in values],
                            'frame': [v.step * current_frame_multiplier for v in values]
                        })
                        df_reward.dropna(subset=['reward', 'frame'], inplace=True)
                        # No non-negativity constraint typically needed for reward
                        df_reward.drop_duplicates(subset=['frame'], keep='last', inplace=True)
                        df_reward.sort_values(by='frame', inplace=True)

                        if not df_reward.empty:
                            # Calculate and store last N reward avg
                            avg_last_n_rew = calculate_avg_last_n(df_reward, 'reward', n=NUM_EPOCHS_FOR_FINAL_AVG)
                            if not pd.isna(avg_last_n_rew):
                                task_env_seed_last_epochs_reward_avg[task_id][num_parallel_envs].append(avg_last_n_rew)
                            # Note: We don't store the full reward DF unless needed for plotting later
                            # processed_datapoints_count += len(df_reward) # Avoid double counting steps

        except Exception as e_outer:
            print(f"  ERROR processing run '{item_name}': {e_outer}")


    print("-" * 30)
    # ... (summary print statements: runs processed, tasks found, envs found) ...
    print(f"Found data for {len(found_task_ids)} distinct tasks: {sorted(list(found_task_ids))}")
    print(f"Found Num Envs values: {sorted(list(found_num_envs))}")
    print(f"Total success datapoints loaded: {processed_datapoints_count}") # Reflects success points primarily
    print("-" * 30)

    # --- Calculate and Print Per-Task/Per-Env DETAILED Report (Optional) ---
    # ... (Keep or remove the detailed print block as desired) ...


    # --- Generate Success Plots ---
    if not results_dataframes: print("\nNo success data loaded for plotting.")
    else:
        valid_results = {task: env_dict for task, env_dict in results_dataframes.items() if env_dict}
        if valid_results: plot_task_success_grid(valid_results); print("\nPlotting complete!")
        else: print("\nNo valid success data for plotting.")


    # --- Aggregate Overall Results (Success and Reward) ---
    overall_success_per_env = defaultdict(list)
    overall_reward_per_env = defaultdict(list) # New dict for reward aggregation

    # Aggregate Success
    if task_env_seed_last_epochs_avg:
        print("\nAggregating SUCCESS results for overall summary table...")
        for task_id, env_dict in task_env_seed_last_epochs_avg.items():
            for num_envs, seed_avg_list in env_dict.items():
                valid_averages = [avg for avg in seed_avg_list if not pd.isna(avg)]
                if valid_averages: overall_success_per_env[num_envs].extend(valid_averages)

    # Aggregate Reward
    if task_env_seed_last_epochs_reward_avg:
        print("Aggregating REWARD results for overall summary table...")
        for task_id, env_dict in task_env_seed_last_epochs_reward_avg.items():
            for num_envs, seed_avg_list in env_dict.items():
                valid_averages = [avg for avg in seed_avg_list if not pd.isna(avg)]
                if valid_averages: overall_reward_per_env[num_envs].extend(valid_averages)


    # --- Calculate Final Overall Stats per Num Envs (Success and Reward) ---
    final_stats_by_env_summary = defaultdict(dict) # Use regular dict inside now
    print("Calculating final statistics for summary table...")
    # Use union of keys found in both success and reward aggregations
    all_found_num_envs = set(overall_success_per_env.keys()) | set(overall_reward_per_env.keys())

    for num_envs in sorted(list(all_found_num_envs)):
        stats = {}
        # Calculate Success Stats
        if num_envs in overall_success_per_env and overall_success_per_env[num_envs]:
            all_success_averages = overall_success_per_env[num_envs]
            n_success = len(all_success_averages)
            mean_success = np.mean(all_success_averages)
            std_success = np.std(all_success_averages)
            sem_success = std_success / np.sqrt(n_success) if n_success > 1 else np.nan
            stats['mean_success'] = mean_success
            stats['sem_success'] = sem_success
            stats['n_success'] = n_success
            print(f"  Num Envs: {num_envs} (Success) -> Mean: {mean_success:.4f}, SEM: {sem_success:.4f}, N: {n_success}")
        else:
            stats['mean_success'] = np.nan; stats['sem_success'] = np.nan; stats['n_success'] = 0
            print(f"  Num Envs: {num_envs} (Success) -> No valid data.")

        # Calculate Reward Stats
        if num_envs in overall_reward_per_env and overall_reward_per_env[num_envs]:
            all_reward_averages = overall_reward_per_env[num_envs]
            n_reward = len(all_reward_averages)
            mean_reward = np.mean(all_reward_averages)
            std_reward = np.std(all_reward_averages)
            sem_reward = std_reward / np.sqrt(n_reward) if n_reward > 1 else np.nan
            stats['mean_reward'] = mean_reward
            stats['sem_reward'] = sem_reward
            stats['n_reward'] = n_reward
            print(f"  Num Envs: {num_envs} (Reward)  -> Mean: {mean_reward:.2f}, SEM: {sem_reward:.2f}, N: {n_reward}")
        else:
            stats['mean_reward'] = np.nan; stats['sem_reward'] = np.nan; stats['n_reward'] = 0
            print(f"  Num Envs: {num_envs} (Reward)  -> No valid data.")

        final_stats_by_env_summary[num_envs] = stats


    # --- Generate and Print LaTeX Table ---
    if final_stats_by_env_summary:
        print("\n--- LaTeX Summary Table (Success% and Reward) ---")
        latex_table_code = generate_latex_summary_table_combined(final_stats_by_env_summary)
        print(latex_table_code)
        print("---------------------------------------------------\n")
    else:
         print("\nSkipping overall summary table generation as no final statistics were calculated.")