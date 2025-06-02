import os
import re
import yaml # For reading config files
import traceback # For detailed error printing

import pandas as pd
import numpy as np # Make sure numpy is imported
from scipy.stats import trim_mean # For IQM
from tensorboard.backend.event_processing import event_accumulator
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # For color mapping

# --- Helper Function to Extract Setup Info from Filename ---
def extract_setup_info(run_name, setup_info_dict):
    """
    Extracts setup type, number of environments, and seed number from a run name.
    Args:
        run_name (str): The name of the run directory.
        setup_info_dict (dict): The SETUP_INFO dictionary.
    Returns:
        tuple: (setup_type, num_envs, seed_number) or (None, None, None) if no match.
    """
    for setup_type_key, info in setup_info_dict.items():
        match = re.search(info['pattern'], run_name)
        if match:
            try:
                num_envs = int(match.group(1))
                seed_number = int(match.group(2))
                return setup_type_key, num_envs, seed_number
            except IndexError:
                tqdm.write(f"Warning: Regex pattern for {setup_type_key} may not have enough capture groups: {info['pattern']}")
                return None, None, None
            except ValueError:
                tqdm.write(f"Warning: Could not convert num_envs or seed to int for {run_name} with pattern {info['pattern']}")
                return None, None, None
    return None, None, None

def generate_metric_matrix_dict(eval_df, metric_type: str):
    """
    Generates a dictionary of metric matrices from eval_df based on the specified metric_type.
    Each matrix has tasks as rows and runs (seeds) as columns.
    Only data from the last env_step for each experiment is considered.

    Args:
    - eval_df (pandas DataFrame): DataFrame containing the evaluation data.
    - metric_type (str): The metric to be extracted (e.g., 'task_success_rate').

    Returns:
    - metric_matrix_dict (dict): Dictionary where keys are experiment names
                                 and values are 2D numpy arrays representing metric matrices.
    """
    filtered_df = eval_df[eval_df["metric"] == metric_type]
    metric_matrix_dict = {}
    exp_names = filtered_df["exp_name"].unique()

    for exp_name in exp_names:
        exp_df = filtered_df[filtered_df["exp_name"] == exp_name]
        if exp_df.empty:
            continue
        last_env_step = exp_df["env_step"].max()
        exp_data_last_step = exp_df[exp_df["env_step"] == last_env_step]

        if exp_data_last_step.empty:
            continue

        pivot_table = exp_data_last_step.pivot_table(
            values="value", index="env_name", columns="seed", aggfunc="first"
        )
        metric_matrix = pivot_table.to_numpy()

        if np.isnan(metric_matrix).any():
            nan_cols = np.any(np.isnan(metric_matrix), axis=0)
            if np.all(nan_cols):
                continue
            metric_matrix = metric_matrix[:, ~nan_cols]

        if metric_matrix.shape[1] == 0:
            continue
        metric_matrix_dict[exp_name] = metric_matrix
    return metric_matrix_dict

# --- Metric Calculation Functions ---
def calculate_iqm(data):
    if data is None or len(data) == 0:
        return np.nan
    return trim_mean(data, 0.25)

def calculate_median(data):
    if data is None or len(data) == 0:
        return np.nan
    return np.median(data)

def calculate_mean(data):
    if data is None or len(data) == 0:
        return np.nan
    return np.mean(data)

# --- Stratified Bootstrap Function ---
def stratified_bootstrap_ci(data_matrix, metric_func, n_bootstrap_samples=2000, alpha=0.05):
    if data_matrix is None or data_matrix.ndim != 2 or data_matrix.shape[0] == 0 or data_matrix.shape[1] == 0:
        return (np.nan, np.nan), np.nan

    n_tasks, n_runs = data_matrix.shape 
    bootstrap_metric_values = []
    original_run_scores = np.mean(data_matrix, axis=0)
    point_estimate = metric_func(original_run_scores)

    for _ in range(n_bootstrap_samples):
        bootstrap_sample_matrix = np.zeros((n_tasks, n_runs))
        for i in range(n_tasks):
            task_scores = data_matrix[i, :]
            resampled_indices = np.random.choice(n_runs, size=n_runs, replace=True) # among all the runs for this task
            bootstrap_sample_matrix[i, :] = task_scores[resampled_indices] 

        bootstrap_run_scores = np.mean(bootstrap_sample_matrix, axis=0)
        metric_val = metric_func(bootstrap_run_scores)

        bootstrap_metric_values.append(metric_val)

    lower_bound = np.percentile(bootstrap_metric_values, (alpha / 2) * 100)
    upper_bound = np.percentile(bootstrap_metric_values, (1 - alpha / 2) * 100)
    return (lower_bound, upper_bound), point_estimate

def plot_confidence_intervals(algorithms, all_point_estimates, all_confidence_intervals,
                              metric_names, plot_title, colors_dict, xlabel_name,
                              output_filename="aggregate_scores_plot.png"):
    num_metrics = len(metric_names)
    num_algorithms = len(algorithms)

    if num_algorithms == 0:
        print(f"No algorithms to plot for {plot_title}. Skipping plot generation.")
        return

    fig_height = max(4, num_algorithms * 0.7 + 2)
    # Removed sharey=True
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, fig_height))
    if num_metrics == 1:
        axes = [axes] # Ensure axes is always a list

    fig.suptitle(plot_title, fontsize=16, y=1.03)

    for i, metric_name in enumerate(metric_names):
        ax = axes[i]
        ax.set_title(metric_name, fontsize=14)

        y_pos = np.arange(num_algorithms)

        point_values = [all_point_estimates.get(algo, {}).get(metric_name, np.nan) for algo in algorithms]
        ci_lowers = [all_confidence_intervals.get(algo, {}).get(metric_name, (np.nan, np.nan))[0] for algo in algorithms]
        ci_uppers = [all_confidence_intervals.get(algo, {}).get(metric_name, (np.nan, np.nan))[1] for algo in algorithms]

        valid_indices = [idx for idx, (l, u) in enumerate(zip(ci_lowers, ci_uppers)) if not (np.isnan(l) or np.isnan(u))]

        # Plot data first
        for idx_data in valid_indices:
            algo = algorithms[idx_data]
            lower = ci_lowers[idx_data]
            upper = ci_uppers[idx_data]
            bar_width = upper - lower
            ax.barh(y_pos[idx_data], bar_width, left=lower, height=0.7, align='center',
                    color=colors_dict.get(algo, '#808080'), alpha=0.75, zorder=2)
            if not np.isnan(point_values[idx_data]):
                ax.vlines(point_values[idx_data], y_pos[idx_data] - 0.4, y_pos[idx_data] + 0.4,
                          color='black', linestyle='-', linewidth=2.0, zorder=3)

        # Handle "No data" text
        if not valid_indices and num_algorithms > 0:
            ax.text(0.5, 0.5, "No data for this metric", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)

        # --- Y-Tick and Label Handling (No sharey) ---
        ax.set_yticks(y_pos) # Set tick positions for all axes for alignment

        if i == 0: # Leftmost plot
            current_labels = algorithms if algorithms else [''] * num_algorithms
            ax.set_yticklabels(current_labels) # Set labels only for the first plot
            # Ensure left ticks and labels are on for the first plot
            ax.tick_params(axis='y', labelsize=12, left=True, labelleft=True)
        else: # Other plots
            ax.set_yticklabels([]) # No labels
            # Turn off left ticks and labels for subsequent plots
            ax.tick_params(axis='y', left=False, labelleft=False)
        # --- End Y-Tick and Label Handling ---

        ax.invert_yaxis()
        ax.grid(axis='x', linestyle=':', linewidth=0.7, color='gray', zorder=0)
        ax.set_xlabel(xlabel_name, fontsize=12)
        ax.tick_params(axis='x', labelsize=10)

        # X-axis limits
        current_metric_vals_for_xlim = []
        for idx_val in valid_indices:
            current_metric_vals_for_xlim.append(ci_lowers[idx_val])
            current_metric_vals_for_xlim.append(ci_uppers[idx_val])
            if not np.isnan(point_values[idx_val]):
                current_metric_vals_for_xlim.append(point_values[idx_val])

        if current_metric_vals_for_xlim:
            min_val = np.nanmin(current_metric_vals_for_xlim)
            max_val = np.nanmax(current_metric_vals_for_xlim)
            if not (np.isnan(min_val) or np.isnan(max_val)):
                padding = (max_val - min_val) * 0.05 if (max_val - min_val) > 1e-9 else 0.1
                lower_x_limit = min_val - padding
                calculated_upper_x_limit = max_val + padding
                # Ensure upper_x_limit is at least 1.0, but can be higher if calculated_upper_x_limit is greater
                upper_x_limit = max(1.0 if max_val <= 1.0 and calculated_upper_x_limit <=1.0 else calculated_upper_x_limit, calculated_upper_x_limit)

                ax.set_xlim(lower_x_limit, upper_x_limit)
            else:
                ax.set_xlim(0, 1.0)
        else:
            ax.set_xlim(0, 1.0)

    # Spines and final layout adjustments
    for idx, ax_ in enumerate(axes):
        for spine_loc in ['top', 'right']:
            ax_.spines[spine_loc].set_visible(False)

        if idx == 0: # Leftmost plot
            ax_.spines['left'].set_linewidth(1.5)
            ax_.spines['left'].set_visible(True)
            ax_.yaxis.set_ticks_position('left')
        else: # Other plots
            ax_.spines['left'].set_visible(False) # Hide left spine for subsequent plots

        ax_.spines['bottom'].set_linewidth(1.5)
        ax_.xaxis.set_ticks_position('bottom')


    try:
        # Adjusted rect to give more space for y-labels on the first plot if needed
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    except Exception as e:
        print(f"Warning: tight_layout encountered an issue: {e}. Proceeding with fallback adjustments.")
        try:
            # Adjust left margin more if tight_layout fails, to accommodate y-axis labels
            fig.subplots_adjust(top=0.9, bottom=0.15, left=0.2 if num_metrics > 1 else 0.15, right=0.95, hspace=0.2, wspace=0.1) # Reduced wspace
        except Exception as e2:
            print(f"Warning: subplots_adjust also failed: {e2}")

    try:
        if output_filename:
            dirname = os.path.dirname(output_filename)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            fig.savefig(output_filename, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to {output_filename}")
    except Exception as e:
        print(f"Error saving plot to {output_filename}: {e}")
    finally:
        plt.close(fig)

# --- Main Execution Logic ---
if __name__ == "__main__":
    LOG_DIR_DEFAULT = "/work/08962/vjoshi12/ls6/IsaacGymEnvs/runs/"
    parser = argparse.ArgumentParser(description="Extract per-task TensorBoard scalar data, calculate aggregate metrics with bootstrap CIs, and plot.")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR_DEFAULT, help="Root directory containing TensorBoard run folders.")
    parser.add_argument("--is_mt10", action="store_true", help="MT10 specific setup. Affects 'setting' variable and NUM_TASKS_CONFIG.")
    parser.add_argument("--horizon", type=int, default=32, help="Horizon value used in 'env_step' calculation (step * num_envs * HORIZON).")
    parser.add_argument("--bootstrap_reps", type=int, default=2000, help="Number of bootstrap repetitions for CI calculation.")
    parser.add_argument("--confidence_level", type=float, default=0.95, help="Confidence level for CI (e.g., 0.95 for 95% CI).")
    parser.add_argument("--output_dir", type=str, default="scripts/figures/custom_bootstrap", help="Directory to save plots.")
    args = parser.parse_args()

    LOG_DIR = args.log_dir
    HORIZON = args.horizon
    N_BOOTSTRAP_SAMPLES = args.bootstrap_reps
    ALPHA = 1.0 - args.confidence_level
    OUTPUT_PLOT_DIR = args.output_dir

    if args.is_mt10:
        setting = "mt10"
        NUM_TASKS_CONFIG = 10
    else:
        setting = "mt50"
        NUM_TASKS_CONFIG = 50

    print(f"--- Configuration ---")
    print(f"Setting: {setting.upper()}")
    print(f"Log Directory: {LOG_DIR}")
    print(f"Horizon for env_step calculation: {HORIZON}")
    print(f"Number of Tasks Default: {NUM_TASKS_CONFIG}")
    print(f"Bootstrap Repetitions: {N_BOOTSTRAP_SAMPLES}")
    print(f"Confidence Level: {(1-ALPHA)*100:.1f}% (Alpha: {ALPHA:.3f})")
    print(f"Output Plot Directory: {OUTPUT_PLOT_DIR}")
    print(f"---------------------\n")

    # --- PATTERNS (User Provided) ---
    ### VANILLA ###
    PATTERN_SHPPO_VANILLA = rf"05_09_ppo_vanilla_{setting}_rand_envs_(\d+)_seed_(\d+).*"
    PATTERN_MHPPO_VANILLA = rf"mhppo_vanilla_{setting}_rand_envs_(\d+)_seed_(\d+).*"
    ### Grad Mani ###
    PATTERN_FAMO = rf"05_09_ppo_famo_{setting}_rand_envs_(\d+)_seed_(\d+).*"
    PATTERN_PCGRAD = rf"05_11_ppo_pcgrad_{setting}_rand_envs_(\d+)_seed_(\d+).*"
    PATTERN_CAGRAD = rf"ppo_cagrad_{setting}_rand_envs_(\d+)_seed_(\d+).*" 
    ### Neural Architecture ###
    PATTERN_SHPPO_PACO = rf"05_16_shppo_paco_{setting}_rand_envs_(\d+)_seed_(\d+).*" 
    PATTERN_MHPPO_MOORE = rf"mhppo_moore_{setting}_rand_envs_(\d+)_seed_(\d+).*"
    PATTERN_SHPPO_MOORE = rf"shppo_moore_{setting}_rand_envs_(\d+)_seed_(\d+).*"
    PATTERN_SOFT_MODULARIZATION = rf"ppo_soft_modularization_{setting}_rand_envs_(\d+)_seed_(\d+).*" 
    PATTERN_MHPPO_CARE = rf"mhppo_care_{setting}_rand_envs_(\d+)_seed_(\d+).*"
    PATTERN_SHPPO_CARE = rf"shppo_care_{setting}_rand_envs_(\d+)_seed_(\d+).*"

    SETUP_INFO = {
        ### VANILLA ###
        'shppo_vanilla': {'pattern': PATTERN_SHPPO_VANILLA, 'name': 'SH-Vanilla', 'required_envs': 24576,
                          'color': '#C8A2C8'},
        # 'mhppo_vanilla': {'pattern': PATTERN_MHPPO_VANILLA, 'color': target_colors.get('MH-Vanilla', '#2ca02c'), 'name': 'MH-Vanilla', 'required_envs': 24576},
        ### Grad Mani ###
        'famo': {'pattern': PATTERN_FAMO, 'name': 'FAMO', 'required_envs': 24576,
                 'color': '#5AA0D5'},
        'pcgrad': {'pattern': PATTERN_PCGRAD, 'name': 'PCGrad', 'required_envs': 24576,
                   'color': '#FF9966'},
                   # 'cagrad': {'pattern': PATTERN_CAGRAD, 'color': '#1f77b4', 'name': 'CAGrad', 'required_envs': 24576},
        ### Neural Architecture ###
        'mhppo_care': {'pattern': PATTERN_MHPPO_CARE, 'name': 'MH-CARE', 'required_envs': 24576,
                       'color': '#E8C050'},
        'shppo_care': {'pattern': PATTERN_SHPPO_CARE, 'name': 'SH-CARE', 'required_envs': 24576,
                       'color': '#5FBF9B'},
        'shppo_moore': {'pattern': PATTERN_SHPPO_MOORE, 'name': 'SH-MOORE', 'required_envs': 24576,
                        'color': '#E57F3A'},
        # 'mhppo_moore': {'pattern': PATTERN_MHPPO_MOORE, 'color': target_colors.get('MH-MOORE', '#00C853'), 'name': 'MH-MOORE', 'required_envs': 24576},
        # 'shppo_paco': {'pattern': PATTERN_SHPPO_PACO, 'color': target_colors.get('SH-PACO', '#FF5722'), 'name': 'SH-PACO', 'required_envs': 24576},
        # 'soft_modularization': {'pattern': PATTERN_SOFT_MODULARIZATION, 'color': target_colors.get('Soft Modularization', '#D500F9'), 'name': 'Soft Modularization', 'required_envs': 24576},
    }

    # This will be used by the plotting function, derived directly from the updated SETUP_INFO
    plot_colors_final = {info['name']: info['color'] for _, info in SETUP_INFO.items()}


    CONFIG_FILENAME = "config.yaml"
    all_records = []

    if not os.path.isdir(LOG_DIR):
        print(f"Error: Log directory '{LOG_DIR}' not found.")
        exit()

    directory_contents = os.listdir(LOG_DIR)
    directories_to_process = [name for name in directory_contents if os.path.isdir(os.path.join(LOG_DIR, name))]
    print(f"Found {len(directories_to_process)} potential run directories to scan in '{LOG_DIR}'.")

    processed_runs_count = 0
    skipped_runs_count = 0

    for item_name in tqdm(directories_to_process, desc="Scanning Log Dirs", unit="dir", ncols=100, leave=False):
        item_path = os.path.join(LOG_DIR, item_name)
        setup_type, num_envs_from_filename, seed_from_filename = extract_setup_info(item_name, SETUP_INFO)

        if setup_type is None:
            skipped_runs_count += 1
            continue

        current_exp_config = SETUP_INFO.get(setup_type)
        if not current_exp_config: # Should not happen if extract_setup_info worked
            skipped_runs_count += 1
            continue

        exp_name_from_setup = current_exp_config['name']
        required_env_count = current_exp_config.get('required_envs')
        if required_env_count is not None and num_envs_from_filename != required_env_count:
            skipped_runs_count += 1
            continue

        task_ids_from_config = list(range(NUM_TASKS_CONFIG)) # Default
        config_path = os.path.join(item_path, CONFIG_FILENAME)
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r") as f:
                    config_yaml = yaml.safe_load(f)
                loaded_task_ids_raw = config_yaml.get('task_id', list(range(NUM_TASKS_CONFIG))) # Use default if not in yaml
                if isinstance(loaded_task_ids_raw, list) and all(isinstance(tid, int) for tid in loaded_task_ids_raw):
                    if NUM_TASKS_CONFIG > 0 and not loaded_task_ids_raw: # Empty list in yaml
                        pass # Keep default (range(NUM_TASKS_CONFIG))
                    elif loaded_task_ids_raw: # Non-empty list in yaml
                        task_ids_from_config = loaded_task_ids_raw
                # If not list or not all ints, or NUM_TASKS_CONFIG is 0, keep default
            except Exception: # Error reading yaml
                pass # Keep default
        # If config file not found, task_ids_from_config remains default

        event_file_path = None
        summaries_dir = os.path.join(item_path, "summaries")
        possible_dirs = []
        if os.path.isdir(summaries_dir): possible_dirs.append(summaries_dir)
        if os.path.isdir(item_path): possible_dirs.append(item_path)

        for p_dir_opt in possible_dirs: # Find newest event file
            if os.path.isdir(p_dir_opt):
                try:
                    event_files = [f for f in os.listdir(p_dir_opt) if f.startswith("events.out.tfevents")]
                    if event_files:
                        # Sort by modification time to get the latest one, robust to naming
                        event_files.sort(key=lambda f: os.path.getmtime(os.path.join(p_dir_opt, f)))
                        event_file_path = os.path.join(p_dir_opt, event_files[-1])
                        break
                except Exception: pass # Ignore errors listing/accessing files, try next dir
        
        if not event_file_path:
            skipped_runs_count += 1
            continue

        run_data_extracted_flag = False
        try:
            ea = event_accumulator.EventAccumulator(event_file_path, size_guidance={'scalars': 0}, purge_orphaned_data=True)
            ea.Reload()
            all_scalar_tags = ea.Tags().get("scalars", [])
            if not all_scalar_tags:
                skipped_runs_count += 1
                continue

            if NUM_TASKS_CONFIG > 0 and task_ids_from_config: # Only extract if tasks are defined
                for actual_task_id_from_tb_tag in task_ids_from_config:
                    s_tag = f"Episode/task_{actual_task_id_from_tb_tag}_success"
                    if s_tag in all_scalar_tags:
                        for event in ea.Scalars(s_tag):
                            all_records.append({
                                'exp_name': exp_name_from_setup,
                                'env_name': f"task_{actual_task_id_from_tb_tag}",
                                'seed': seed_from_filename,
                                'metric': "task_success_rate",
                                'env_step': event.step * num_envs_from_filename * HORIZON,
                                'value': event.value
                            })
                            run_data_extracted_flag = True
                    r_tag = f"Episode/task_{actual_task_id_from_tb_tag}_reward"
                    if r_tag in all_scalar_tags:
                        for event in ea.Scalars(r_tag):
                            all_records.append({
                                'exp_name': exp_name_from_setup,
                                'env_name': f"task_{actual_task_id_from_tb_tag}",
                                'seed': seed_from_filename,
                                'metric': "task_reward",
                                'env_step': event.step * num_envs_from_filename * HORIZON,
                                'value': event.value
                            })
                            run_data_extracted_flag = True
            
            if run_data_extracted_flag:
                processed_runs_count += 1
            else: # No relevant data extracted from this run
                if NUM_TASKS_CONFIG > 0 and task_ids_from_config: pass # Silently skip if expected data not found
                skipped_runs_count += 1
        except Exception: # Errors during EventAccumulator processing
            skipped_runs_count += 1
    
    print("\n--- Data Loading Summary ---")
    if not all_records:
        print("No data records were extracted. Exiting.")
        exit()
    
    print(f"Successfully extracted per-task records from {processed_runs_count} runs.")
    if skipped_runs_count > 0:
        print(f"Skipped {skipped_runs_count} directories/runs (no match, env mismatch, no/empty event file, or no specified per-task metrics).")

    df = pd.DataFrame(all_records)
    if df.empty:
        print("DataFrame is empty after processing. No per-task data to analyze.")
    else:
        print(f"\nTotal per-task records initially extracted: {len(df)}")
        df.sort_values(by=['exp_name', 'env_name', 'metric', 'seed', 'env_step'], inplace=True)
        df.drop_duplicates(subset=['exp_name', 'env_name', 'metric', 'seed', 'env_step'], keep='last', inplace=True)
        print(f"Total per-task records after deduplication: {len(df)}")
        
        output_csv_path = "extracted_per_task_experiment_data.csv"
        try:
            df.to_csv(output_csv_path, index=False)
            print(f"Per-task dataframe saved to {output_csv_path}")
        except Exception as e:
            print(f"\nError saving per-task DataFrame to CSV: {e}")

        metric_types_to_plot = {"task_success_rate": "Success Rate", "task_reward": "Reward"}
        aggregate_metrics_to_calculate = {"IQM": calculate_iqm, "Median": calculate_median, "Mean": calculate_mean}
        
        # Desired order for plotting, algorithms not in data or SETUP_INFO will be ignored
        desired_plot_order = ["SH-Vanilla", "SH-MOORE", "SH-CARE", "MH-CARE", "FAMO", "PCGrad"]
        unique_exp_names_from_df = df['exp_name'].unique().tolist()

        # Filter and order algorithms that are present in the data AND have a color defined
        ordered_algorithms_for_plot = [
            algo for algo in desired_plot_order if algo in unique_exp_names_from_df and algo in plot_colors_final
        ]
        other_available_algos = [
            algo for algo in unique_exp_names_from_df
            if algo not in ordered_algorithms_for_plot and algo in plot_colors_final
        ]
        ordered_algorithms_for_plot.extend(other_available_algos)

        if not ordered_algorithms_for_plot:
            print("\nNo algorithms found in the data that are also in SETUP_INFO for plotting. Skipping CI plots.")
        else:
            print(f"\nAlgorithms to be plotted: {ordered_algorithms_for_plot}")
            for metric_key, xlabel_name_for_plot in metric_types_to_plot.items():
                print(f"\n--- Calculating Aggregates for: {metric_key} ---")
                metric_matrix_dict_current_metric = generate_metric_matrix_dict(df, metric_type=metric_key)
                if not metric_matrix_dict_current_metric:
                    print(f"No data to generate metric matrix for {metric_key}. Skipping plot for this metric.")
                    continue

                # Use only algorithms that have data for the current metric
                current_metric_plot_algos = [
                    algo for algo in ordered_algorithms_for_plot if algo in metric_matrix_dict_current_metric
                ]
                if not current_metric_plot_algos:
                    print(f"No algorithms with valid data matrices for {metric_key} after filtering. Skipping plot for this metric.")
                    continue
                
                all_point_estimates = {algo: {} for algo in current_metric_plot_algos}
                all_confidence_intervals = {algo: {} for algo in current_metric_plot_algos}

                for algo_name in tqdm(current_metric_plot_algos, desc=f"Bootstrapping {metric_key}", unit="algo", ncols=100, leave=False):
                    data_matrix = metric_matrix_dict_current_metric.get(algo_name)
                    # data_matrix validity is already ensured by how current_metric_plot_algos is constructed
                    for agg_name, agg_func in aggregate_metrics_to_calculate.items():
                        ci, point_est = stratified_bootstrap_ci(data_matrix, agg_func, n_bootstrap_samples=N_BOOTSTRAP_SAMPLES, alpha=ALPHA)
                        all_point_estimates[algo_name][agg_name] = point_est
                        all_confidence_intervals[algo_name][agg_name] = ci
                
                plot_filename = f"aggregate_plot_{metric_key}_{setting}.png"
                full_plot_path = os.path.join(OUTPUT_PLOT_DIR, plot_filename)
                plot_confidence_intervals(
                    algorithms=current_metric_plot_algos, # Pass the filtered list for the current metric
                    all_point_estimates=all_point_estimates,
                    all_confidence_intervals=all_confidence_intervals,
                    metric_names=list(aggregate_metrics_to_calculate.keys()),
                    plot_title=f"Aggregate {xlabel_name_for_plot} ({setting.upper()}) - {((1-ALPHA)*100):.0f}% Stratified Bootstrap CI",
                    colors_dict=plot_colors_final, 
                    xlabel_name=xlabel_name_for_plot,
                    output_filename=full_plot_path
                )

                print(f"\n--- LaTeX Table Snippet for: {xlabel_name_for_plot} ({setting.upper()}) ---")
                print(f"% Confidence Level: {((1-ALPHA)*100):.0f}%")
                print(f"% Values: Point Estimate [Lower CI, Upper CI]")

                agg_metric_names_list = list(aggregate_metrics_to_calculate.keys())
                
                # Basic LaTeX tabular environment for structure
                # Using 'l' for algorithm name, and 'r' (right-aligned) for numeric data columns
                latex_tabular_string = f"\\begin{{tabular}}{{l{''.join(['r' for _ in agg_metric_names_list])}}}\n"
                latex_tabular_string += "  \\toprule\n"
                latex_tabular_string += "  Algorithm & " + " & ".join(agg_metric_names_list) + " \\\\\n"
                latex_tabular_string += "  \\midrule\n"

                # Sanitize algorithm names for LaTeX (replace underscore and hyphen)
                def sanitize_latex_simple(text):
                    return text.replace('_', '\\_').replace('-', '\\text{-}')


                for algo_name in current_metric_plot_algos: # Use the filtered list
                    # Sanitize algorithm name for LaTeX display
                    sanitized_algo_name = sanitize_latex_simple(algo_name)
                    row_data = [sanitized_algo_name]
                    
                    for agg_name in agg_metric_names_list:
                        point_est = all_point_estimates.get(algo_name, {}).get(agg_name, np.nan)
                        ci = all_confidence_intervals.get(algo_name, {}).get(agg_name, (np.nan, np.nan))
                        lower_ci, upper_ci = ci

                        if np.isnan(point_est) or np.isnan(lower_ci) or np.isnan(upper_ci):
                            cell_str = "--"
                        else:
                            # Adjust precision as needed (e.g., .2f or .3f)
                            cell_str = f"{point_est:.3f} [{lower_ci:.3f}, {upper_ci:.3f}]"
                        row_data.append(cell_str)
                    
                    latex_tabular_string += "  " + " & ".join(row_data) + " \\\\\n"

                latex_tabular_string += "  \\bottomrule\n"
                latex_tabular_string += "\\end{tabular}\n"
                
                print(latex_tabular_string)
    print("\n--- Script Finished ---")