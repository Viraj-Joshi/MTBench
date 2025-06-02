import os
import re
import yaml # For reading config files
import traceback # For detailed error printing

import pandas as pd
# import numpy as np # Not strictly needed with the current changes but harmless
from tensorboard.backend.event_processing import event_accumulator
import argparse
from tqdm import tqdm

from rliable import library as rly
from rliable import metrics as rly_metrics
from rliable import plot_utils as rly_plot_utils
import numpy as np
import matplotlib.pyplot as plt

def generate_metric_matrix_dict(eval_df, metric_type: str):
    """
    Generates a dictionary of metric matrices from eval_df based on the specified env_step.

    Args:
    - eval_df (pandas DataFrame): DataFrame containing the evaluation data.

    Returns:
    - metric_matrix_dict (dict): Dictionary where keys are experiment names and values are 2D numpy arrays representing metric matrices.
    """
    # Filter data for the specified env_step
    filtered_df = eval_df[
        (eval_df["metric"] == metric_type)
    ]

    # Initialize dictionary to hold the metric matrices
    metric_matrix_dict = {}

    # Get the unique experiment names
    exp_names = filtered_df["exp_name"].unique()

    for exp_name in exp_names:
        # filter df to last env_step
        last_env_step = filtered_df[filtered_df["exp_name"] == exp_name]["env_step"].max()
        exp_data = filtered_df[(filtered_df["exp_name"] == exp_name) & (filtered_df["env_step"] == last_env_step)]
        # Pivot table to have env_names as rows and individual scores as columns
        pivot_table = exp_data.pivot_table(
            values="value", index="env_name", columns="seed", aggfunc="first"
        )

        # Convert the pivot table to a numpy array
        metric_matrix = pivot_table.to_numpy()

        # Check for NaN values in each column (run or seed)
        nan_cols = np.any(np.isnan(metric_matrix), axis=0)

        # Remove columns with NaN values
        metric_matrix = metric_matrix[:, ~nan_cols]

        # Add the metric matrix to the dictionary
        metric_matrix_dict[exp_name] = metric_matrix

    return metric_matrix_dict

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

# --- Main Execution Logic ---
if __name__ == "__main__":
    LOG_DIR_DEFAULT = "/work/08962/vjoshi12/ls6/IsaacGymEnvs/runs/" 
    
    parser = argparse.ArgumentParser(description="Extract per-task TensorBoard scalar data into a Pandas DataFrame.")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR_DEFAULT, help="Root directory containing TensorBoard run folders.")
    parser.add_argument("--is_mt10", action="store_true", help="MT10 specific setup. Affects 'setting' variable and NUM_TASKS_CONFIG.")
    parser.add_argument("--horizon", type=int, default=32, help="Horizon value used in 'env_step' calculation (step * num_envs * HORIZON).")

    args = parser.parse_args()

    LOG_DIR = args.log_dir
    HORIZON = args.horizon 

    if args.is_mt10:
        setting = "mt10"
        NUM_TASKS_CONFIG = 10 
    else:
        setting = "mt50" 
        NUM_TASKS_CONFIG = 50 
        
    print(f"Using setting: {setting}, Horizon for env_step calc: {HORIZON}, Log Dir: {LOG_DIR}")

    ### VANILLA ###
    PATTERN_SHPPO_VANILLA = rf"05_09_ppo_vanilla_{setting}_rand_envs_(\d+)_seed_(\d+).*"
    PATTERN_MHPPO_VANILLA = rf"mhppo_vanilla_{setting}_rand_envs_(\d+)_seed_(\d+).*"
    ### Grad Mani ###
    PATTERN_FAMO = rf"05_09_ppo_famo_{setting}_rand_envs_(\d+)_seed_(\d+).*"
    PATTERN_PCGRAD = rf"05_11_ppo_pcgrad_{setting}_rand_envs_(\d+)_seed_(\d+).*"
    PATTERN_CAGRAD = rf"ppo_cagrad_{setting}_rand_envs_(\d+)_seed_(\d+).*"
    ### Neural Architecture ###
    PATTERN_SHPPO_PACO = rf"shppo_paco_{setting}_rand_envs_(\d+)_seed_(\d+).*"
    PATTERN_MHPPO_MOORE = rf"mhppo_moore_{setting}_rand_envs_(\d+)_seed_(\d+).*"
    PATTERN_SHPPO_MOORE = rf"shppo_moore_{setting}_rand_envs_(\d+)_seed_(\d+).*"
    PATTERN_SOFT_MODULARIZATION = rf"ppo_soft_modularization_{setting}_rand_envs_(\d+)_seed_(\d+).*"
    PATTERN_MHPPO_CARE = rf"05_11_mhppo_care_{setting}_rand_envs_(\d+)_seed_(\d+).*"
    PATTERN_SHPPO_CARE = rf"05_11_shppo_care_{setting}_rand_envs_(\d+)_seed_(\d+).*"

    SETUP_INFO = {
        ### VANILLA ###
        # 'mhppo_vanilla': {'pattern': PATTERN_MHPPO_VANILLA, 'color': '#2ca02c', 'name': 'MH-Vanilla', 'required_envs': 24576},
        'shppo_vanilla': {'pattern': PATTERN_SHPPO_VANILLA, 'color': '#1A85FF', 'name': 'SH-Vanilla', 'required_envs': 24576},
        ### Grad Mani ###
        'famo': {'pattern': PATTERN_FAMO, 'color': '#3690ff', 'name': 'FAMO', 'required_envs': 24576},
        # 'cagrad': {'pattern': PATTERN_CAGRAD, 'color': '#1f77b4', 'name': 'CAGrad', 'required_envs': 24576},
        'pcgrad': {'pattern': PATTERN_PCGRAD, 'color': '#ff7f0e', 'name': 'PCGrad', 'required_envs': 24576},
        ### Neural Architecture ###
        'mhppo_care': {'pattern': PATTERN_MHPPO_CARE, 'color': '#FF9800', 'name': 'MH-CARE', 'required_envs': 24576},
        'shppo_care': {'pattern': PATTERN_SHPPO_CARE, 'color': '#FF5722', 'name': 'SH-CARE', 'required_envs': 24576},
        'shppo_moore': {'pattern': PATTERN_SHPPO_MOORE, 'color': '#FF4081', 'name': 'SH-MOORE', 'required_envs': 24576},
        # 'mhppo_moore': {'pattern': PATTERN_MHPPO_MOORE, 'color': '#00C853', 'name': 'MH-MOORE', 'required_envs': 24576},
        # 'shppo_paco': {'pattern': PATTERN_SHPPO_PACO, 'color': '#FF5722', 'name': 'SH-PACO', 'required_envs': 24576},
        # 'soft_modularization': {'pattern': PATTERN_SOFT_MODULARIZATION, 'color': '#D500F9', 'name': 'Soft Modularization', 'required_envs': 24576},
    }

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

    for item_name in tqdm(directories_to_process, desc="Scanning Log Dirs", unit="dir", ncols=100):
        item_path = os.path.join(LOG_DIR, item_name)
        
        setup_type, num_envs_from_filename, seed_from_filename = extract_setup_info(item_name, SETUP_INFO)

        if setup_type is None:
            skipped_runs_count += 1
            continue
        
        current_exp_config = SETUP_INFO.get(setup_type)
        if not current_exp_config:
            skipped_runs_count += 1
            continue

        exp_name = current_exp_config['name']
        
        required_env_count = current_exp_config.get('required_envs')
        if required_env_count is not None and num_envs_from_filename != required_env_count:
            tqdm.write(f"Skipping {item_name}: Env count mismatch (found {num_envs_from_filename}, need {required_env_count} for {exp_name}).")
            skipped_runs_count += 1
            continue
            
        task_ids_from_config = list(range(NUM_TASKS_CONFIG)) 
        config_path = os.path.join(item_path, CONFIG_FILENAME)
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r") as f:
                    config_yaml = yaml.safe_load(f)
                loaded_task_ids_raw = config_yaml.get('task_id', list(range(NUM_TASKS_CONFIG)))
                if isinstance(loaded_task_ids_raw, list) and all(isinstance(tid, int) for tid in loaded_task_ids_raw):
                    if NUM_TASKS_CONFIG > 0 and not loaded_task_ids_raw:
                        tqdm.write(f"Warning: Empty task_ids in '{config_path}' for {item_name} (NUM_TASKS_CONFIG={NUM_TASKS_CONFIG}). Using default.")
                    elif loaded_task_ids_raw: # Use only if not empty
                        task_ids_from_config = loaded_task_ids_raw
                elif NUM_TASKS_CONFIG > 0:
                     tqdm.write(f"Warning: Invalid task_ids in '{config_path}' for {item_name}. Using default task IDs.")
            except Exception as e:
                tqdm.write(f"Error reading or parsing config '{config_path}': {e}. Using default task IDs.")
        elif NUM_TASKS_CONFIG > 0 and not task_ids_from_config:
             tqdm.write(f"Warning: Config file missing and NUM_TASKS_CONFIG={NUM_TASKS_CONFIG}, using default task IDs for {item_name}.")

        event_file_path = None
        summaries_dir = os.path.join(item_path, "summaries")
        possible_dirs = [summaries_dir] if os.path.isdir(summaries_dir) else []
        possible_dirs.append(item_path)

        for p_dir_opt in possible_dirs:
            if os.path.isdir(p_dir_opt):
                try:
                    event_files = [f for f in os.listdir(p_dir_opt) if f.startswith("events.out.tfevents")]
                    if event_files:
                        event_files.sort() 
                        event_file_path = os.path.join(p_dir_opt, event_files[-1])
                        break 
                except Exception as e:
                    tqdm.write(f"Warning: Could not access or list files in '{p_dir_opt}' for {item_name}: {e}")
        
        if not event_file_path:
            tqdm.write(f"Skipping '{item_name}': No event file found.")
            skipped_runs_count += 1
            continue

        run_data_extracted_flag = False
        try:
            ea = event_accumulator.EventAccumulator(event_file_path, size_guidance={'scalars': 0}, purge_orphaned_data=True)
            ea.Reload()
            all_scalar_tags = ea.Tags().get("scalars", [])

            if not all_scalar_tags:
                tqdm.write(f"Warning: No scalars in event file for {item_name}.")
                skipped_runs_count += 1
                continue
            
            # --- Only Per-task success rates & Per-task rewards will be extracted ---
            if NUM_TASKS_CONFIG > 0 and task_ids_from_config:
                for actual_task_id_from_tb_tag in task_ids_from_config:
                    # Per-task success
                    s_tag = f"Episode/task_{actual_task_id_from_tb_tag}_success"
                    if s_tag in all_scalar_tags:
                        scalar_events_s = ea.Scalars(s_tag)
                        for event in scalar_events_s:
                            all_records.append({
                                'exp_name': exp_name,
                                'env_name': f"task_{actual_task_id_from_tb_tag}",
                                'seed': seed_from_filename,
                                'metric': "task_success_rate",
                                'env_step': event.step * num_envs_from_filename * HORIZON,
                                'value': event.value
                            })
                            run_data_extracted_flag = True
                    
                    # Per-task reward
                    r_tag = f"Episode/task_{actual_task_id_from_tb_tag}_reward"
                    if r_tag in all_scalar_tags:
                        scalar_events_r = ea.Scalars(r_tag)
                        for event in scalar_events_r:
                            all_records.append({
                                'exp_name': exp_name,
                                'env_name': f"task_{actual_task_id_from_tb_tag}",
                                'seed': seed_from_filename,
                                'metric': "task_reward",
                                'env_step': event.step * num_envs_from_filename * HORIZON,
                                'value': event.value
                            })
                            run_data_extracted_flag = True
            
            if run_data_extracted_flag:
                processed_runs_count += 1
            else:
                # This warning might trigger if a run has no per-task data but was otherwise valid
                # Or if NUM_TASKS_CONFIG is 0 / task_ids_from_config is empty
                if NUM_TASKS_CONFIG > 0 and task_ids_from_config: # Only warn if we expected to find per-task data
                    tqdm.write(f"Warning: No specified per-task metric data extracted from '{item_name}'.")
                skipped_runs_count += 1
        
        except Exception as e:
            tqdm.write(f"ERROR processing run '{item_name}' with event file '{event_file_path}': {e}")
            traceback.print_exc()
            skipped_runs_count += 1

    print("\n--- Processing Summary ---")
    if not all_records:
        print("No data records were extracted. Exiting.")
        exit()

    print(f"Successfully processed data from {processed_runs_count} runs leading to per-task record extraction.")
    if skipped_runs_count > 0:
        print(f"Skipped {skipped_runs_count} directories/runs for various reasons (no match, env mismatch, no event file, no scalars, or no specified per-task metrics).")

    df = pd.DataFrame(all_records)

    if df.empty:
        print("DataFrame is empty after processing. No per-task data to show.")
    else:
        print("\n--- Extracted Per-Task Data Summary ---")
        print(f"Total per-task records extracted: {len(df)}")

        df.sort_values(by=['exp_name', 'env_name', 'metric', 'seed', 'env_step'], inplace=True)
        df.drop_duplicates(subset=['exp_name', 'env_name', 'metric', 'seed', 'env_step'], keep='last', inplace=True)
        
        print(f"Total per-task records after deduplication: {len(df)}")
        
        print("\nDataFrame Info:")
        df.info()
        
        print("\nDataFrame Head:")
        print(df.head())

        print("\nDataFrame Tail:")
        print(df.tail())

    aggregate_func = lambda x: np.array([
        rly_metrics.aggregate_iqm(x),
        rly_metrics.aggregate_median(x),
        rly_metrics.aggregate_mean(x),])

    metric_matrix_dict = generate_metric_matrix_dict(
        df, 
        metric_type='task_success_rate'
    )

    print("\n--- Metric Matrix Dictionary ---")
    for key, value in metric_matrix_dict.items():
        print(f"{key}: {value}")

    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        metric_matrix_dict, aggregate_func, reps=2000,
    )

    print("\n--- Aggregate Scores and Confidence Intervals ---")
    print(f"Aggregate Scores: {aggregate_scores}")
    print(f"Aggregate Score CIs: {aggregate_score_cis}")

    fig, axes = rly_plot_utils.plot_interval_estimates(
        aggregate_scores, aggregate_score_cis,
        metric_names=['IQM', 'Median', 'Mean'],
        algorithms=list(metric_matrix_dict.keys()),
        xlabel_y_coordinate=-1.5,
        xlabel='Success Rate',
    )

    save_path = "scripts/figures/iqm_rliable/aggregate_scores_plot.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
        # ----- END MODIFICATION -----
    print(f"\nPlot saved to {save_path}")


