import os
from collections import defaultdict
import re
import math

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap # Used for coloring different num_envs lines
from tensorboard.backend.event_processing import event_accumulator

# --- Constants ---
LOG_DIR = "runs/"
TARGET_TASK = 38 # The specific task ID we want to analyze
# Pattern to capture Task ID, Num Envs, and Seed
# It expects format like: ppo_vanilla_task_ID_rand_envs_NUMENVS_anything_seed_SEED
FILENAME_PATTERN = r"^ppo_vanilla_task_(\d+)_rand_envs_(\d+)_seed_(\d+)_.*$"
STEPS_PER_BATCH = 32 # Constant part of the frame multiplier (num_envs * STEPS_PER_BATCH)

# --- Helper Function to Extract Run Info from Pattern ---
def extract_run_info(run_name):
    """
    Extracts task number, num_envs, and seed number from the run_name
    if it matches the specific pattern. Returns (task, num_envs, seed)
    or None if no match.
    """
    match = re.match(FILENAME_PATTERN, run_name)
    if match:
        task_number = int(match.group(1))
        num_envs = int(match.group(2))
        seed_number = int(match.group(3))
        return task_number, num_envs, seed_number
    else:
        return None

def adjust_data(df):
    """
    Adjusts the DataFrame by adding a row at frame 0 with success = 0.
    Handles empty DataFrame case and ensures unique frames.
    """
    if df.empty:
        return pd.DataFrame({'success': [0.0], 'frame': [0]})

    # Ensure unique frames, keeping the last value for duplicates
    df = df.drop_duplicates(subset=['frame'], keep='last').sort_values('frame').reset_index(drop=True)

    # Ensure frame 0 exists with success 0
    if 0 not in df['frame'].values:
         new_row = pd.DataFrame({'success': [0.0], 'frame': [0]})
         result_df = pd.concat([new_row, df], ignore_index=True).sort_values('frame').reset_index(drop=True)
    elif df.loc[df['frame'] == 0, 'success'].iloc[0] != 0.0:
        # If frame 0 exists but success isn't 0, replace it or adjust
        df = df[df['frame'] > 0].reset_index(drop=True) # Keep only data after frame 0
        new_row = pd.DataFrame({'success': [0.0], 'frame': [0]})
        result_df = pd.concat([new_row, df], ignore_index=True).sort_values('frame').reset_index(drop=True)
    else:
        result_df = df # Already has frame 0 with success 0 and sorted

    return result_df

# --- Plotting Function (Combined Average Curves) ---
def plot_combined_avg_curves(dataframes_by_num_envs, task_number):
    """
    Plots average success rate vs frames for multiple num_envs configurations
    for a single task on one plot, showing mean and shaded std dev region.
    """
    print(f"Generating combined average success curve plot for Task {task_number}...")

    if not dataframes_by_num_envs:
        print(f"No data available to plot for Task {task_number}.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get a colormap
    num_configs = len(dataframes_by_num_envs)
    # Use a colormap that provides distinct colors, e.g., 'viridis', 'plasma', 'tab10'
    cmap = get_cmap('viridis', num_configs)
    colors = [cmap(i) for i in range(num_configs)]
    
    # Sort num_envs values for consistent plotting order and color assignment
    sorted_num_envs = sorted(dataframes_by_num_envs.keys())

    for i, num_envs in enumerate(sorted_num_envs):
        list_of_dataframes = dataframes_by_num_envs[num_envs]

        # Adjust each dataframe (add frame 0, success 0)
        adjusted_data = [adjust_data(df.copy()) for df in list_of_dataframes]
        # Filter out any dataframes that became empty or have only one point after adjustment
        adjusted_data = [df for df in adjusted_data if not df.empty and len(df) > 1]

        if not adjusted_data:
            print(f"Warning: No valid data for Task {task_number}, NumEnvs {num_envs} after adjustment. Skipping.")
            continue

        try:
            # --- Frame Alignment to Maximum Length for this group ---
            longest_run_df = max(adjusted_data, key=lambda df: df['frame'].max())
            ref_frames_full, unique_indices = np.unique(longest_run_df['frame'].values, return_index=True)

            if len(ref_frames_full) < 2:
                 print(f"Warning: Reference frame axis for Task {task_number}, NumEnvs {num_envs} has less than 2 points. Skipping.")
                 continue

            # --- Interpolation onto the Full Reference Axis for this group ---
            interpolated_successes = []
            for df in adjusted_data:
                unique_frames, unique_indices = np.unique(df['frame'].values, return_index=True)
                unique_success = df['success'].values[unique_indices]

                if len(unique_frames) < 2:
                     print(f"Warning: Skipping a run for Task {task_number}, NumEnvs {num_envs} due to insufficient unique frame points ({len(unique_frames)}).")
                     continue

                interp_success = np.interp(ref_frames_full, unique_frames, unique_success)
                interpolated_successes.append(interp_success)

            if not interpolated_successes:
                print(f"Warning: No runs successfully interpolated for Task {task_number}, NumEnvs {num_envs}. Skipping.")
                continue

            # --- Statistics Calculation ---
            frame_successes = np.array(interpolated_successes)
            frame_mean = np.mean(frame_successes, axis=0)
            frame_std = np.std(frame_successes, axis=0)

            # --- Plotting on the single axes ---
            current_color = colors[i]
            label = f'NumEnvs: {num_envs} ({len(list_of_dataframes)} seeds)' # Label includes num_envs and seed count

            ax.plot(ref_frames_full, frame_mean, color=current_color, linewidth=1.5, label=label)
            ax.fill_between(ref_frames_full,
                            frame_mean - frame_std,
                            frame_mean + frame_std,
                            color=current_color, alpha=0.1) # Use same color, lower alpha

        except Exception as e:
            print(f"Error processing data for Task {task_number}, NumEnvs {num_envs}: {e}")
            import traceback
            traceback.print_exc()
            continue # Continue with the next num_envs group

    # --- Final Plot Customization ---
    ax.set_xlabel("Frames", fontsize=12)
    ax.set_ylabel("Average Success Rate", fontsize=12)
    ax.set_title(f"Task {task_number} - Average Success Rate vs Frames (by NumEnvs)", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.legend(title="Configurations")

    plt.tight_layout()

    save_dir = "scripts/figures/vanilla"
    os.makedirs(save_dir, exist_ok=True)
    save_path_base = os.path.join(save_dir, f"task_{task_number}_combined_avg_success_curves")
    plt.savefig(f"{save_path_base}.pdf")
    plt.savefig(f"{save_path_base}.png", dpi=300)
    print(f"Combined plot saved to {save_path_base}.png/.pdf")
    plt.close(fig)


# --- Main Execution Logic ---
if __name__ == "__main__":

    # Store dataframes and final success rates, grouped by num_envs
    dataframes_by_num_envs = defaultdict(list)
    final_success_by_num_envs = defaultdict(list)

    if not os.path.isdir(LOG_DIR):
        print(f"Error: Log directory '{LOG_DIR}' not found.")
        exit()

    print(f"Scanning directory: {LOG_DIR} for runs matching pattern '{FILENAME_PATTERN}' for Task {TARGET_TASK}")
    processed_runs_total = 0
    processed_for_target_task = 0

    for item_name in os.listdir(LOG_DIR):
        item_path = os.path.join(LOG_DIR, item_name)
        if not os.path.isdir(item_path): continue

        run_info = extract_run_info(item_name)

        if run_info is None:
            # print(f"Skipping '{item_name}': Does not match pattern '{FILENAME_PATTERN}'")
            continue # Skip items that don't match the pattern

        task_number, num_envs, seed_number = run_info

        if task_number != TARGET_TASK:
             # print(f"Skipping '{item_name}': Task ID {task_number} does not match target task {TARGET_TASK}")
             continue # Skip runs that are not for the target task

        # Found a run for the target task with a specific num_envs, now locate the event file
        current_frame_multiplier = num_envs * STEPS_PER_BATCH

        event_file_path = None
        potential_paths = [os.path.join(item_path, "summaries"), item_path]
        sp = None
        for p in potential_paths:
             if os.path.isdir(p):
                 sp = p
                 break
        if sp is None:
             print(f"  Warning: Could not find a valid directory containing event files for '{item_name}'. Skipping.")
             continue

        try:
            event_files = [f for f in os.listdir(sp) if f.startswith("events.out.tfevents")]
            if not event_files:
                 # print(f"  Warning: No event file found in '{sp}' for item '{item_name}'. Skipping.")
                 continue
            event_file_path = os.path.join(sp, event_files[0])
        except Exception as e:
            print(f"Error accessing path '{sp}' for item '{item_name}': {e}")
            continue

        if event_file_path is None:
            continue # Should not happen due to checks above, but defensive

        # print(f"Processing: Task {TARGET_TASK}, NumEnvs {num_envs}, Seed {seed_number} (from '{item_name}')")
        try:
            ea = event_accumulator.EventAccumulator(event_file_path,
                size_guidance={'scalars': 0}, purge_orphaned_data=True)
            ea.Reload()
            tags = ea.Tags().get("scalars", [])

            success_tag = None
            exact_match_tags = [t for t in tags if t.endswith('/average_task_success_rate')]
            if exact_match_tags:
                 success_tag = min(exact_match_tags, key=len)
            else:
                 potential_tags = [t for t in tags if 'average_task_success_rate' in t]
                 if potential_tags:
                     success_tag = min(potential_tags, key=len)

            if success_tag is None:
                print(f"  Warning: Could not find success rate tag (e.g., '.../average_task_success_rate') for '{item_name}'. Available tags: {tags}. Skipping.")
                continue

            values = ea.Scalars(success_tag)
            if not values:
                print(f"  Warning: No data points found for tag '{success_tag}' in '{item_name}'. Skipping.")
                continue

            if not (hasattr(values[0], 'step') and hasattr(values[0], 'value')):
                 print(f"  Error: Scalar data missing 'step' or 'value' attribute in '{item_name}'. Skipping.")
                 continue

            df = pd.DataFrame({
                'success': [x.value for x in values],
                'frame': [x.step * current_frame_multiplier for x in values]
            })

            df.dropna(inplace=True)
            df = df[df['success'] >= 0]

            if df.empty:
                print(f"  Warning: DataFrame empty after cleaning for '{item_name}'. Skipping.")
                continue

            # Ensure final success is valid number before appending
            final_success = df['success'].iloc[-1]
            if pd.isna(final_success):
                 # print(f"  Warning: Final success value is NaN for '{item_name}'. Skipping run for final success calculation, but including curve data if valid.")
                 pass # Just skip for final success calculation, but add to df list
            else:
                 final_success_by_num_envs[num_envs].append(final_success)

            # Append the dataframe to the list for this num_envs group
            dataframes_by_num_envs[num_envs].append(df)

            processed_runs_total += 1
            processed_for_target_task += 1

        except Exception as e:
            print(f"  ERROR processing run '{item_name}': {e}")
            import traceback
            traceback.print_exc()

    print(f"\nFinished scanning. Found {processed_for_target_task} runs for Task {TARGET_TASK} across {len(dataframes_by_num_envs)} different NumEnvs configurations.")

    if not dataframes_by_num_envs:
        print(f"No data loaded for Task {TARGET_TASK} matching the criteria. Exiting.")
    else:
        print("\n--- Results per NumEnvs Configuration ---")
        # Calculate and print final success stats for each group
        sorted_num_envs = sorted(dataframes_by_num_envs.keys())
        for num_envs in sorted_num_envs:
            num_runs_in_group = len(dataframes_by_num_envs[num_envs])
            print(f"\nNumEnvs: {num_envs} ({num_runs_in_group} runs)")
            if num_envs in final_success_by_num_envs and final_success_by_num_envs[num_envs]:
                 avg_final_success = np.mean(final_success_by_num_envs[num_envs])
                 std_final_success = np.std(final_success_by_num_envs[num_envs])
                 print(f"  Final Success (Averaged over {len(final_success_by_num_envs[num_envs])} runs):")
                 print(f"    Mean: {avg_final_success:.4f}")
                 print(f"    Std Dev: {std_final_success:.4f}")
            else:
                 print(f"  No valid final success data points found for NumEnvs {num_envs}.")

        # Plot all combined curves on a single plot
        plot_combined_avg_curves(dataframes_by_num_envs, TARGET_TASK)

        print("\nAll plotting complete!")