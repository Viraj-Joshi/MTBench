import os
from collections import defaultdict
import re
import math # For ceiling function for grid dimensions

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from tensorboard.backend.event_processing import event_accumulator

# --- Constants ---
LOG_DIR = "runs/"
FRAME_MULTIPLIER = 4096 * 32 # PPO frame multiplier (Verify this value!)
FILENAME_PATTERN = r"^ppo_vanilla_task_(\d+)_rand_envs_4096.*_seed_(\d+)"
NUM_TASKS = 50 # Tasks 0 to 49

# --- Helper Function to Extract Task/Seed from Specific Pattern ---
def extract_task_seed_from_pattern(run_name):
    """
    Extracts the task and seed number if the run_name matches the specific pattern.
    """
    match = re.match(FILENAME_PATTERN, run_name)
    if match:
        task_number = int(match.group(1))
        seed_number = int(match.group(2))
        if 0 <= task_number < NUM_TASKS:
            return task_number, seed_number
        else:
            return None, None
    else:
        return None, None

def adjust_data(df):
    """
    Adjusts the DataFrame by adding a row at index 0 with success = 0, frame = 0.
    Handles empty DataFrame case.
    """
    if df.empty:
        return pd.DataFrame({'success': [0.0], 'frame': [0]})

    # Ensure frame 0 exists with success 0
    if 0 not in df['frame'].values:
         new_row = pd.DataFrame({'success': [0.0], 'frame': [0]})
         result_df = pd.concat([new_row, df], ignore_index=True).sort_values('frame').reset_index(drop=True)
    elif df.loc[df['frame'] == 0, 'success'].iloc[0] != 0.0:
        # If frame 0 exists but success isn't 0, ensure it starts at 0
        df = df[df['frame'] > 0] # Remove existing frame 0 temporarily
        new_row = pd.DataFrame({'success': [0.0], 'frame': [0]})
        result_df = pd.concat([new_row, df], ignore_index=True).sort_values('frame').reset_index(drop=True)
    else:
        result_df = df.sort_values('frame').reset_index(drop=True) # Already has frame 0 with success 0

    return result_df


# --- Plotting Function 1: Bar Plot of Average Final Success ---
def plot_average_final_success(task_final_success_map):
    """
    Plots a bar chart of the average final success rate for tasks 0-49.
    """
    print("Generating average final success bar plot...")
    avg_success_rates = np.zeros(NUM_TASKS)
    task_ids = np.arange(NUM_TASKS)

    for task_num in range(NUM_TASKS):
        if task_num in task_final_success_map and task_final_success_map[task_num]:
            avg_success_rates[task_num] = np.mean(task_final_success_map[task_num])
        else:
            avg_success_rates[task_num] = 0.0

    fig, ax = plt.subplots(figsize=(18, 6))
    bars = ax.bar(task_ids, avg_success_rates, color='skyblue')

    ax.set_xlabel("Task ID", fontsize=12)
    ax.set_ylabel("Average Final Success Rate", fontsize=12)
    ax.set_title(f"Average Final Success Rate per Task (0-{NUM_TASKS-1})", fontsize=14)
    ax.set_xticks(task_ids[::2])
    ax.set_xticklabels(task_ids[::2], rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    save_dir = "scripts/figures/vanilla"
    os.makedirs(save_dir, exist_ok=True)
    save_path_base = os.path.join(save_dir, "avg_final_success_bar")
    plt.savefig(f"{save_path_base}.pdf")
    plt.savefig(f"{save_path_base}.png", dpi=300)
    print(f"Bar plot saved to {save_path_base}.png/.pdf")
    plt.close(fig)

# --- Plotting Function 2: Grid Plot of Success Curves ---
# *** MODIFIED FUNCTION ***
def plot_task_success_grid(results_dict):
    """
    Plots individual success rate vs frames curves for tasks 0-49 in a grid,
    showing the mean line and a shaded region for ±1 standard deviation.
    Curves are extended to the maximum frame count observed for any seed within that task.
    """
    print("Generating grid plot of task success curves...")
    ncols = 8
    nrows = math.ceil(NUM_TASKS / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2),
                             sharex=True, sharey=True, squeeze=False)
    fig.suptitle(f'Success Rate vs Frames per Task (0-{NUM_TASKS-1}) - Mean ± 1 Std Dev', fontsize=16)

    axes_flat = axes.flatten()

    for task_num in range(NUM_TASKS):
        ax = axes_flat[task_num] # Get the subplot for this task

        if task_num in results_dict and results_dict[task_num]:
            run_data = results_dict[task_num]
            # Add frame 0 with success 0 if missing, crucial for interpolation start
            run_data_adjusted = [adjust_data(data.copy()) for data in run_data]
            # Filter out any completely empty dataframes after adjustment
            run_data_adjusted = [df for df in run_data_adjusted if not df.empty and len(df) > 1]

            if not run_data_adjusted: # Skip if no valid data remains
                 ax.set_title(f"Task {task_num} (No Data)", fontsize=10)
                 ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, color='grey', fontsize=9)
                 ax.axis('off')
                 continue

            try:
                # --- Frame Alignment to Maximum Length ---
                max_frames_per_seed = [df['frame'].max() for df in run_data_adjusted]
                if not max_frames_per_seed: continue # Should not happen due to check above, but safety first
                global_max_frame = max(max_frames_per_seed)

                # Find the run that achieved the global_max_frame (or is longest) to use its frame steps as reference
                longest_run_df = max(run_data_adjusted, key=lambda df: df['frame'].max())
                # Use the frame steps from the longest run as the common x-axis
                # Ensure uniqueness and sorted order, crucial for interpolation
                ref_frames_full, unique_indices = np.unique(longest_run_df['frame'].values, return_index=True)
                if len(ref_frames_full) < 2: continue # Need at least 2 points to interpolate

                # --- Interpolation onto the Full Reference Axis ---
                interpolated_successes = []
                for df in run_data_adjusted:
                    # Ensure data has unique, sorted frames for interpolation
                    unique_frames, unique_indices = np.unique(df['frame'].values, return_index=True)
                    unique_success = df['success'].values[unique_indices]

                    if len(unique_frames) < 2: continue # Need at least start and one more point

                    # Interpolate onto the *full* reference frame axis.
                    # np.interp will automatically extrapolate using the first/last success value
                    # for frames outside the range of unique_frames.
                    interp_success = np.interp(ref_frames_full, unique_frames, unique_success)
                    interpolated_successes.append(interp_success)

                if not interpolated_successes: continue # Skip if interpolation failed for all runs

                # --- Statistics Calculation ---
                # All arrays in interpolated_successes now have the same length (len(ref_frames_full))
                frame_successes = np.array(interpolated_successes)
                frame_mean = np.mean(frame_successes, axis=0)
                frame_std = np.std(frame_successes, axis=0) # Standard Deviation

                # --- Plotting on Subplot ---
                ax.plot(ref_frames_full, frame_mean, color='royalblue', linewidth=1.5)
                ax.fill_between(ref_frames_full,
                                frame_mean - frame_std, # Lower bound: Mean - 1 Std Dev
                                frame_mean + frame_std, # Upper bound: Mean + 1 Std Dev
                                alpha=0.2, color='lightblue') # Shaded region

                ax.set_title(f"Task {task_num}", fontsize=10)
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                ax.set_ylim(0, 1.05) # Ensure y-axis limits are consistent

                # --- Ticks and labels adjustments ---
                # Only show y-labels for the first column
                if task_num % ncols != 0:
                    plt.setp(ax.get_yticklabels(), visible=False)
                else:
                     for label in ax.get_yticklabels(): label.set_fontsize(8)

                # Only show x-labels for the last row (or potentially incomplete last row)
                is_last_row = (task_num // ncols == nrows - 1)
                # Check if there are tasks below the current one
                has_task_below = (task_num + ncols < NUM_TASKS)

                if has_task_below: # If not in last row (visually), hide x-labels
                     plt.setp(ax.get_xticklabels(), visible=False)
                else: # Last row needs x-labels
                    for label in ax.get_xticklabels():
                        label.set_fontsize(8)
                        label.set_rotation(30) # Rotate slightly for better fit
                        label.set_ha('right') # Align rotated labels
                    ax.xaxis.get_offset_text().set_fontsize(8)


            except Exception as e:
                print(f"Error plotting grid for task {task_num}: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for debugging
                ax.set_title(f"Task {task_num} (Error)", fontsize=10)
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes, color='red')
                ax.axis('off')
        else:
            # Task data not found
            ax.set_title(f"Task {task_num} (No Data)", fontsize=10) # Changed title slightly
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, color='grey', fontsize=9)
            ax.axis('off')

    # Turn off axes for unused subplots
    for i in range(NUM_TASKS, nrows * ncols):
        axes_flat[i].axis('off')

    # Add common labels (optional but good practice)
    fig.text(0.5, 0.01, 'Frames', ha='center', va='center', fontsize=12)
    fig.text(0.01, 0.5, 'Average Success Rate', ha='center', va='center', rotation='vertical', fontsize=12)


    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95]) # Adjust rect for suptitle and common labels

    save_dir = "figures"
    os.makedirs(save_dir, exist_ok=True)
    save_path_base = os.path.join(save_dir, "task_success_grid")
    plt.savefig(f"{save_path_base}.pdf")
    plt.savefig(f"{save_path_base}.png", dpi=300)
    print(f"Grid plot saved to {save_path_base}.png/.pdf")
    plt.close(fig)


# --- Main Execution Logic (No changes needed here) ---
if __name__ == "__main__":

    results_dataframes = defaultdict(list)
    task_final_success = defaultdict(list)

    if not os.path.isdir(LOG_DIR):
        print(f"Error: Log directory '{LOG_DIR}' not found.")
        exit()

    print(f"Scanning directory: {LOG_DIR} for pattern '{FILENAME_PATTERN}'")
    processed_runs = 0
    for item_name in os.listdir(LOG_DIR):
        item_path = os.path.join(LOG_DIR, item_name)
        if not os.path.isdir(item_path): continue

        task_number, seed_number = extract_task_seed_from_pattern(item_name)
        if task_number is None: continue

        event_file_path = None
        # Check both 'summaries' subdir and the main dir for event files
        potential_paths = [os.path.join(item_path, "summaries"), item_path]
        sp = None
        for p in potential_paths:
             if os.path.isdir(p):
                  sp = p
                  break # Found a valid directory
        if sp is None:
             print(f"  Warning: Could not find a valid directory containing event files for '{item_name}'. Skipping.")
             continue

        try:
            event_files = [f for f in os.listdir(sp) if f.startswith("events.out.tfevents")]
            if not event_files:
                 # If no event file found in the determined path `sp`, skip this run.
                 # Avoid printing a warning if the directory structure is simply different but contains no events.
                 # print(f"  Warning: No event file found in '{sp}' for item '{item_name}'. Skipping.")
                 continue
            # Select the first event file found
            event_file_path = os.path.join(sp, event_files[0])
        except Exception as e:
            print(f"Error accessing path '{sp}' for item '{item_name}': {e}")
            continue

        # Check if event_file_path was actually found and assigned
        if event_file_path is None:
            # This case should technically be covered by the 'if not event_files' check above,
            # but adding an explicit check for robustness.
             continue


        print(f"Processing: Task {task_number}, Seed {seed_number} (from '{item_name}')")
        try:
            ea = event_accumulator.EventAccumulator(event_file_path,
                size_guidance={'scalars': 0}, purge_orphaned_data=True)
            ea.Reload()
            tags = ea.Tags().get("scalars", [])

            success_tag = None
            # Prioritize tags that look exactly like 'average_task_success_rate/...'
            exact_match_tags = [t for t in tags if t.endswith('/average_task_success_rate')]
            if exact_match_tags:
                success_tag = min(exact_match_tags, key=len) # Pick shortest if multiple e.g. train/eval
            else:
                # Fallback to simple substring search if no exact match
                potential_tags = [t for t in tags if 'average_task_success_rate' in t]
                if potential_tags:
                    success_tag = min(potential_tags, key=len) # Pick shortest among potential

            if success_tag is None:
                print(f"  Warning: Could not find success rate tag (e.g., '.../average_task_success_rate') for '{item_name}'. Available tags: {tags}. Skipping.")
                continue

            values = ea.Scalars(success_tag)
            if not values:
                print(f"  Warning: No data points found for tag '{success_tag}' in '{item_name}'. Skipping.")
                continue

            # Check structure of the first element (more robust check)
            if not (hasattr(values[0], 'step') and hasattr(values[0], 'value')):
                 print(f"  Error: Scalar data missing 'step' or 'value' attribute in '{item_name}'. Skipping.")
                 continue


            df = pd.DataFrame({
                'success': [x.value for x in values],
                'frame': [x.step * FRAME_MULTIPLIER for x in values]
            })

            df.dropna(inplace=True)
            df = df[df['success'] >= 0] # Keep only non-negative success rates
            # Keep the last recorded success for any given frame step
            df.drop_duplicates(subset=['frame'], keep='last', inplace=True)
            df.sort_values(by='frame', inplace=True) # Ensure frames are sorted

            if df.empty:
                print(f"  Warning: DataFrame empty after cleaning for '{item_name}'. Skipping.")
                continue

            # Ensure final success is valid number before appending
            final_success = df['success'].iloc[-1]
            if pd.isna(final_success):
                 print(f"  Warning: Final success value is NaN for '{item_name}'. Skipping run for final success calculation, but including curve data if valid.")
            else:
                 task_final_success[task_number].append(final_success)

            # Append the dataframe even if final success was NaN, as long as df is not empty
            results_dataframes[task_number].append(df)
            
            processed_runs += 1

        except Exception as e:
            print(f"  ERROR processing run '{item_name}': {e}")
            import traceback
            traceback.print_exc()

    for task_num in results_dataframes:
        print(f"Task {task_num}: {len(results_dataframes[task_num])} runs processed.")

    print(f"\nFinished processing. Found data for {len(results_dataframes)} tasks across {processed_runs} runs.")
    if not results_dataframes:
        print("No data loaded matching the criteria. Exiting.")
    else:
        # Filter task_final_success to only include tasks that have data in results_dataframes
        valid_final_success = {task: success_list for task, success_list in task_final_success.items() if task in results_dataframes and results_dataframes[task]}
        plot_average_final_success(valid_final_success)
        plot_task_success_grid(results_dataframes)
        print("All plotting complete!")