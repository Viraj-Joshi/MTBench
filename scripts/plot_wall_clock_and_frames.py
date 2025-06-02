import os
from collections import defaultdict
import yaml
import re

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter 
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm

# --- Constants for Statistical Analysis ---
BOOTSTRAP_ITERATIONS = 2000 # Number of bootstrap samples for CI calculation
CI_PERCENT = 95.0 # Confidence Interval percentage

# --- Helper Function for Bootstrapped Confidence Interval ---
def get_percentile_ci(bootstrap_stats_distribution, ci_level=CI_PERCENT):
    """Calculates CI from a pre-computed distribution of bootstrap statistics."""
    bootstrap_stats_distribution = np.asarray(bootstrap_stats_distribution)
    # Corrected NaN filtering for regular numpy arrays
    valid_stats = bootstrap_stats_distribution[~np.isnan(bootstrap_stats_distribution)]
    if valid_stats.size < 2:
        return np.nan, np.nan

    alpha = (100.0 - ci_level) / 2.0
    lower_bound = np.percentile(valid_stats, alpha)
    upper_bound = np.percentile(valid_stats, 100.0 - alpha)
    return lower_bound, upper_bound

def bootstrap_confidence_interval(data, metric_func, n_iterations=BOOTSTRAP_ITERATIONS, ci_level=CI_PERCENT, desc="Bootstrapping CI", disable_tqdm=True):
    """Calculates the bootstrapped confidence interval for a given metric on data."""
    data = np.asarray(data)
    # Corrected NaN filtering for regular numpy arrays
    valid_data = data[~np.isnan(data)]
    if valid_data.size < 2: # Meaningful CI requires at least 2 non-NaN points
        return np.nan, np.nan

    bootstrap_stats = []
    # tqdm is used here. If disable_tqdm is False, it will show a progress bar.
    for _ in tqdm(range(n_iterations), desc=desc, leave=False, ncols=80, disable=disable_tqdm):
        # Ensure valid_data is not empty before choice, though size check above should handle it
        if valid_data.size == 0: # This case should be rare due to the check above
            bootstrap_stats.append(np.nan)
            continue
        sample = np.random.choice(valid_data, size=len(valid_data), replace=True)
        # sample.size check is mostly for safety, choice from non-empty valid_data should yield non-empty sample
        if sample.size == 0: # This case should also be rare
            bootstrap_stats.append(np.nan)
            continue
        stat = metric_func(sample)
        bootstrap_stats.append(stat)

    # Filter NaNs from bootstrap_stats before percentile calculation
    valid_bootstrap_stats = np.array([s for s in bootstrap_stats if not np.isnan(s)])
    if valid_bootstrap_stats.size < 2: # Ensure enough bootstrap stats for CI
        return np.nan, np.nan

    return get_percentile_ci(valid_bootstrap_stats, ci_level)


def convert_to_elapsed_hours(df):
    """
    Converts wall clock timestamps to elapsed hours starting from 0

    Parameters:
    df (pandas.DataFrame): DataFrame (timestamp, success_rate)

    Returns:
    pandas.DataFrame: Modifies input DataFrame with (success_rate, frames, elapsed_hours) tuples
    """
    if 'wall_time' not in df.columns or df['wall_time'].empty:
        return
    start_time = df['wall_time'].min()
    timestamps = df['wall_time']
    elapsed_hours = (timestamps - start_time) / 3600  # Convert seconds to hours
    df['wall_time'] = elapsed_hours

def adjust_data(df): # df is the DataFrame for a single run, after convert_to_elapsed_hours
    """
    Adjusts the DataFrame by:
    1. Adding a row at index 0 with wall_time = 0, success = 0, frame = 0.
    2. Ensuring original data points (especially the first) have positive, spaced-out wall_times
       if they were all zero or clumped at zero, to facilitate plotting.
    """
    df_copy = df.copy()

    if not df_copy.empty:
        all_times_near_zero = (df_copy['wall_time'].abs() < 1e-9).all()
        if all_times_near_zero:
            min_step = 0.0001
            current_time_offset = 0
            for idx in df_copy.index:
                current_time_offset += min_step
                df_copy.loc[idx, 'wall_time'] = current_time_offset
        elif df_copy['wall_time'].iloc[0] < 1e-9: # First point is at zero, but not all are
            if len(df_copy) > 1:
                if df_copy['wall_time'].iloc[1] < 1e-5 : # Next point also very close
                    df_copy.loc[df_copy.index[0], 'wall_time'] = 0.00005 # Smallest initial step
                else: # Next point is further out
                    df_copy.loc[df_copy.index[0], 'wall_time'] = df_copy['wall_time'].iloc[1] / 2.0
            else: # Only one point, and it was at zero
                df_copy.loc[df_copy.index[0], 'wall_time'] = 0.0001

    new_row = pd.DataFrame({'success': [0.0], 'wall_time': [0.0], 'frame': [0]})
    result_df = pd.concat([new_row, df_copy], ignore_index=True)
    return result_df

# Custom tick formatter for Millions and Billions
def M_B_formatter(x, pos):
    if x == 0:
        return '0'
    abs_x = abs(x)
    if abs_x < 1e-9:
        return '0'
    if abs_x < 1e9:
        if abs_x < 1e3:
            return f'{x:.1e}' if (abs_x > 1e-6 and abs_x < 1) else (f'{x:.3f}' if abs_x >=1 else '0')
        return f'{x/1e6:.0f}M'
    else:
        return f'{x/1e9:.0f}B'

def plot_metrics(results_dict):
    """
    Plots success rate vs both wall clock time and frame count with bootstrapped
    confidence interval bounds in a 2x2 grid, separating MT10 and MT50 benchmarks.
    The MT10-Frames plot uses a symlog scale for the x-axis.
    """
    SMALL_SIZE = 20
    MEDIUM_SIZE = 25
    BIGGER_SIZE = 30
    MT10_FRAMES_XTICK_FONTSIZE = SMALL_SIZE - 6 # e.g., 14
    # Fontsize for the scientific notation offset text (e.g., "1e9")
    SCIENTIFIC_OFFSET_FONTSIZE = SMALL_SIZE - 4 # e.g. 16, making it larger than default

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    colors = {'MTSAC': '#3690ff', 'MTPPO': '#fba501', 'MTPQN': '#fb6347', 'MTGRPO': '#2ed573'}
    LINE_WIDTH = 3.5

    time_window_sizes = {
        "MTSAC-MT10": 0.05, "MTSAC-MT50": 0.05,
        "MTPPO-MT10": 0.01, "MTPPO-MT50": 0.05,
        "MTPQN-MT10": 0.05,
        "MTGRPO-MT10": 0.05, "MTGRPO-MT50": 0.05
    }

    mt10_results = {k: v for k, v in results_dict.items() if "MT10" in k}
    mt50_results = {k: v for k, v in results_dict.items() if "MT50" in k}

    max_x_val_mt10_frames_overall = 0
    if mt10_results:
        for exp_name_key in mt10_results.keys():
            run_data_frames_list = results_dict.get(exp_name_key, [])
            if run_data_frames_list:
                for df_original in run_data_frames_list:
                    if not df_original.empty and 'frame' in df_original.columns and not df_original['frame'].empty:
                        max_x_val_mt10_frames_overall = max(max_x_val_mt10_frames_overall, df_original['frame'].max())

    def process_benchmark(benchmark_results, row_idx):
        for exp_name, run_data_list_orig in benchmark_results.items():
            if not run_data_list_orig:
                print(f"Skipping {exp_name} due to no run data.")
                continue

            algo = exp_name.split('-')[0]
            processed_run_data = []
            for single_run_df in run_data_list_orig:
                if isinstance(single_run_df, pd.DataFrame) and not single_run_df.empty:
                    processed_run_data.append(adjust_data(single_run_df))

            if not processed_run_data:
                print(f"Skipping {exp_name} as all run data became empty after adjustment.")
                continue

            # Process wall clock time plot
            valid_run_data_for_time = [df for df in processed_run_data if 'wall_time' in df.columns and not df['wall_time'].empty and df['wall_time'].max() > 1e-9]
            if not valid_run_data_for_time:
                print(f"Skipping wall clock plot for {exp_name} due to no valid time data after processing.")
            else:
                max_time = min(df['wall_time'].max() for df in valid_run_data_for_time)
                current_time_window_size = time_window_sizes.get(exp_name, 0.05)
                if current_time_window_size <= 0: current_time_window_size = 0.001

                if max_time < current_time_window_size / 10 and max_time > 0:
                    time_bins = np.array([0, max_time * 1.05])
                elif max_time > 0 :
                    time_bins = np.arange(0, max_time + current_time_window_size, current_time_window_size)
                else:
                    print(f"Cannot create time_bins for {exp_name} as max_time ({max_time}) is not positive.")
                    time_bins = np.array([])

                time_interpolated_data = []
                if len(time_bins) > 1:
                    for run_df in valid_run_data_for_time:
                        time_binned_success = []
                        for i in range(len(time_bins)-1):
                            mask = (run_df['wall_time'] >= time_bins[i]) & (run_df['wall_time'] < time_bins[i+1])
                            if mask.any():
                                time_binned_success.append(run_df.loc[mask, 'success'].mean())
                            else:
                                time_binned_success.append(np.nan)
                        time_interpolated_data.append(time_binned_success)

                if not time_interpolated_data or not any(any(pd.notna(val) for val in sublist) for sublist in time_interpolated_data if sublist):
                    print(f"No binnable time data after processing for {exp_name}")
                else:
                    time_interpolated = np.array(time_interpolated_data)
                    if time_interpolated.ndim == 1 and len(time_interpolated_data) == 1 and isinstance(time_interpolated_data[0], list):
                        time_interpolated = np.array(time_interpolated_data)

                    if time_interpolated.size == 0 or (time_interpolated.ndim > 1 and time_interpolated.shape[1] == 0):
                         print(f"Time interpolated array is effectively empty for {exp_name}")
                    else:
                        time_mean = np.nanmean(time_interpolated, axis=0)
                        time_ci_low = np.full(time_mean.shape, np.nan)
                        time_ci_high = np.full(time_mean.shape, np.nan)

                        for i in range(time_interpolated.shape[1]):
                            bin_data_for_ci = time_interpolated[:, i]
                            low, high = bootstrap_confidence_interval(
                                bin_data_for_ci, metric_func=np.mean, disable_tqdm=True
                            )
                            time_ci_low[i] = low
                            time_ci_high[i] = high

                        valid_time_mask = ~np.isnan(time_mean)
                        plottable_time_bins = time_bins[:-1] if len(time_bins)>1 else np.array([])

                        if plottable_time_bins.size > 0 and np.any(valid_time_mask):
                            valid_time_bins = plottable_time_bins[valid_time_mask]
                            valid_time_mean = time_mean[valid_time_mask]
                            valid_time_ci_low = time_ci_low[valid_time_mask]
                            valid_time_ci_high = time_ci_high[valid_time_mask]

                            if valid_time_bins.size > 0 :
                                axes[row_idx, 0].plot(valid_time_bins, valid_time_mean, label=algo[:2]+"-"+algo[2:], color=colors.get(algo, 'k'), linewidth=LINE_WIDTH)
                                axes[row_idx, 0].fill_between(valid_time_bins,
                                                              valid_time_ci_low,
                                                              valid_time_ci_high,
                                                              alpha=0.2, color=colors.get(algo, 'k'))
                        else:
                            print(f"No valid time bins to plot after masking for {exp_name}")

            # Process frame count plot
            valid_run_data_for_frame = [df for df in processed_run_data if 'frame' in df.columns and 'success' in df.columns and not df.empty]
            if not valid_run_data_for_frame:
                 print(f"Skipping frame plot for {exp_name} due to missing columns or empty dataframes after adjustment.")
            else:
                base_frames_series = valid_run_data_for_frame[0]['frame']
                if not isinstance(base_frames_series, pd.Series) or base_frames_series.empty:
                    print(f"Base frames from first run are invalid for {exp_name}. Skipping frame plot.")
                    continue
                exp_frames = np.sort(np.unique(base_frames_series.values))

                if exp_frames.size == 0:
                    print(f"No experiment frames found for {exp_name}. Skipping frame plot.")
                    continue

                aligned_frame_successes = []
                for run_df in valid_run_data_for_frame:
                    current_run_frames = run_df['frame'].values
                    current_run_success = run_df['success'].values
                    sort_indices = np.argsort(current_run_frames)
                    sorted_current_run_frames = current_run_frames[sort_indices]
                    sorted_current_run_success = current_run_success[sort_indices]

                    right_val_interp = 0.0
                    if sorted_current_run_success.size > 0:
                        right_val_interp = sorted_current_run_success[-1]

                    if sorted_current_run_frames.size == 0:
                        interp_success = np.full(exp_frames.shape, np.nan if exp_frames.size >0 else 0.0)
                    else:
                        interp_success = np.interp(exp_frames, sorted_current_run_frames, sorted_current_run_success, left=0.0, right=right_val_interp)
                    aligned_frame_successes.append(interp_success)

                if not aligned_frame_successes:
                    print(f"Skipping frame plot for {exp_name} as no data after aligning frames.")
                else:
                    frame_successes_np = np.array(aligned_frame_successes)
                    frame_mean = np.mean(frame_successes_np, axis=0)
                    frame_ci_low = np.full(frame_mean.shape, np.nan)
                    frame_ci_high = np.full(frame_mean.shape, np.nan)

                    for i in range(frame_successes_np.shape[1]):
                        frame_data_for_ci = frame_successes_np[:, i]
                        low, high = bootstrap_confidence_interval(
                            frame_data_for_ci, metric_func=np.mean, disable_tqdm=True
                        )
                        frame_ci_low[i] = low
                        frame_ci_high[i] = high

                    axes[row_idx, 1].plot(exp_frames, frame_mean, label=algo[:2]+"-"+algo[2:], color=colors.get(algo, 'k'), linewidth=LINE_WIDTH)
                    axes[row_idx, 1].fill_between(exp_frames,
                                                  frame_ci_low,
                                                  frame_ci_high,
                                                  alpha=0.2, color=colors.get(algo, 'k'))

    # Process MT10 data (top row)
    process_benchmark(mt10_results, 0)

    # Apply symlog scale and custom ticks to MT10-Frames plot (axes[0, 1])
    ax_mt10_frames = axes[0, 1]
    symlog_linthresh = 250e6 # Linear part up to 250 Million

    if max_x_val_mt10_frames_overall > symlog_linthresh:
        print(f"Applying symlog scale to MT10-Frames plot. Max frames: {max_x_val_mt10_frames_overall/1e6:.0f}M, Linthresh: {symlog_linthresh/1e6:.0f}M")
        ax_mt10_frames.set_xscale('symlog', linthresh=symlog_linthresh, base=10)

        # User-specified ticks
        combined_ticks = np.array([0, 50e6, 100e6, 200e6, 1e9, 2e9, 4e9, 6e9])
        # Filter ticks to be within the actual data range for cleanliness
        combined_ticks = combined_ticks[combined_ticks <= max_x_val_mt10_frames_overall * 1.01] # *1.01 for a little buffer
        if 0 not in combined_ticks : combined_ticks = np.insert(combined_ticks, 0, 0) # Ensure 0 is a tick
        combined_ticks = np.unique(combined_ticks)


        ax_mt10_frames.set_xticks(combined_ticks)
        ax_mt10_frames.xaxis.set_major_formatter(FuncFormatter(M_B_formatter))
        ax_mt10_frames.tick_params(axis='x', labelsize=MT10_FRAMES_XTICK_FONTSIZE, rotation=0)

        current_plot_min_x = -0.01 * symlog_linthresh
        ax_mt10_frames.set_xlim(left=current_plot_min_x, right=max_x_val_mt10_frames_overall * 1.05)

    else:
        ax_mt10_frames.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        try:
            offset_text_obj = ax_mt10_frames.xaxis.get_offset_text()
            if offset_text_obj.get_text() != '':
                 offset_text_obj.set_fontsize(SCIENTIFIC_OFFSET_FONTSIZE) # Use defined larger size
        except AttributeError:
            pass
        ax_mt10_frames.set_xlim(left=0)

    # Process MT50 data (bottom row)
    process_benchmark(mt50_results, 1)

    for i in range(2):
        for j in range(2):
            axes[i, j].grid(True, alpha=0.3)
            axes[i, j].set_facecolor('#f2f2f2')
            for spine in axes[i, j].spines.values():
                spine.set_visible(False)

            if not (i == 0 and j == 1 and max_x_val_mt10_frames_overall > symlog_linthresh):
                axes[i, j].tick_params(axis='x', labelsize=SMALL_SIZE)
            axes[i, j].tick_params(axis='y', labelsize=SMALL_SIZE)

            if j == 0:
                axes[i, j].set_ylabel('Success Rate', fontsize=MEDIUM_SIZE)

            if j == 1:
                is_mt10_frames_symlog = (i == 0 and max_x_val_mt10_frames_overall > symlog_linthresh)
                if not is_mt10_frames_symlog: # Not the MT10 symlog plot
                    axes[i, j].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                    try:
                        offset_text_obj = axes[i, j].xaxis.get_offset_text()
                        if offset_text_obj.get_text() != '':
                             offset_text_obj.set_fontsize(SCIENTIFIC_OFFSET_FONTSIZE) # Apply larger font size here
                    except AttributeError:
                        pass
            axes[i, j].legend(fontsize=SMALL_SIZE)

    for j_idx in range(2):
        axes[1, j_idx].set_xlabel('Wall Clock Time (hours)' if j_idx == 0 else 'Frames', fontsize=MEDIUM_SIZE)

    axes[0, 0].text(-0.2, 0.5, "MT10", fontsize=BIGGER_SIZE, fontweight='bold',
                    transform=axes[0, 0].transAxes, va='center', ha='center', rotation=90)
    axes[1, 0].text(-0.2, 0.5, "MT50", fontsize=BIGGER_SIZE, fontweight='bold',
                    transform=axes[1, 0].transAxes, va='center', ha='center', rotation=90)

    plt.tight_layout(rect=[0.03, 0, 1, 1])

    output_dir = "scripts/figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, "mt10_mt50_comparison.pdf"))
    plt.savefig(os.path.join(output_dir, "mt10_mt50_comparison.png"))
    print(f"Plots saved: {output_dir}/mt10_mt50_comparison.pdf/png")


if __name__ == "__main__":
    log_dir = "runs/"

    if not os.path.exists(log_dir) or not os.listdir(log_dir):
        print(f"Log directory '{log_dir}' not found or is empty. Please create/populate it.")
        if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)

    _listdir = []
    if os.path.exists(log_dir) and os.path.isdir(log_dir):
        _listdir = os.listdir(log_dir)
    else:
        print(f"Warning: Log directory '{log_dir}' does not exist or is not a directory. Patterns will likely find no files.")

    runname_to_exps = {
        "MTPPO-MT10": [f for f in _listdir if "05_09_ppo_vanilla_mt10_rand_envs" in f],
        "MTPQN-MT10": [f for f in _listdir if "vanilla_pqn_rand_long" in f and "mt50" not in f.lower()],
        "MTSAC-MT10": [f for f in _listdir if "vanilla_sac_rand_long" in f and "mt50" not in f.lower()],
        "MTGRPO-MT10": [f for f in _listdir if "05_26_grpo_vanilla_mt10_rand" in f],

        "MTSAC-MT50": [f for f in _listdir if "vanilla_sac_mt50_rand" in f and 'seed_46' not in f],
        "MTPPO-MT50": [f for f in _listdir if re.search(r"^ppo_vanilla_mt50_rand_envs",f)],
        "MTGRPO-MT50": [f for f in _listdir if "05_26_grpo_vanilla_mt50_rand" in f],
    }

    results = defaultdict(list)

    for exp_name, run_names in sorted(runname_to_exps.items()):
        print(f"Processing {exp_name}...")
        if not run_names:
            print(f"No run files found for {exp_name} based on patterns in '{log_dir}'.")
            continue
        for run_name in run_names:
            sp_options = []
            current_algo_type_for_path = exp_name.split('-')[0].lower()
            if "ppo" in current_algo_type_for_path or "grpo" in current_algo_type_for_path:
                sp_options.append(os.path.join(log_dir, run_name, "summaries"))
            sp_options.append(os.path.join(log_dir, run_name))

            event_file_path = None
            for sp_try in sp_options:
                if os.path.isdir(sp_try):
                    try:
                        event_files = [f for f in os.listdir(sp_try) if f.startswith("events.out.tfevents")]
                        if event_files:
                            event_file_path = os.path.join(sp_try, sorted(event_files)[-1])
                            break
                    except FileNotFoundError:
                        print(f"Warning: Path {sp_try} not found during event file search for {run_name}.")
                        continue
                    except Exception as e_list:
                        print(f"Warning: Error listing files in {sp_try} for {run_name}: {e_list}")
                        continue

            if not event_file_path:
                continue

            print(f"Processing {run_name} from {event_file_path}")
            try:
                ea = event_accumulator.EventAccumulator(event_file_path, size_guidance={'scalars': 0})
                ea.Reload()
                tags = ea.Tags().get("scalars", [])

                success_tag_key = None
                current_algo_type = exp_name.split('-')[0].lower()

                if "ppo" in current_algo_type or "pqn" in current_algo_type or "grpo" in current_algo_type:
                    for tag in tags:
                        if tag.startswith("Episode/average_task_success_rate"):
                            success_tag_key = tag; break
                elif "sac" in current_algo_type:
                     for tag in tags:
                        if tag.startswith("episode/average_task_success_rate/frame"):
                            success_tag_key = tag; break

                if not success_tag_key:
                    for tag in tags:
                        if tag.startswith("Episode/average_task_success_rate"): success_tag_key = tag; break
                    if not success_tag_key:
                         for tag in tags:
                            if tag.startswith("episode/average_task_success_rate"): success_tag_key = tag; break

                if not success_tag_key:
                    print(f"Could not find success rate tag for {run_name} (algo type {current_algo_type}). Available tags (first 5): {tags[:5]}...")
                    continue

                values = ea.Scalars(success_tag_key)
                if not values:
                    print(f"No data for tag '{success_tag_key}' in {run_name}")
                    continue

                # Frame multipliers
                frame_multiplier = 1
                if "ppo" in current_algo_type:
                    frame_multiplier = 24576*32
                elif "pqn" in current_algo_type:
                    frame_multiplier = 8192*16
                elif "grpo" in current_algo_type:
                    frame_multiplier = 4096*150 if "MT10" in exp_name else 24576*150
                elif "sac" in current_algo_type:
                    frame_multiplier = 1


                df = pd.DataFrame({
                    'success': [x.value for x in values],
                    'wall_time': [x.wall_time for x in values],
                    'frame': [x.step * frame_multiplier if frame_multiplier != 1 else x.step for x in values]
                })

                if df.empty:
                    print(f"DataFrame empty after loading data for {run_name}")
                    continue

                convert_to_elapsed_hours(df)
                results[exp_name].append(df)

            except Exception as e:
                print(f"Error processing {run_name}: {e}")
                import traceback
                traceback.print_exc()

    print("Starting plotting...")
    if not results or not any(results.values()):
        print("No results were loaded. Skipping plotting.")
    else:
        plot_metrics(results)
    print("Done!")