import os
from collections import defaultdict
import glob
import multiprocessing

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm

# --- Constants ---
BOOTSTRAP_ITERATIONS = 1000
CI_PERCENT = 95.0

# --- Plotting Constants ---
TITLE_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 22
TICK_LABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 16

# ======================================================================================
# UTILITY FUNCTIONS
# ======================================================================================
def get_percentile_ci(bootstrap_stats_distribution, ci_level=CI_PERCENT):
    valid_stats = np.asarray(bootstrap_stats_distribution)
    valid_stats = valid_stats[~np.isnan(valid_stats)]
    if valid_stats.size < 2: return np.nan, np.nan
    alpha = (100.0 - ci_level) / 2.0
    return np.percentile(valid_stats, alpha), np.percentile(valid_stats, 100.0 - alpha)

def bootstrap_confidence_interval(data, metric_func=np.mean, n_iterations=BOOTSTRAP_ITERATIONS, ci_level=CI_PERCENT):
    valid_data = np.asarray(data)[~np.isnan(np.asarray(data))]
    if valid_data.size < 2: return np.nan, np.nan, np.nan
    bootstrap_stats = [metric_func(np.random.choice(valid_data, size=len(valid_data), replace=True)) for _ in range(n_iterations)]
    mean_stat = metric_func(valid_data)
    lower_bound, upper_bound = get_percentile_ci(bootstrap_stats, ci_level)
    return mean_stat, lower_bound, upper_bound

def load_run_timeseries_dual_axis(args):
    """Loads a time series and calculates both frames and gradient updates as x-axes."""
    plot_label, run_path, success_tag, gradients_per_epoch, frames_per_epoch = args
    try:
        ea = event_accumulator.EventAccumulator(run_path, size_guidance={'scalars': 0})
        ea.Reload()

        scalar_events = ea.Scalars(success_tag)
        if not scalar_events:
            return plot_label, None

        epochs = np.array([event.step for event in scalar_events])
        gradient_updates = epochs * gradients_per_epoch
        frames = epochs * frames_per_epoch
        success = np.array([event.value for event in scalar_events])

        df = pd.DataFrame({'gradient_updates': gradient_updates, 'frames': frames, 'success': success})
        start_point = pd.DataFrame({'gradient_updates': [0], 'frames': [0], 'success': [0.0]})
        df = pd.concat([start_point, df], ignore_index=True).sort_values(by='frames').drop_duplicates()
        return plot_label, df

    except Exception:
        return plot_label, None

# ======================================================================================
# PLOTTING
# ======================================================================================
def plot_dual_efficiency_curves(processed_data, styles_map, output_dir):
    """Generates and saves a dual-panel plot for sample and gradient efficiency."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # --- Plot 1: Sample Efficiency (vs. Frames) ---
    ax1 = axes[0]
    ax1.set_title("Sample Efficiency", fontsize=TITLE_FONTSIZE, pad=20)
    for plot_label, style in styles_map.items():
        if plot_label in processed_data and 'sample_efficiency' in processed_data[plot_label]:
            data = processed_data[plot_label]['sample_efficiency']
            ax1.plot(data['x_values'], data['mean'], color=style['color'], label=style['label'], linewidth=3, zorder=3)
            ax1.fill_between(data['x_values'], data['low'], data['high'], color=style['color'], alpha=0.2, zorder=2)

    # --- Plot 2: Gradient Efficiency (vs. Gradient Updates) ---
    ax2 = axes[1]
    ax2.set_title("Gradient Efficiency", fontsize=TITLE_FONTSIZE, pad=20)
    for plot_label, style in styles_map.items():
        if plot_label in processed_data and 'gradient_efficiency' in processed_data[plot_label]:
            data = processed_data[plot_label]['gradient_efficiency']
            ax2.plot(data['x_values'], data['mean'], color=style['color'], label=style['label'], linewidth=3, zorder=3)
            ax2.fill_between(data['x_values'], data['low'], data['high'], color=style['color'], alpha=0.2, zorder=2)

    # --- Formatting ---
    def custom_formatter(x, pos):
        if x == 0: return '0'
        if x >= 1e9: return f'{x*1e-9:g}B'
        if x >= 1e6: return f'{x*1e-6:g}M'
        return f'{x*1e-3:g}K'

    # Format ax1 (Frames)
    ax1.set_xlabel("Frames", fontsize=AXIS_LABEL_FONTSIZE, labelpad=15)
    ax1.set_ylabel("Average Success Rate", fontsize=AXIS_LABEL_FONTSIZE, labelpad=15)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE, pad=5)
    ax1.set_xscale('symlog', linthresh=100_000_000) # Use symlog scale
    custom_ticks_frames = [0, 100e6, 250e6, 500e6, 1000e6, 3000e6, 5000e6]
    ax1.set_xticks(custom_ticks_frames) # Set custom ticks
    ax1.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    ax1.grid(True, which='major', axis='both', linestyle='-', color='gainsboro', zorder=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Format ax2 (Gradient Updates)
    ax2.set_xlabel("Gradient Updates", fontsize=AXIS_LABEL_FONTSIZE, labelpad=15)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE, pad=5)
    ax2.set_xscale('symlog', linthresh=1000)
    custom_ticks_grads = [0, 1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 400_000, 800_000]
    ax2.set_xticks(custom_ticks_grads)
    ax2.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    ax2.grid(True, which='major', axis='both', linestyle='-', color='gainsboro', zorder=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(fontsize=LEGEND_FONTSIZE, loc='best')

    fig.tight_layout()
    save_path_png = os.path.join(output_dir, "dual_efficiency_curves.png")
    save_path_pdf = os.path.join(output_dir, "dual_efficiency_curves.pdf")
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"\nDual efficiency plot saved to {output_dir}")

# ======================================================================================
# MAIN EXECUTION
# ======================================================================================
if __name__ == "__main__":
    LOG_DIR = "runs/"
    OUTPUT_DIR = "scripts/parameter_scaling/"
    SUCCESS_TAG = "Episode/average_task_success_rate"

    # --- 1. Define Default Calculation Constants ---
    K = 5
    m = 32768
    n = 4096      # Default number of environments
    T = 32

    # --- 2. Define Experiments to Plot ---
    EXPERIMENTS_TO_PLOT = {
        "Multi-Task MLP (1M)": {
            'type': 'multi-task',
            'n': 4096,
            'pattern': os.path.join(LOG_DIR, "ppo_vanilla_mt50_rand_scaling/4096", "*_width_1_*"),
            'style': {'color': 'royalblue', 'marker': 'o', 'label': 'Multi-Task MLP (1M)'}
        },
        "SimbaV2 (1M)": {
            'type': 'multi-task',
            'n': 24576,
            'pattern': os.path.join(LOG_DIR, "ppo_simbav2_mt50_rand_scaling", "*_width_1_*"),
            'style': {'color': 'mediumseagreen', 'marker': '^', 'label': 'Multi-Task SimbaV2 (1M)'}
        },
        "Single-Task Average (1M)": {
            'type': 'single-task',
            'n': 4096,
            'pattern': os.path.join(LOG_DIR, "single-task", "uid_ppo_vanilla_task_*"),
            'style': {'color': 'firebrick', 'marker': 's', 'label': 'Single-Task Average (1M)'}
        }
    }

    STYLES_MAP = {label: config['style'] for label, config in EXPERIMENTS_TO_PLOT.items()}

    # --- 3. Find and Prepare Tasks for Data Loading ---
    tasks = []
    print("Finding event files...")
    for plot_label, config in EXPERIMENTS_TO_PLOT.items():
        num_envs = config.get('n', n)

        frames_per_epoch = num_envs * T
        grad_updates_per_epoch = K * (num_envs * T) / m

        if config['type'] == 'single-task':
            grad_updates_per_epoch *= 50
            frames_per_epoch *= 50

        run_dirs = glob.glob(config['pattern'])
        if not run_dirs:
            print(f"  - Warning: No runs found for '{plot_label}' using pattern '{config['pattern']}'")
            continue

        print(f"  - Found {len(run_dirs)} runs for '{plot_label}'")
        for run_dir in run_dirs:
            event_files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
            if event_files:
                tasks.append((plot_label, sorted(event_files)[-1], SUCCESS_TAG, grad_updates_per_epoch, frames_per_epoch))

    # --- 4. Load Data and Process for Plotting ---
    run_data = defaultdict(list)
    if not tasks:
        print("\nError: No event files were found.")
    else:
        print(f"\nLoading time-series data from {len(tasks)} runs...")
        with multiprocessing.Pool() as pool:
            pbar = tqdm(pool.imap_unordered(load_run_timeseries_dual_axis, tasks), total=len(tasks))
            for plot_label, df in pbar:
                if df is not None and not df.empty:
                    run_data[plot_label].append(df)

    processed_data = defaultdict(dict)
    print("\nAligning data for both axes and calculating confidence intervals...")
    for plot_label, df_list in run_data.items():
        if not df_list: continue

        # Process for Sample Efficiency (Frames)
        all_frames = set()
        for df in df_list: all_frames.update(df['frames'].values)
        if all_frames:
            unified_frames = np.sort(list(all_frames))
            aligned_data = np.array([np.interp(unified_frames, df['frames'], df['success'], right=df['success'].iloc[-1]) for df in df_list])

            mean_curve, low_curve, high_curve = [], [], []
            for i in range(aligned_data.shape[1]):
                mean, low, high = bootstrap_confidence_interval(aligned_data[:, i])
                mean_curve.append(mean); low_curve.append(low); high_curve.append(high)

            processed_data[plot_label]['sample_efficiency'] = {
                'x_values': unified_frames, 'mean': np.array(mean_curve),
                'low': np.array(low_curve), 'high': np.array(high_curve)
            }

        # Process for Gradient Efficiency (Updates)
        all_grad_updates = set()
        for df in df_list: all_grad_updates.update(df['gradient_updates'].values)
        if all_grad_updates:
            unified_gradients = np.sort(list(all_grad_updates))
            aligned_data = np.array([np.interp(unified_gradients, df['gradient_updates'], df['success'], right=df['success'].iloc[-1]) for df in df_list])

            mean_curve, low_curve, high_curve = [], [], []
            for i in range(aligned_data.shape[1]):
                mean, low, high = bootstrap_confidence_interval(aligned_data[:, i])
                mean_curve.append(mean); low_curve.append(low); high_curve.append(high)

            processed_data[plot_label]['gradient_efficiency'] = {
                'x_values': unified_gradients, 'mean': np.array(mean_curve),
                'low': np.array(low_curve), 'high': np.array(high_curve)
            }

    # --- 5. Generate the Plot ---
    if processed_data:
        plot_dual_efficiency_curves(processed_data, STYLES_MAP, OUTPUT_DIR)
    else:
        print("\nError: No valid data was processed.")