# Filename: generate_specific_learning_curves_v2.py

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
PLOT_INTERVAL = 100_000_000  # Evaluate performance every 100M frames

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

def load_run_timeseries(args):
    """Loads the full time series (frames, success) from a single run."""
    setting, plot_label, run_path, success_tag_key, frame_multiplier = args
    try:
        ea = event_accumulator.EventAccumulator(run_path, size_guidance={'scalars': 0})
        ea.Reload()
        
        scalar_events = ea.Scalars(success_tag_key)
        if not scalar_events:
            return setting, plot_label, None

        frames = [event.step * frame_multiplier for event in scalar_events]
        success = [event.value for event in scalar_events]
        
        df = pd.DataFrame({'frame': frames, 'success': success})
        start_point = pd.DataFrame({'frame': [0], 'success': [0.0]})
        df = pd.concat([start_point, df], ignore_index=True)
        return setting, plot_label, df
        
    except Exception:
        return setting, plot_label, None

# ======================================================================================
# PLOTTING
# ======================================================================================
def plot_learning_curves(processed_data, variant_styles, output_dir):
    """Generates and saves learning curve plots."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    # --- MODIFICATION: Font size configuration ---
    TITLE_FONTSIZE = 20
    AXIS_LABEL_FONTSIZE = 20
    TICK_LABEL_FONTSIZE = 20
    LEGEND_FONTSIZE = 14

    settings = ["MT10 rand", "MT50 rand"]
    for i, setting in enumerate(settings):
        ax = axes[i]
        ax.set_title(setting, fontsize=TITLE_FONTSIZE, pad=15)

        if setting in processed_data:
            for plot_label in variant_styles:
                if plot_label not in processed_data[setting]:
                    continue
                
                data = processed_data[setting][plot_label]
                style = variant_styles[plot_label]
                frames = data['frames']
                means = data['mean']
                ci_low = data['low']
                ci_high = data['high']
                
                ax.plot(frames, means, color=style['color'], label=style['label'], linewidth=2.5)
                ax.fill_between(frames, ci_low, ci_high, color=style['color'], alpha=0.2)
                ax.plot(frames, means, marker=style['marker'], color=style['color'], linestyle='None',
                        markersize=8, markeredgecolor='white', markeredgewidth=1.0)

        # --- Formatting ---
        ax.set_xlabel("Frames", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
        ax.grid(True, which='major', axis='both', linestyle='-', color='gainsboro', zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    axes[0].set_ylabel("Success Rate", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    axes[0].legend(fontsize=LEGEND_FONTSIZE, loc='best')

    fig.tight_layout()
    save_path_png = os.path.join(output_dir, "simbav2_vs_mlp.png")
    save_path_pdf = os.path.join(output_dir, "simbav2_vs_mlp.pdf")
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"\nLearning curve plots saved to {output_dir}")

# ======================================================================================
# MAIN EXECUTION
# ======================================================================================
if __name__ == "__main__":
    LOG_DIR = "runs/"
    OUTPUT_DIR = "scripts/parameter_scaling/"
    SUCCESS_TAG = "Episode/average_task_success_rate"

    # --- Configuration ---
    PPO_FRAME_MULTIPLIER = 24576 * 32
    
    # Define which specific variants and sizes to plot
    RUN_MAPPING = {
        "MT10 rand": {
            "MLP (1M)":       {'folder': "ppo_vanilla_mt10_rand_scaling", 'width': 1},
            "SimbaV2 (1M)":   {'folder': "ppo_simbav2_mt10_rand_scaling", 'width': 1},
            "SimbaV2 (4M)":   {'folder': "ppo_simbav2_mt10_rand_scaling", 'width': 2},
            "SimbaV2 (16M)":  {'folder': "ppo_simbav2_mt10_rand_scaling", 'width': 3},
            "SimbaV2 (64M)":  {'folder': "ppo_simbav2_mt10_rand_scaling", 'width': 4},
            "SimbaV2 (256M)": {'folder': "ppo_simbav2_mt10_rand_scaling", 'width': 5}
        },
        "MT50 rand": {
            "MLP (1M)":       {'folder': "ppo_vanilla_mt50_rand_scaling", 'width': 1},
            "SimbaV2 (1M)":   {'folder': "ppo_simbav2_mt50_rand_scaling", 'width': 1},
            "SimbaV2 (4M)":   {'folder': "ppo_simbav2_mt50_rand_scaling", 'width': 2},
            "SimbaV2 (16M)":  {'folder': "ppo_simbav2_mt50_rand_scaling", 'width': 3},
            "SimbaV2 (64M)":  {'folder': "ppo_simbav2_mt50_rand_scaling", 'width': 4},
            "SimbaV2 (256M)": {'folder': "ppo_simbav2_mt50_rand_scaling", 'width': 5}
        }
    }
    
    # Define styles for the specific plot lines
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 6))
    VARIANT_STYLES = {
        "MLP (1M)":       {'color': "dimgray", 'marker': 'o', 'label': "MLP (1M)"},
        "SimbaV2 (1M)":   {'color': colors[0], 'marker': '^', 'label': "SimbaV2 (1M)"},
        "SimbaV2 (4M)":   {'color': colors[1], 'marker': '^', 'label': "SimbaV2 (4M)"},
        "SimbaV2 (16M)":  {'color': colors[2], 'marker': '^', 'label': "SimbaV2 (16M)"},
        "SimbaV2 (64M)":  {'color': colors[3], 'marker': '^', 'label': "SimbaV2 (64M)"},
        "SimbaV2 (256M)": {'color': colors[4], 'marker': '^', 'label': "SimbaV2 (256M)"},
    }

    # --- Find all relevant runs based on the new mapping ---
    tasks = []
    print("Finding event files for specific variants and sizes...")
    for setting, variants_to_plot in RUN_MAPPING.items():
        for plot_label, config in variants_to_plot.items():
            base_folder = config['folder']
            width_index = config['width']
            
            search_pattern = os.path.join(LOG_DIR, base_folder, f"*_width_{width_index}_*")
            run_dirs = glob.glob(search_pattern)
            
            if not run_dirs:
                print(f"  - Warning: No runs found for '{plot_label}' in '{setting}' using pattern '{search_pattern}'")
                continue
                
            for run_dir in run_dirs:
                event_files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
                if event_files:
                    tasks.append((setting, plot_label, sorted(event_files)[-1], SUCCESS_TAG, PPO_FRAME_MULTIPLIER))

    # --- Load all time-series data in parallel ---
    run_data = defaultdict(lambda: defaultdict(list))
    if not tasks:
        print("\nError: No event files were found.")
    else:
        print(f"\nLoading time-series data from {len(tasks)} runs...")
        with multiprocessing.Pool() as pool:
            pbar = tqdm(pool.imap_unordered(load_run_timeseries, tasks), total=len(tasks))
            for setting, plot_label, df in pbar:
                if df is not None and not df.empty:
                    run_data[setting][plot_label].append(df)

    # --- Process and align data for plotting ---
    processed_data = defaultdict(lambda: defaultdict(dict))
    print("\nAligning data and calculating confidence intervals...")
    for setting, variants in run_data.items():
        for plot_label, df_list in variants.items():
            if not df_list: continue

            max_frame = max(df['frame'].max() for df in df_list)
            frame_intervals = np.arange(0, max_frame + PLOT_INTERVAL, PLOT_INTERVAL)
            
            aligned_data = [np.interp(frame_intervals, df['frame'], df['success']) for df in df_list]
            aligned_data = np.array(aligned_data)
            
            mean_curve, ci_low_curve, ci_high_curve = [], [], []
            for i in range(aligned_data.shape[1]):
                mean, low, high = bootstrap_confidence_interval(aligned_data[:, i])
                mean_curve.append(mean)
                ci_low_curve.append(low)
                ci_high_curve.append(high)
                
            processed_data[setting][plot_label] = {
                'frames': frame_intervals, 'mean': np.array(mean_curve),
                'low': np.array(ci_low_curve), 'high': np.array(ci_high_curve),
            }

    # --- Generate the plot ---
    if processed_data:
        plot_learning_curves(processed_data, VARIANT_STYLES, OUTPUT_DIR)
    else:
        print("\nError: No valid data was processed.")