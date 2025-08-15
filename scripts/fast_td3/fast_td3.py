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
CI_PERCENT = 95.0
PLOT_INTERVAL_FRAMES = 10_000_000  # Increased granularity for smoother plots
PLOT_INTERVAL_HOURS = 0.25        # Align data every 15 minutes

# ======================================================================================
# DATA LOADING & MULTIPROCESSING WRAPPER
# ======================================================================================

def run_task_wrapper(task):
    """Helper function to be called by the multiprocessing pool."""
    return task['func'](task['args'])

def load_run_data(args):
    """
    Loads a time series: success, frames, and wall-clock time.
    Handles both direct and calculated frame modes.
    """
    setting, exp_name, event_path, tag, multiplier, is_direct = args
    try:
        ea = event_accumulator.EventAccumulator(event_path, size_guidance={'scalars': 0})
        ea.Reload()

        if tag not in ea.Tags()['scalars']: return setting, exp_name, None
        scalar_events = ea.Scalars(tag)
        if not scalar_events: return setting, exp_name, None

        # Load raw data
        steps = np.array([event.step for event in scalar_events])
        values = np.array([event.value for event in scalar_events])
        wall_times = np.array([event.wall_time for event in scalar_events])
        
        # Calculate frames based on mode
        frames = steps if is_direct else steps * multiplier
        
        # Calculate elapsed time in hours
        elapsed_hours = (wall_times - wall_times[0]) / 3600.0
        
        df = pd.DataFrame({'frame': frames, 'time': elapsed_hours, 'success': values})
        start_point = pd.DataFrame({'frame': [0], 'time': [0.0], 'success': [0.0]})
        df = pd.concat([start_point, df], ignore_index=True).drop_duplicates(subset=['frame', 'time'], keep='last')
        return setting, exp_name, df
    except Exception:
        return setting, exp_name, None

# ======================================================================================
# PLOTTING & TABLE GENERATION
# ======================================================================================

def plot_learning_curves(processed_data, styles, output_dir):
    """Generates a 2x2 plot with custom axes and labels, styled to match the target."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharey='row')
    # This line was removed: fig.set_facecolor('#f2f2f2')

    AXIS_LABEL_FONTSIZE = 24
    TICK_LABEL_FONTSIZE = 18
    LEGEND_FONTSIZE = 20
    Y_AXIS_LABEL_FONTSIZE = 30

    settings = ["MT10", "MT50"]
    
    # --- Populate the 2x2 grid ---
    for row, setting in enumerate(settings):
        ax_time = axes[row, 0]
        ax_frames = axes[row, 1]

        if setting in processed_data:
            sorted_exp_names = [name for name in styles if name in processed_data[setting]]
            for exp_name in sorted_exp_names:
                style = styles[exp_name]
                
                # Plot vs. Time
                time_data = processed_data[setting][exp_name]['time']
                ax_time.plot(time_data['x'], time_data['mean'], color=style['color'], label=style['label'], linewidth=3.0, zorder=3)
                ax_time.fill_between(time_data['x'], time_data['low'], time_data['high'], color=style['color'], alpha=0.1, zorder=2) # Reduced alpha

                # Plot vs. Frames
                frame_data = processed_data[setting][exp_name]['frames']
                ax_frames.plot(frame_data['x'], frame_data['mean'], color=style['color'], label=style['label'], linewidth=3.0, zorder=3)
                ax_frames.fill_between(frame_data['x'], frame_data['low'], frame_data['high'], color=style['color'], alpha=0.1, zorder=2) # Reduced alpha

    # --- Formatting ---
    axes[0, 0].set_ylabel("MT10\n\nSuccess Rate", fontsize=Y_AXIS_LABEL_FONTSIZE, labelpad=15)
    axes[1, 0].set_ylabel("MT50\n\nSuccess Rate", fontsize=Y_AXIS_LABEL_FONTSIZE, labelpad=15)

    axes[1, 0].set_xlabel("Wall Clock Time (hours)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    axes[1, 1].set_xlabel("Frames", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)

    # --- Custom Ticks, Formatters, and Limits ---
    def format_mt10_time_ticks(x, pos):
        if x == 0: return '0'
        if x < 1: return f'{int(x * 60)}min'
        return f'{int(x)}hr'
    
    mt10_time_ticks_hours = [0, 30/60, 1, 2, 3, 4] 
    axes[0, 0].set_xticks(mt10_time_ticks_hours)
    axes[0, 0].xaxis.set_major_formatter(FuncFormatter(format_mt10_time_ticks))
    axes[0, 0].set_xlim(left=0, right=4)

    mt10_frame_ticks_millions = [0, 50, 100, 200, 300, 400, 500]
    mt10_frame_ticks = [val * 1e6 for val in mt10_frame_ticks_millions]
    axes[0, 1].set_xticks(mt10_frame_ticks)
    axes[0, 1].xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    axes[0, 1].set_xlim(left=0, right=500e6)

    axes[1, 0].xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}'))
    axes[1, 0].set_xlim(left=0)

    axes[1, 1].xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    axes[1, 1].set_xlim(left=0, right=1000e6)

    for ax in axes.flatten():
        ax.set_facecolor('#f2f2f2') # Set axes background color
        ax.grid(True, which='major', axis='both', linestyle='-', color='white', linewidth=1.5, zorder=0) # White, thicker grid
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('darkgrey') # Style remaining spines
        ax.spines['left'].set_color('darkgrey')
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE, direction='in', color='darkgrey') # Inward ticks

    # --- Legend ---
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Place legend inside the top-right plot area
    axes[0, 1].legend(handles, labels, fontsize=LEGEND_FONTSIZE, loc='lower right', frameon=False) 

    fig.tight_layout(rect=[0, 0, 1, 1.0]) # Adjust layout to remove space for old legend
    
    save_path_png = os.path.join(output_dir, "fast_td3_efficiency.png")
    save_path_pdf = os.path.join(output_dir, "fast_td3_efficiency.pdf")
    # The 'facecolor' argument is removed from savefig to use the default (white)
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"\n✅ 2x2 comparison plot saved to {output_dir}")


def generate_latex_table(run_data):
    """
    Calculates final performance for Fast TD3 and prints a LaTeX table.
    """
    print("\n" + "="*80)
    print("Generating LaTeX Table for Fast TD3 + SimbaV2 Final Performance...")
    
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Final performance and wall-clock time for Fast TD3 + SimbaV2 (16M). Results show mean $\\pm$ standard deviation across random seeds.}",
        "\\label{tab:fast_td3_performance}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "\\textbf{Setting} & \\textbf{Final Success Rate} & \\textbf{Wall-Clock Time (Hours)} \\\\",
        "\\midrule"
    ]
    
    settings = ["MT10", "MT50"]
    exp_name = "Fast TD3 + SimbaV2"

    for setting in settings:
        df_list = run_data.get(setting, {}).get(exp_name, [])
        
        if not df_list:
            row = f"{setting} & --- & --- \\\\"
        else:
            final_perfs = [df['success'].iloc[-1] for df in df_list]
            final_times = [df['time'].iloc[-1] for df in df_list]
            
            mean_perf, std_perf = np.mean(final_perfs), np.std(final_perfs)
            mean_time, std_time = np.mean(final_times), np.std(final_times)
            
            perf_str = f"${mean_perf:.2f} \\pm {std_perf:.2f}$"
            time_str = f"${mean_time:.2f} \\pm {std_time:.2f}$"
            
            row = f"{setting} & {perf_str} & {time_str} \\\\"
        
        lines.append(row)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    latex_string = "\n".join(lines)
    print("\n--- Generated LaTeX Code ---")
    print("(Requires LaTeX packages: booktabs, amsmath)")
    print(latex_string)
    print("="*80 + "\n")


# ======================================================================================
# MAIN EXECUTION
# ======================================================================================
if __name__ == "__main__":
    LOG_DIR = "runs/"
    OUTPUT_DIR = "scripts/fast_td3/"
    
    # --- STYLES UPDATED ---
    STYLES = {
        "Fast TD3 + SimbaV2": {'color': '#2ed573', 'label': 'Fast TD3 + SimbaV2 (16M)'}, # Bright Green
        "Vanilla":            {'color': '#fba501', 'label': 'Vanilla PPO (16M)'},             # Bright Orange
        "Vanilla TE":         {'color': '#fb6347', 'label': 'Vanilla PPO+TE (16M)'},         # Bright Red
    }

    EXPERIMENTS_TO_PLOT = {
        "MT10": { "Fast TD3 + SimbaV2": { 'pattern': os.path.join(LOG_DIR, "fast_td3_simba_v2_mt10_rand*"), 'tag': 'eval/overall_success_rate', 'direct_x_axis': True }, "Vanilla": { 'pattern': os.path.join(LOG_DIR, "ppo_vanilla_mt10_rand_scaling", "*_width_3_*"), 'tag': 'Episode/average_task_success_rate', 'n': 24576 * 32 }, "Vanilla TE": { 'pattern': os.path.join(LOG_DIR, "ppo_vanilla_TE_mt10_rand_scaling", "*_width_3_*"), 'tag': 'Episode/average_task_success_rate', 'n': 24576 * 32 }, },
        "MT50": { "Fast TD3 + SimbaV2": { 'pattern': "/var/local/viraj/fast_td3_mt50_rand/fast_td3/*", 'tag': 'eval/overall_success_rate', 'direct_x_axis': True }, "Vanilla": { 'pattern': os.path.join(LOG_DIR, "ppo_vanilla_mt50_rand_scaling", "*_width_3_*"), 'tag': 'Episode/average_task_success_rate', 'n': 24576 * 32 }, "Vanilla TE": { 'pattern': os.path.join(LOG_DIR, "ppo_vanilla_TE_mt50_rand_scaling", "*_width_3_*"), 'tag': 'Episode/average_task_success_rate', 'n': 24576 * 32 }, }
    }

    tasks = []
    print("🔍 Finding event files...")
    for setting, experiments in EXPERIMENTS_TO_PLOT.items():
        for exp_name, config in experiments.items():
            run_dirs = glob.glob(config['pattern'])
            if not run_dirs:
                print(f"   - Warning: No runs for '{exp_name}' in {setting} using pattern '{config['pattern']}'")
                continue
            
            print(f"  - Found {len(run_dirs)} runs for '{exp_name}' in {setting}")
            for run_dir in run_dirs:
                event_files = sorted(glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True))
                if event_files:
                    args = (setting, exp_name, event_files[-1], config['tag'], config.get('n'), config.get('direct_x_axis', False))
                    tasks.append({'func': load_run_data, 'args': args})

    run_data = defaultdict(lambda: defaultdict(list))
    if not tasks:
        print("\n❌ Error: No event files were found.")
    else:
        print(f"\n🔄 Loading data from {len(tasks)} runs...")
        with multiprocessing.Pool() as pool:
            results = list(tqdm(pool.imap_unordered(run_task_wrapper, tasks), total=len(tasks)))
            for setting, exp_name, df in results:
                if df is not None and not df.empty:
                    run_data[setting][exp_name].append(df)

    # --- Generate LaTeX Table (New Step) ---
    if run_data:
        generate_latex_table(run_data)

    # --- Process and align data for plotting ---
    processed_data = defaultdict(lambda: defaultdict(dict))
    print("\n📊 Aligning data for frames and time...")
    for setting, experiments in run_data.items():
        for exp_name, df_list in experiments.items():
            if not df_list: continue

            max_frame = max(df['frame'].max() for df in df_list)
            frame_intervals = np.arange(0, max_frame + PLOT_INTERVAL_FRAMES, PLOT_INTERVAL_FRAMES)
            aligned_frames = np.array([np.interp(frame_intervals, df['frame'], df['success']) for df in df_list])
            
            max_time = max(df['time'].max() for df in df_list)
            time_intervals = np.arange(0, max_time + PLOT_INTERVAL_HOURS, PLOT_INTERVAL_HOURS)
            aligned_time = np.array([np.interp(time_intervals, df['time'], df['success']) for df in df_list])
            
            processed_data[setting][exp_name] = {
                'frames': { 'x': frame_intervals, 'mean': np.mean(aligned_frames, axis=0), 'low': np.percentile(aligned_frames, (100 - CI_PERCENT) / 2, axis=0), 'high': np.percentile(aligned_frames, 100 - (100 - CI_PERCENT) / 2, axis=0) },
                'time': { 'x': time_intervals, 'mean': np.mean(aligned_time, axis=0), 'low': np.percentile(aligned_time, (100 - CI_PERCENT) / 2, axis=0), 'high': np.percentile(aligned_time, 100 - (100 - CI_PERCENT) / 2, axis=0) }
            }
            print(f"  - Processed {exp_name} in {setting} ({len(df_list)} runs)")

    if processed_data:
        plot_learning_curves(processed_data, STYLES, OUTPUT_DIR)
    else:
        print("\n❌ Error: No valid data was processed.")