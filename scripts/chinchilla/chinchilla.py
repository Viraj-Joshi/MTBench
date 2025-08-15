import os
import colorsys
from collections import defaultdict
import glob
import multiprocessing

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm

# ======================================================================================
# UTILITY AND DATA LOADING FUNCTIONS
# ======================================================================================
def load_run_timeseries(args):
    """Loads the full time series (frames, success) from a single run."""
    run_path, success_tag, frames_per_epoch = args
    try:
        ea = event_accumulator.EventAccumulator(run_path, size_guidance={'scalars': 0})
        ea.Reload()
        scalar_events = ea.Scalars(success_tag)
        if not scalar_events: return None

        epochs = np.array([event.step for event in scalar_events])
        frames = epochs * frames_per_epoch
        success = np.array([event.value for event in scalar_events])
        
        df = pd.DataFrame({'frame': frames, 'success': success})
        start_point = pd.DataFrame({'frame': [0], 'success': [0.0]})
        df = pd.concat([start_point, df], ignore_index=True)
        return df
    except Exception:
        return None

# ======================================================================================
# GFLOPs CALCULATION
# ======================================================================================
def calculate_mlp_params(width):
    """Estimates parameters for a 3-layer square MLP."""
    return 2 * width**2

def calculate_training_compute(model_width, num_envs, total_frames):
    """Calculates the total training compute in GigaFLOPs."""
    params = calculate_mlp_params(model_width)
    gflops_fwd_per_example = (2 * params) / 1e9
    batch_size = num_envs * HORIZON
    num_gradient_updates = total_frames * PPO_K / PPO_MINIBATCH_SIZE
    total_compute_gflops = gflops_fwd_per_example * batch_size * num_gradient_updates * 3
    return total_compute_gflops

# ======================================================================================
# PLOTTING
# ======================================================================================
def plot_compute_frontiers(processed_data, styles_map, model_sizes, envs, output_dir):
    """Generates and saves the Chinchilla-style plot with an ordered legend."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 10))

    for config in processed_data:
        style = styles_map[config['label']]
        ax.plot(
            config['compute_gflops'], 
            config['error_rates'],
            marker=style['marker'],
            markersize=style['markersize'],
            linestyle='-',
            linewidth=2.5,
            color=style['color'],
            label=config['label']
        )

    # --- Formatting ---
    ax.set_xscale('log')
    ax.set_xlabel("Training Compute (Gflops)", fontsize=24)
    ax.set_ylabel("Average Failure Rate (%)", fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.grid(True, which='both', linestyle='-', color='black', alpha=0.1)
    
    # --- Logic for reordering the legend ---
    handles, labels = ax.get_legend_handles_labels()
    label_handle_map = dict(zip(labels, handles))
    
    ordered_labels = []
    for n_env in envs:
        for size_label in model_sizes:
            label = f'{size_label}/{n_env}'
            if label in label_handle_map:
                ordered_labels.append(label)
    
    ordered_handles = [label_handle_map[lbl] for lbl in ordered_labels]
    
    ax.legend(ordered_handles, ordered_labels, 
              fontsize=16, 
              ncol=len(model_sizes), 
              loc='upper right', 
              frameon=True, 
              framealpha=0.9, 
              title="Model Size / Num. Envs")
    
    fig.tight_layout()
    save_path = os.path.join(output_dir, "compute_vs_performance.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

    output_filename_pdf = os.path.join(output_dir, "compute_vs_performance.pdf")
    plt.savefig(output_filename_pdf, bbox_inches='tight')
    print(f"📈 Plot also saved as '{output_filename_pdf}'")

# ======================================================================================
# MAIN EXECUTION
# ======================================================================================
if __name__ == "__main__":
    LOG_DIR = "runs/"
    OUTPUT_DIR = "scripts/chinchilla/"
    SUCCESS_TAG = "Episode/average_task_success_rate"
    
    # --- PPO & Compute Constants ---
    HORIZON = 32
    PPO_K = 5
    PPO_MINIBATCH_SIZE = 32768
    
    EVALUATION_FRAMES = np.array([100e6, 200e6, 300e6, 400e6, 500e6])

    # --- Define Experiments ---
    MODEL_TYPES = {
        'MLP': 'ppo_vanilla_mt50_rand_scaling',
    }
    
    MODEL_SIZES = {
        'XS': {'width': 768,  'idx': 1},
        'S':  {'width': 1500, 'idx': 2},
        'M':  {'width': 3000, 'idx': 3},
        'L':  {'width': 6000, 'idx': 4},
        'XL': {'width': 12000,'idx': 5}
    }
    
    ENVS = [4096, 8192, 16384, 24576, 32768]
    
    # --- Styling scheme with dynamic marker sizes and shapes ---
    STYLES_MAP = {}
    # <<< MODIFIED: Changed the start of linspace from 0 to 0.15 to make the first color darker >>>
    model_size_colors = plt.cm.magma_r(np.linspace(0.15, 0.9, len(MODEL_SIZES)))
    
    # Marker size corresponds to model capacity (XS -> small, XL -> large)
    model_capacity_marker_sizes = {
        'XS': 8,
        'S':  10,
        'M':  12,
        'L':  14,
        'XL': 16,
    }
    
    for i, (size_label, size_config) in enumerate(MODEL_SIZES.items()):
        for j, n_envs in enumerate(ENVS):
            label = f'{size_label}/{n_envs}'
            STYLES_MAP[label] = {
                'color': model_size_colors[i],
                'markersize': model_capacity_marker_sizes.get(size_label, 8),
                'marker': 'o'
            }

    # --- Generate all experiment configurations ---
    EXPERIMENTS = []
    for model_type, base_folder in MODEL_TYPES.items():
        for size_label, size_config in MODEL_SIZES.items():
            for n_envs in ENVS:
                if n_envs == 24576:
                    pattern = os.path.join(LOG_DIR, base_folder, f"*_width_{size_config['idx']}_*")
                else:
                    pattern = os.path.join(LOG_DIR, base_folder, str(n_envs), f"*_width_{size_config['idx']}_*")
                
                EXPERIMENTS.append({
                    'label': f'{size_label}/{n_envs}',
                    'width': size_config['width'],
                    'n_envs': n_envs,
                    'pattern': pattern
                })
                
    # --- Data Loading and Processing ---
    processed_data_for_plot = []
    all_tasks = []

    print("Finding event files...")
    for exp_config in EXPERIMENTS:
        run_dirs = glob.glob(exp_config['pattern'])
        if not run_dirs: continue
        
        frames_per_epoch = exp_config['n_envs'] * HORIZON
        
        for run_dir in run_dirs:
            event_files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
            if event_files:
                all_tasks.append({
                    "config": exp_config,
                    "task_args": (sorted(event_files)[-1], SUCCESS_TAG, frames_per_epoch)
                })
    
    tasks_by_experiment = defaultdict(list)
    for task in all_tasks:
        tasks_by_experiment[task['config']['label']].append(task)
        
    print(f"\nProcessing {len(tasks_by_experiment)} experiments found...")
    for label, tasks in tqdm(tasks_by_experiment.items(), desc="Processing experiments"):
        exp_config = tasks[0]['config']
        task_args_list = [t['task_args'] for t in tasks]
        
        with multiprocessing.Pool() as pool:
            results = pool.map(load_run_timeseries, task_args_list)
        
        df_list = [df for df in results if df is not None]
        if not df_list: continue

        success_at_eval_points = []
        for df in df_list:
            df = df.sort_values('frame').drop_duplicates('frame')
            interp_success = np.interp(EVALUATION_FRAMES, df['frame'], df['success'], right=df['success'].iloc[-1])
            success_at_eval_points.append(interp_success)
        
        success_array = np.array(success_at_eval_points)
        mean_success_rates = np.mean(success_array, axis=0)
        error_rates = (1.0 - mean_success_rates) * 100
        
        compute_gflops = np.array([
            calculate_training_compute(exp_config['width'], exp_config['n_envs'], frames)
            for frames in EVALUATION_FRAMES
        ])
        
        processed_data_for_plot.append({
            'label': exp_config['label'],
            'compute_gflops': compute_gflops,
            'error_rates': error_rates
        })

    # --- Generate the Plot ---
    if processed_data_for_plot:
        processed_data_for_plot.sort(key=lambda x: x['label'])
        plot_compute_frontiers(processed_data_for_plot, STYLES_MAP, list(MODEL_SIZES.keys()), ENVS, OUTPUT_DIR)
    else:
        print("\nError: No data was processed. Plot not generated.")