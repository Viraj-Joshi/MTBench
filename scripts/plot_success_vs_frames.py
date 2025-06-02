# pip install pandas numpy matplotlib tensorboard PyYAML scipy tqdm
import os
from collections import defaultdict
import re
import math # For ceiling function for grid dimensions
import yaml # For reading config files
import warnings # Added for LaTeX table helper
import traceback # For detailed error printing

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# from matplotlib.cm import get_cmap # Not used with fixed colors
from tensorboard.backend.event_processing import event_accumulator
import argparse
from scipy.stats import scoreatpercentile # For CI functions if IQM was used
from tqdm import tqdm # For progress bars, especially during bootstrapping

# --- Constants for Statistical Analysis ---
BOOTSTRAP_ITERATIONS = 2000 # Number of bootstrap samples for CI calculation
CI_PERCENT = 95.0 # Confidence Interval percentage

# --- Helper Function for Bootstrapped Confidence Interval (from a distribution of stats) ---
def get_percentile_ci(bootstrap_stats_distribution, ci_level=CI_PERCENT):
    """Calculates CI from a pre-computed distribution of bootstrap statistics."""
    bootstrap_stats_distribution = np.asarray(bootstrap_stats_distribution)
    valid_stats = bootstrap_stats_distribution[~np.isnan(bootstrap_stats_distribution)]
    if valid_stats.size < 2:
        return np.nan, np.nan

    alpha = (100.0 - ci_level) / 2.0
    lower_bound = np.percentile(valid_stats, alpha)
    upper_bound = np.percentile(valid_stats, 100.0 - alpha)
    return lower_bound, upper_bound

# --- Helper Function for Bootstrapped Confidence Interval (generic, on a sample of data points) ---
def bootstrap_confidence_interval(data, metric_func, n_iterations=BOOTSTRAP_ITERATIONS, ci_level=CI_PERCENT, desc="Bootstrapping CI", disable_tqdm=True):
    """Calculates the bootstrapped confidence interval for a given metric on data."""
    data = np.asarray(data)
    valid_data = data[~np.isnan(data)]
    if valid_data.size < 2:
        if valid_data.size == 1:
            return np.nan, np.nan
        return np.nan, np.nan


    bootstrap_stats = []
    for _ in range(n_iterations):
        sample = np.random.choice(valid_data, size=len(valid_data), replace=True)
        if sample.size == 0:
            bootstrap_stats.append(np.nan)
            continue
        stat = metric_func(sample)
        bootstrap_stats.append(stat)

    valid_bootstrap_stats = [s for s in bootstrap_stats if not np.isnan(s)]
    if not valid_bootstrap_stats or len(valid_bootstrap_stats) < 2:
        return np.nan, np.nan

    return get_percentile_ci(np.array(valid_bootstrap_stats), ci_level)


# --- Helper Function to Generate Comparison String ---
def get_comparison_string(setup_keys):
    if not setup_keys: return "comparison"
    if 'SETUP_INFO' not in globals() or not isinstance(SETUP_INFO, dict):
        return "_vs_".join(sorted([str(k) for k in setup_keys]))

    names = [SETUP_INFO[key]['name'] for key in setup_keys if key in SETUP_INFO and 'name' in SETUP_INFO[key]]
    if not names: return "_vs_".join(sorted([str(k) for k in setup_keys]))
    names.sort()
    sanitized_names = [re.sub(r'\W+', '', name) for name in names]
    return "_vs_".join(sanitized_names)

# --- Helper Function to Extract Setup Info from Filename ---
def extract_setup_info(run_name):
    if 'SETUP_INFO' not in globals() or not isinstance(SETUP_INFO, dict):
        return None, None, None
    for setup_type, info in SETUP_INFO.items(): # Iterates over the current global SETUP_INFO
        match = re.search(info['pattern'], run_name)
        if match:
            num_envs = int(match.group(1))
            seed_number = int(match.group(2))
            return setup_type, num_envs, seed_number
    return None, None, None

# --- Helper Function to Adjust Success Data ---
def adjust_data(df, value_col_name='success'):
    if df is None or df.empty:
        return pd.DataFrame({value_col_name: [0.0], 'frame': [0]})
    df_copy = df.copy()
    if 0 not in df_copy['frame'].values:
        new_row_data = {col: [np.nan] for col in df_copy.columns}
        new_row_data[value_col_name] = [0.0]
        new_row_data['frame'] = [0]
        new_row = pd.DataFrame(new_row_data, columns=df_copy.columns if 'frame' in df_copy.columns else [value_col_name, 'frame'])
        result_df = pd.concat([new_row, df_copy], ignore_index=True)

    elif df_copy.loc[df_copy['frame'] == 0, value_col_name].iloc[0] != 0.0:
        result_df = df_copy
        result_df.loc[result_df['frame'] == 0, value_col_name] = 0.0
    else:
        result_df = df_copy
    if value_col_name in result_df.columns:
      result_df = result_df[result_df[value_col_name] >= 0]
      result_df.dropna(subset=[value_col_name, 'frame'], inplace=True)
    else:
      result_df.dropna(subset=['frame'], inplace=True)


    result_df = result_df.sort_values('frame').reset_index(drop=True)
    result_df.drop_duplicates(subset=['frame'], keep='last', inplace=True)
    return result_df

# --- Helper Function to Adjust Generic Scalar Data (like Reward) ---
def adjust_generic_scalar_data(df, value_col_name='value'):
    if df is None or df.empty:
        return pd.DataFrame(columns=[value_col_name, 'frame'])
    df_copy = df.copy()
    if 'frame' not in df_copy.columns or value_col_name not in df_copy.columns:
        return pd.DataFrame(columns=[value_col_name, 'frame'])

    df_copy.dropna(subset=['frame', value_col_name], inplace=True)
    if df_copy.empty:
        return pd.DataFrame(columns=[value_col_name, 'frame'])

    df_copy = df_copy.sort_values('frame').reset_index(drop=True)
    df_copy.drop_duplicates(subset=['frame'], keep='last', inplace=True)
    return df_copy


# --- Core plotting logic for a single axis/set of methods ---
def _plot_curves_on_ax(ax, overall_results_dict, setup_run_counts, 
                       current_setup_info, methods_to_plot_keys, horizon_val):
    LINE_WIDTH = 3.0 # Slightly reduced for grid
    # SMALL_FONT_SIZE = 14 # Adjusted for legend if needed
    # MEDIUM_FONT_SIZE = 18 # Adjusted for labels if needed

    all_max_frames_subplot = 0
    
    # Filter and sort methods_to_plot_keys based on their order in current_setup_info
    setup_info_keys_ordered = list(current_setup_info.keys())
    sorted_methods_to_plot = sorted(
        [m_key for m_key in methods_to_plot_keys if m_key in current_setup_info and m_key in overall_results_dict],
        key=lambda x: setup_info_keys_ordered.index(x) if x in setup_info_keys_ordered else float('inf')
    )

    for setup_type in sorted_methods_to_plot: # Use tqdm here if this part is slow overall
        if setup_type not in overall_results_dict: # Should be caught by filter above
            continue
            
        overall_run_data = overall_results_dict[setup_type].get('overall_avg', [])
        if not overall_run_data:
            continue

        setup_config = current_setup_info[setup_type]
        required_envs = setup_config.get('required_envs')

        env_k_str = "N/A"
        if required_envs is not None:
            try: env_k_str = f"{int(required_envs) // 1024}k"
            except (ValueError, TypeError): pass

        run_data_adjusted = [adjust_data(data.copy(), value_col_name='success') for data in overall_run_data]
        run_data_adjusted = [df for df in run_data_adjusted if not df.empty and len(df) > 1]

        if not run_data_adjusted: continue
        
        try:
            max_frames_per_seed = [df['frame'].max() for df in run_data_adjusted]
            if not max_frames_per_seed: continue
            setup_max_frame = max(max_frames_per_seed)
            all_max_frames_subplot = max(all_max_frames_subplot, setup_max_frame)

            all_frames_this_setup = set([0])
            for df_ in run_data_adjusted: all_frames_this_setup.update(df_['frame'].unique())
            ref_frames_setup = np.unique(sorted(list(all_frames_this_setup)))
            
            if ref_frames_setup.size == 0 or (ref_frames_setup.size > 0 and ref_frames_setup[-1] < setup_max_frame):
                 ref_frames_setup = np.unique(np.append(ref_frames_setup, setup_max_frame))
            if ref_frames_setup.size < 1: continue

            interpolated_successes = []
            for df in run_data_adjusted:
                unique_frames, unique_indices = np.unique(df['frame'].values, return_index=True)
                unique_values = df['success'].values[unique_indices]
                if len(unique_frames) < 1: continue
                left_fill = unique_values[0] if len(unique_values) > 0 else 0.0
                right_fill = unique_values[-1] if len(unique_values) > 0 else 0.0
                interp_val = np.interp(ref_frames_setup, unique_frames, unique_values, left=left_fill, right=right_fill)
                interpolated_successes.append(interp_val)

            if not interpolated_successes: continue

            interpolated_successes_array = np.array(interpolated_successes)
            mean_curve = np.mean(interpolated_successes_array, axis=0)
            lower_ci_curve = np.full_like(mean_curve, np.nan)
            upper_ci_curve = np.full_like(mean_curve, np.nan)

            for i in range(interpolated_successes_array.shape[1]):
                data_at_frame = interpolated_successes_array[:, i]
                if len(data_at_frame[~np.isnan(data_at_frame)]) > 1:
                    low_ci, high_ci = bootstrap_confidence_interval(data_at_frame, np.mean, disable_tqdm=True)
                    lower_ci_curve[i] = low_ci
                    upper_ci_curve[i] = high_ci
                else:
                    lower_ci_curve[i], upper_ci_curve[i] = np.nan, np.nan
            
            label = f"{setup_config['name']}" # Simpler label for grid
            color = setup_config['color']
            ax.plot(ref_frames_setup, mean_curve, color=color, linewidth=LINE_WIDTH, label=label)
            valid_ci_mask = ~np.isnan(lower_ci_curve) & ~np.isnan(upper_ci_curve)
            if np.any(valid_ci_mask):
                ax.fill_between(ref_frames_setup[valid_ci_mask],
                                np.maximum(0, lower_ci_curve[valid_ci_mask]),
                                np.minimum(1, upper_ci_curve[valid_ci_mask]),
                                alpha=0.2, color=color)
        except Exception as e:
            tqdm.write(f"  Subplot Error for setup '{setup_type}': {e}") # Use tqdm.write if in a tqdm loop

    # Styling for the current ax
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f2f2f2')
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(axis='both', labelsize=12) # Adjusted for grid
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    if ax.xaxis.offsetText.get_text(): ax.xaxis.offsetText.set_fontsize(12) # Adjusted
    
    ax.set_ylim(0, 1.05)
    if all_max_frames_subplot > 0:
        ax.set_xlim(0, all_max_frames_subplot * 1.05)
    else:
        ax.set_xlim(0, 1)
    
    legend = ax.legend(fontsize=10) # Adjusted for grid
    if legend:
      for text in legend.get_texts():
          text.set_alpha(0.85) # Make legend text slightly transparent


# --- Plotting Function for Single Overall Success Plot ---
def plot_overall_success_single(overall_results_dict, setup_run_counts, current_setting_name_arg, horizon_arg):
    # This is the original plot_overall_success, renamed and slightly adapted
    # It uses the global HORIZON for its label, ensure it's set or passed
    print(f"Generating single overall success rate plot for {current_setting_name_arg}...")
    fig, ax = plt.subplots(figsize=(12, 7)) # Original size

    # Call the core plotting logic, but for all methods found in overall_results_dict
    # It needs the global SETUP_INFO for the current_setting_name_arg
    
    # Ensure global SETUP_INFO is for current_setting_name_arg if not passed directly
    # (This is handled by main logic setting global SETUP_INFO before calling)
    methods_to_plot = list(overall_results_dict.keys())

    _plot_curves_on_ax(ax, overall_results_dict, setup_run_counts, 
                       SETUP_INFO, methods_to_plot, horizon_arg) # Pass global SETUP_INFO

    # Original styling for single plot labels
    ax.set_xlabel(f"Frames", fontsize=MEDIUM_FONT_SIZE) # Original font size
    ax.set_ylabel("Success Rate", fontsize=MEDIUM_FONT_SIZE) # Original font size
    # Legend is handled by _plot_curves_on_ax, might need adjustment if styles differ drastically

    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    save_dir = "scripts/figures"; os.makedirs(save_dir, exist_ok=True)
    save_path_base = os.path.join(save_dir, f"overall_success_{current_setting_name_arg}")
    try:
        plt.savefig(f"{save_path_base}.pdf"); plt.savefig(f"{save_path_base}.png", dpi=300)
        print(f"Overall success plot for {current_setting_name_arg} saved to {save_path_base}.png/.pdf")
    except Exception as e: print(f"ERROR saving single overall plot: {e}")
    plt.close(fig)


# --- Method Categories for Grid Plot ---
GRAD_MANI_METHODS_KEYS = ['pcgrad', 'cagrad', 'famo']
NEURAL_ARCH_METHODS_KEYS = [
    'soft_modularization', 'shppo_care', 'mhppo_care',
    'shppo_paco', 'shppo_moore', 'mhppo_moore'
]
ALL_SETUP_KEYS_ORDERED = [ # Define a canonical order if needed for legends, etc.
    'shppo_vanilla', 'mhppo_vanilla', 'pcgrad', 'cagrad', 'famo',
    'soft_modularization', 'shppo_care', 'mhppo_care', 'shppo_paco',
    'shppo_moore', 'mhppo_moore'
]


# --- New 2x2 Grid Plotting Function ---
def plot_success_grid_2x2(collected_data, horizon_val):
    print("Generating 2x2 grid plot for success vs frames...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex='col', sharey='row')
    
    MEDIUM_FONT_SIZE_GRID = 16 # For axis labels in grid
    ROW_LABEL_FONT_SIZE = 14

    method_categories = {
        0: ("Gradient Manipulation", GRAD_MANI_METHODS_KEYS),
        1: ("Neural Architectures", NEURAL_ARCH_METHODS_KEYS)
    }
    settings_for_cols = {
        0: "mt10",
        1: "mt50"
    }

    for i in range(2): # Rows (method categories)
        for j in range(2): # Columns (settings: MT10, MT50)
            ax = axes[i, j]
            setting_name = settings_for_cols[j]
            category_name, method_keys = method_categories[i]

            if setting_name in collected_data:
                data_for_setting = collected_data[setting_name]
                _plot_curves_on_ax(ax, 
                                   data_for_setting['results'], 
                                   data_for_setting['counts'],
                                   data_for_setting['setup_info'], # Use the specific SETUP_INFO for this setting
                                   method_keys,
                                   horizon_val)
            else:
                ax.text(0.5, 0.5, f"No data for\n{setting_name}\n{category_name}", 
                        ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_facecolor('#e0e0e0') # Light grey for empty

    # Apply labels and titles
    axes[0,0].set_ylabel("Success Rate", fontsize=MEDIUM_FONT_SIZE_GRID)
    axes[1,0].set_ylabel("Success Rate", fontsize=MEDIUM_FONT_SIZE_GRID)

    axes[1,0].set_xlabel(r"$\bf{MT10}$", fontsize=MEDIUM_FONT_SIZE_GRID)
    axes[1,1].set_xlabel(r"$\bf{MT50}$", fontsize=MEDIUM_FONT_SIZE_GRID)
    
    fig.supxlabel("Frames", fontsize=MEDIUM_FONT_SIZE_GRID + 2, y=0.02) # Adjusted y for visibility

    # Row titles (text on the left of the first column)
    # Adjust x coordinate as needed; -0.1 might be too close with tight_layout
    # Using fig.text for more control over positioning relative to the figure
    fig.text(0.06, 0.70, 'Gradient\nManipulation', ha='center', va='center', rotation='vertical', fontsize=ROW_LABEL_FONT_SIZE)
    fig.text(0.06, 0.30, 'Neural\nArchitectures', ha='center', va='center', rotation='vertical', fontsize=ROW_LABEL_FONT_SIZE)


    plt.tight_layout(rect=[0.08, 0.05, 0.98, 0.95]) # Adjust rect to make space for supxlabel and row titles
    
    save_dir = "scripts/figures"; os.makedirs(save_dir, exist_ok=True)
    save_path_base = os.path.join(save_dir, "success_vs_frames") # Fixed filename

    try:
        plt.savefig(f"{save_path_base}.pdf"); plt.savefig(f"{save_path_base}.png", dpi=300)
        print(f"2x2 Grid plot saved to {save_path_base}.png/.pdf")
    except Exception as e: print(f"ERROR saving 2x2 grid plot: {e}")
    plt.close(fig)


# --- Dynamic Pattern and SETUP_INFO generation ---
def get_patterns_for_setting(setting_str):
    patterns = {}
    if setting_str == "mt10":
        patterns['PATTERN_SHPPO_VANILLA'] = rf"05_09_ppo_vanilla_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_MHPPO_VANILLA'] = rf"mhppo_vanilla_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_FAMO'] = rf"05_09_ppo_famo_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_PCGRAD'] = rf"05_11_ppo_pcgrad_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_CAGRAD'] = rf"ppo_cagrad_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_SHPPO_PACO'] = rf"shppo_paco_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_MHPPO_MOORE'] = rf"mhppo_moore_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_SHPPO_MOORE'] = rf"shppo_moore_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_SOFT_MODULARIZATION'] = rf"ppo_soft_modularization_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_MHPPO_CARE'] = rf"05_11_mhppo_care_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_SHPPO_CARE'] = rf"shppo_care_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
    else: # Default MT50 patterns
        patterns['PATTERN_SHPPO_VANILLA'] = rf"^ppo_vanilla_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_MHPPO_VANILLA'] = rf"05_07_mhppo_vanilla_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_FAMO'] = rf"ppo_famo_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_PCGRAD'] = rf"ppo_pcgrad_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_CAGRAD'] = rf"ppo_cagrad_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_SHPPO_PACO'] = rf"shppo_paco_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_MHPPO_MOORE'] = rf"mhppo_moore_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_SHPPO_MOORE'] = rf"shppo_moore_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_MHPPO_CARE'] = rf"mhppo_care_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_SHPPO_CARE'] = rf"shppo_care_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
        patterns['PATTERN_SOFT_MODULARIZATION'] = rf"ppo_soft_modularization_{setting_str}_rand_envs_(\d+)_seed_(\d+).*"
    return patterns

def get_setup_info_for_setting(setting_str, current_patterns):
    required_envs_val = 24576 
    if setting_str == "mt10":
        pc_grad_required_envs = 24576
        cagrad_required_envs = 24576
    else: # mt50
        pc_grad_required_envs = 8192
        cagrad_required_envs = 6144

    return {
        'shppo_vanilla': {'pattern': current_patterns['PATTERN_SHPPO_VANILLA'], 'color': '#1A85FF', 'name': 'SH-Vanilla', 'required_envs': required_envs_val},
        'mhppo_vanilla': {'pattern': current_patterns['PATTERN_MHPPO_VANILLA'], 'color': '#2ca02c', 'name': 'MH-Vanilla', 'required_envs': required_envs_val},
        'pcgrad': {'pattern': current_patterns['PATTERN_PCGRAD'], 'color': '#ff7f0e', 'name': 'PCGrad', 'required_envs': pc_grad_required_envs},
        'cagrad': {'pattern': current_patterns['PATTERN_CAGRAD'], 'color': '#1f77b4', 'name': 'CAGrad', 'required_envs': cagrad_required_envs},
        'famo': {'pattern': current_patterns['PATTERN_FAMO'], 'color': '#3690ff', 'name': 'FAMO', 'required_envs': required_envs_val},
        'soft_modularization': {'pattern': current_patterns['PATTERN_SOFT_MODULARIZATION'], 'color': '#FF4081', 'name': 'Soft-Modularization', 'required_envs': required_envs_val},
        'shppo_care': {'pattern': current_patterns['PATTERN_SHPPO_CARE'], 'color': '#FF5722', 'name': 'SH-CARE', 'required_envs': required_envs_val},
        'mhppo_care': {'pattern': current_patterns['PATTERN_MHPPO_CARE'], 'color': '#FF9800', 'name': 'MH-CARE', 'required_envs': required_envs_val},
        'shppo_paco': {'pattern': current_patterns['PATTERN_SHPPO_PACO'], 'color': '#E91E63', 'name': 'SH-PaCo', 'required_envs': required_envs_val},
        'shppo_moore': {'pattern': current_patterns['PATTERN_SHPPO_MOORE'], 'color': '#C2185B', 'name': 'SH-MOORE', 'required_envs': required_envs_val},
        'mhppo_moore': {'pattern': current_patterns['PATTERN_MHPPO_MOORE'], 'color': '#F50057', 'name': 'MH-MOORE', 'required_envs': required_envs_val},
    }

# --- Global variables that might be set per setting iteration ---
SETUP_INFO = {} 
# NUM_TASKS = 0 # Set based on setting
# HORIZON = 0 # Set from args

# --- Main Execution Logic ---
if __name__ == "__main__":
    LOG_DIR_DEFAULT = "./runs/"
    if not os.path.exists(LOG_DIR_DEFAULT) and os.path.exists("/work/08962/vjoshi12/ls6/IsaacGymEnvs/runs/"): # Example HPC path
        LOG_DIR_DEFAULT = "/work/08962/vjoshi12/ls6/IsaacGymEnvs/runs/"
        print(f"Default ./runs/ not found, using HPC path: {LOG_DIR_DEFAULT}")
    elif not os.path.exists(LOG_DIR_DEFAULT):
        print(f"Warning: Default log directory '{LOG_DIR_DEFAULT}' not found. Please ensure runs are present or specify --log_dir.")
        os.makedirs(LOG_DIR_DEFAULT, exist_ok=True)
        print(f"Created directory: {LOG_DIR_DEFAULT}")

    parser = argparse.ArgumentParser(description="Plot success vs frames from TensorBoard logs with Bootstrapped CI.")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR_DEFAULT, help="Root directory containing TensorBoard run folders.")
    parser.add_argument("--is_mt10", action="store_true", help="MT10 specific setup (for single plot mode).")
    parser.add_argument("--horizon", type=int, default=32, help="Horizon value for frame calculation.")
    parser.add_argument("--grid_plot", action="store_true", help="Generate a 2x2 grid plot for MT10/MT50 success.")


    args = parser.parse_args()

    LOG_DIR = args.log_dir
    HORIZON = args.horizon # Global HORIZON

    print(f"Arguments: {args}")

    if args.grid_plot:
        collected_data_for_grid = {}
        settings_for_grid = ["mt10", "mt50"]
        
        for current_setting_iterator in settings_for_grid:
            print(f"\n--- Processing data for GRID PLOT: {current_setting_iterator.upper()} ---")
            # Explicitly set globals for this iteration
            setting_name_global = current_setting_iterator # Used by extract_setup_info via global SETUP_INFO
            num_tasks_global = 10 if setting_name_global == "mt10" else 50
            
            # Generate and set global SETUP_INFO for the current setting iteration
            current_patterns_iter = get_patterns_for_setting(setting_name_global)
            SETUP_INFO = get_setup_info_for_setting(setting_name_global, current_patterns_iter) # Set global SETUP_INFO
            
            if not SETUP_INFO:
                print(f"FATAL: SETUP_INFO empty for {setting_name_global} in grid processing. Skipping.")
                continue

            local_results_data = defaultdict(lambda: defaultdict(list))
            local_setup_run_counts = defaultdict(int)
            processed_runs_iter = 0
            skipped_runs_iter = 0

            dir_items_iter = os.listdir(LOG_DIR)
            for item_name_iter in tqdm(dir_items_iter, desc=f"Scanning for {setting_name_global}"):
                item_path_iter = os.path.join(LOG_DIR, item_name_iter)
                if not os.path.isdir(item_path_iter): continue

                # extract_setup_info will use the global SETUP_INFO set for this iteration
                setup_type, num_envs, seed_number = extract_setup_info(item_name_iter) 
                if setup_type is None: continue

                setup_config = SETUP_INFO[setup_type]
                required_env_count = setup_config.get('required_envs')
                if required_env_count is None or num_envs != required_env_count:
                    skipped_runs_iter +=1
                    continue
                
                event_file_path = None
                for p_dir in [os.path.join(item_path_iter, "summaries"), item_path_iter]:
                    if os.path.isdir(p_dir):
                        try:
                            event_files = [f for f in os.listdir(p_dir) if f.startswith("events.out.tfevents")]
                            if event_files: event_file_path = os.path.join(p_dir, sorted(event_files)[-1]); break
                        except Exception: pass
                if event_file_path is None: skipped_runs_iter +=1; continue
                
                try:
                    ea = event_accumulator.EventAccumulator(event_file_path, size_guidance={'scalars': 0}, purge_orphaned_data=True)
                    ea.Reload(); all_scalar_tags = ea.Tags().get("scalars", [])
                    if not all_scalar_tags: skipped_runs_iter +=1; continue
                    
                    overall_success_tag = None # Logic to find tag
                    literal_tag_atsr = "Episode/average_task_success_rate"
                    if literal_tag_atsr in all_scalar_tags: overall_success_tag = literal_tag_atsr
                    # ... (add other fallback logic for tag if needed) ...

                    if overall_success_tag:
                        values = ea.Scalars(overall_success_tag)
                        if values and hasattr(values[0], 'step'):
                            df_overall = pd.DataFrame({'success': [x.value for x in values], 'frame': [x.step * num_envs * HORIZON for x in values]})
                            if not df_overall.empty:
                                local_results_data[setup_type]['overall_avg'].append(df_overall.copy())
                                local_setup_run_counts[setup_type] += 1
                                processed_runs_iter += 1
                            else: skipped_runs_iter +=1
                        else: skipped_runs_iter +=1
                    else: skipped_runs_iter +=1
                except Exception: skipped_runs_iter +=1
            
            print(f"Finished processing for {setting_name_global}: {processed_runs_iter} runs added, {skipped_runs_iter} skipped.")
            if processed_runs_iter > 0:
                collected_data_for_grid[setting_name_global] = {
                    'results': local_results_data,
                    'counts': local_setup_run_counts,
                    'setup_info': SETUP_INFO.copy() # Store the specific SETUP_INFO for this setting
                }
        
        if "mt10" in collected_data_for_grid and "mt50" in collected_data_for_grid:
            plot_success_grid_2x2(collected_data_for_grid, HORIZON)
        else:
            print("Could not generate grid plot: Data for both MT10 and MT50 was not successfully processed.")

    else: # Original behavior: single plot
        if args.is_mt10:
            setting = "mt10"
            NUM_TASKS = 10
        else:
            setting = "mt50"
            NUM_TASKS = 50
        print(f"Using single plot mode for setting: {setting}")

        current_patterns = get_patterns_for_setting(setting)
        SETUP_INFO = get_setup_info_for_setting(setting, current_patterns) # Set global SETUP_INFO

        if not SETUP_INFO:
            print(f"FATAL: SETUP_INFO is empty for single plot mode setting '{setting}'. Exiting.")
            exit()

        results_data_single = defaultdict(lambda: defaultdict(list))
        setup_run_counts_single = defaultdict(int)
        processed_runs_total_single = 0; skipped_runs_count_single = 0

        dir_items_single = os.listdir(LOG_DIR)
        for item_name_single in tqdm(dir_items_single, desc=f"Scanning for single plot {setting}"):
            item_path_single = os.path.join(LOG_DIR, item_name_single)
            if not os.path.isdir(item_path_single): continue
            
            setup_type, num_envs, seed_number = extract_setup_info(item_name_single)
            if setup_type is None: continue

            setup_config = SETUP_INFO[setup_type]
            required_env_count = setup_config.get('required_envs')
            if required_env_count is None or num_envs != required_env_count:
                skipped_runs_count_single += 1
                continue
            
            event_file_path = None # Standard event file finding
            for p_dir in [os.path.join(item_path_single, "summaries"), item_path_single]:
                if os.path.isdir(p_dir):
                    try:
                        event_files = [f for f in os.listdir(p_dir) if f.startswith("events.out.tfevents")]
                        if event_files: event_file_path = os.path.join(p_dir, sorted(event_files)[-1]); break
                    except Exception: pass
            if event_file_path is None: skipped_runs_count_single +=1; continue

            try: # Data extraction
                ea = event_accumulator.EventAccumulator(event_file_path, size_guidance={'scalars': 0}, purge_orphaned_data=True)
                ea.Reload(); all_scalar_tags = ea.Tags().get("scalars", [])
                if not all_scalar_tags: skipped_runs_count_single +=1; continue
                
                overall_success_tag = None # Find tag
                literal_tag_atsr = "Episode/average_task_success_rate"
                if literal_tag_atsr in all_scalar_tags: overall_success_tag = literal_tag_atsr
                # ... (add other fallback logic for tag if needed) ...

                if overall_success_tag:
                    values = ea.Scalars(overall_success_tag)
                    if values and hasattr(values[0], 'step'):
                        df_overall = pd.DataFrame({'success': [x.value for x in values], 'frame': [x.step * num_envs * HORIZON for x in values]})
                        if not df_overall.empty:
                            results_data_single[setup_type]['overall_avg'].append(df_overall.copy())
                            setup_run_counts_single[setup_type] += 1
                            processed_runs_total_single += 1
                        else: skipped_runs_count_single +=1
                    else: skipped_runs_count_single +=1
                else: skipped_runs_count_single +=1
            except Exception: skipped_runs_count_single +=1

        print("\n--- Single Plot Processing Summary ---")
        if not results_data_single or processed_runs_total_single == 0 :
            print("No valid data loaded for single plot. Exiting.")
        else:
            print(f"Successfully processed data from {processed_runs_total_single} runs for {setting}.")
            if skipped_runs_count_single > 0: print(f"Skipped {skipped_runs_count_single} directories/runs.")
            plot_overall_success_single(results_data_single, setup_run_counts_single, setting, HORIZON)
            print("\nSingle plot generation complete!")