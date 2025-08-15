import os
from collections import defaultdict
import glob
import multiprocessing

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm

# --- Constants for Statistical Analysis ---
BOOTSTRAP_ITERATIONS = 2000
CI_PERCENT = 95.0

# --- Constants for Plotting ---
TITLE_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 22
TICK_LABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 16

# ======================================================================================
# CONFIDENCE INTERVAL & DATA LOADING
# ======================================================================================
def get_percentile_ci(bootstrap_stats_distribution, ci_level=CI_PERCENT):
    valid_stats = np.asarray(bootstrap_stats_distribution)
    valid_stats = valid_stats[~np.isnan(valid_stats)]
    if valid_stats.size < 2: return np.nan, np.nan
    alpha = (100.0 - ci_level) / 2.0
    return np.percentile(valid_stats, alpha), np.percentile(valid_stats, 100.0 - alpha)

def bootstrap_confidence_interval(data, metric_func=np.mean, n_iterations=BOOTSTRAP_ITERATIONS, ci_level=CI_PERCENT):
    valid_data = np.asarray(data)[~np.isnan(np.asarray(data))]
    if valid_data.size < 2:
        # For small samples, CI is just the min/max
        return np.mean(valid_data), np.min(valid_data), np.max(valid_data)

    bootstrap_stats = [metric_func(np.random.choice(valid_data, size=len(valid_data), replace=True)) for _ in range(n_iterations)]
    valid_bootstrap_stats = np.array([s for s in bootstrap_stats if not np.isnan(s)])
    if valid_bootstrap_stats.size < 2: return np.nan, np.nan, np.nan
    mean_stat = metric_func(valid_data)
    lower_bound, upper_bound = get_percentile_ci(valid_bootstrap_stats, ci_level)
    return mean_stat, lower_bound, upper_bound

def load_performance_at_frame(args):
    """
    Loads performance from a single run.
    If target_frame is None, it returns the last value.
    If target_frame is set, it interpolates the value at that frame.
    """
    setting, variant, params, run_path, success_tag_key, target_frame, frame_multiplier = args
    try:
        ea = event_accumulator.EventAccumulator(run_path, size_guidance={'scalars': 0})
        ea.Reload()

        scalar_events = ea.Scalars(success_tag_key)
        if not scalar_events:
            return setting, variant, params, np.nan

        # If no target frame is specified, just return the last value
        if target_frame is None:
            return setting, variant, params, scalar_events[-1].value

        # Otherwise, find the performance at the target frame
        frames = np.array([event.step * frame_multiplier for event in scalar_events])
        values = np.array([event.value for event in scalar_events])

        # If the run is too short, np.interp will correctly use the last value
        # when the 'right' argument is set to the last value.
        performance_at_target = np.interp(target_frame, frames, values, right=values[-1])
        return setting, variant, params, performance_at_target

    except Exception as e:
        # print(f"Warning: Could not process {run_path}. Error: {e}")
        return setting, variant, params, np.nan

# ======================================================================================
# PLOTTING
# ======================================================================================
def plot_performance_curves(processed_data, variant_styles, output_dir):
    """
    Generates and saves a plot of performance vs. parameter count.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    settings = ["MT10 rand", "MT50 rand"]
    for i, setting in enumerate(settings):
        ax = axes[i]
        ax.set_title(setting, fontsize=TITLE_FONTSIZE, pad=15)

        all_params_for_setting = set()

        if setting in processed_data:
            sorted_variants = [v for v in variant_styles if v in processed_data[setting]]

            for variant_name in sorted_variants:
                results = processed_data[setting][variant_name]
                style = variant_styles[variant_name]

                sorted_params = sorted(results.keys())
                all_params_for_setting.update(sorted_params)

                means = np.array([results[p]['mean'] for p in sorted_params])
                ci_low = np.array([results[p]['low'] for p in sorted_params])
                ci_high = np.array([results[p]['high'] for p in sorted_params])

                params_in_millions = np.array([p / 1e6 for p in sorted_params])

                # Main plot line
                ax.plot(params_in_millions, means, marker=style['marker'], color=style['color'],
                        label=style['label'], markersize=12, linewidth=3, zorder=3, markeredgecolor='white', markeredgewidth=1.5)

                # Error bars
                y_err = [means - ci_low, ci_high - means]
                ax.errorbar(params_in_millions, means, yerr=y_err, fmt='none', ecolor=style['color'],
                            capsize=5, elinewidth=2, capthick=1.5, zorder=2)

        # --- Axis & Grid Formatting ---
        ax.set_xscale('log', base=10)
        ax.set_xlabel("Total parameter count (M)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE, pad=5)

        if all_params_for_setting:
            tick_values_in_millions = sorted([p / 1e6 for p in all_params_for_setting])
            ax.set_xticks(tick_values_in_millions)
            ax.set_xticks([], minor=True)

        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))

        if i == 0:
            ax.set_ylabel("Success Rate", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)

        ax.grid(True, which='major', axis='both', linestyle='-', color='gainsboro', zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].legend(fontsize=LEGEND_FONTSIZE, loc='best', frameon=True, facecolor='white', framealpha=0.8)

    fig.tight_layout()

    save_path_png = os.path.join(output_dir, "parameter_scaling.png")
    save_path_pdf = os.path.join(output_dir, "parameter_scaling.pdf")
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"\nPlots saved to {output_dir}")

# ======================================================================================
# LATEX TABLE GENERATION
# ======================================================================================
def format_param_count(p_count):
    """Formats parameter count into a human-readable string (e.g., 100K, 4M)."""
    if p_count >= 1_000_000:
        return f"{p_count / 1_000_000:g}M"
    else:
        return f"{p_count / 1_000:g}K"

def generate_and_save_latex_table(processed_data, run_mapping, width_to_params, variant_styles, output_dir):
    """Generates a LaTeX table from the data and saves it to a .tex file."""

    settings_order = list(run_mapping.keys())
    params_order = sorted(width_to_params.values())
    variant_labels = [style['label'] for style in variant_styles.values()]
    variant_keys = list(variant_styles.keys())

    # --- Start building the LaTeX string ---
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Success Rate (SR) with 95\\% bootstrapped confidence intervals. Results are shown as Mean [Lower, Upper].}",
        "\\label{tab:parameter_scaling_results}"
    ]

    for i, setting in enumerate(settings_order):
        # Add vertical spacing between tables
        if i > 0:
            lines.append("\\\\[1.5em]")

        # Define table structure for the current setting
        num_variants = len(variant_labels)
        table_header = [
            "\\begin{tabular}{l" + "c" * num_variants + "}",
            "\\toprule",
            f"& \\multicolumn{{{num_variants}}}{{c}}{{{setting.replace(' rand', '-rand')}}} \\\\",
            f"\\cmidrule(lr){{2-{num_variants + 1}}}",
            "\\textbf{Parameters} & " + " & ".join([f"\\textbf{{{label}}}" for label in variant_labels]) + " \\\\",
            "\\midrule"
        ]
        lines.extend(table_header)

        # Populate rows with data
        for p_count in params_order:
            row_cells = [format_param_count(p_count)]
            for variant_key in variant_keys:
                if setting in processed_data and variant_key in processed_data[setting] and p_count in processed_data[setting][variant_key]:
                    stats = processed_data[setting][variant_key][p_count]
                    # Format as "Mean [Lower, Upper]" and scale by 100
                    mean = stats['mean'] * 100
                    low = stats['low'] * 100
                    high = stats['high'] * 100
                    cell_str = f"{mean:.2f} [{low:.2f}, {high:.2f}]"
                    row_cells.append(cell_str)
                else:
                    row_cells.append("---")
            lines.append(" & ".join(row_cells) + " \\\\")

        # End the current tabular environment
        lines.extend(["\\bottomrule", "\\end{tabular}"])

    lines.append("\\end{table*}")
    latex_string = "\n".join(lines)

    # Save the string to a file
    save_path_tex = os.path.join(output_dir, "parameter_scaling_results.tex")
    with open(save_path_tex, 'w') as f:
        f.write(latex_string)

    print(f"LaTeX table saved to {save_path_tex}")
    print("\n--- Generated LaTeX Code ---")
    print("(Requires LaTeX packages: booktabs, amsmath)")
    print("-" * 28 + "\n" + latex_string)


# ======================================================================================
# MAIN EXECUTION
# ======================================================================================
if __name__ == "__main__":
    # --- 0. CHOOSE EVALUATION METHOD ---
    # Set to None to evaluate at the final frame of each run.
    TARGET_FRAME = None

    # --- General Configuration ---
    LOG_DIR = "runs/"
    OUTPUT_DIR = "scripts/parameter_scaling/"
    SUCCESS_TAG = "Episode/average_task_success_rate"

    # --- 1. CONFIGURE FRAME CALCULATION ---
    # Multiplier to convert TensorBoard steps to environment frames.
    PPO_FRAME_MULTIPLIER = 24576 * 32
    FRAME_MULTIPLIERS = {
        "Vanilla":    PPO_FRAME_MULTIPLIER,
        "Vanilla TE": PPO_FRAME_MULTIPLIER,
        "Simba V2":   PPO_FRAME_MULTIPLIER,
        "SimbaV2 TE": PPO_FRAME_MULTIPLIER,
    }

    # --- 2. CONFIGURE PLOT STYLES ---
    VARIANT_STYLES = {
        "Vanilla":    {"color": "dimgray", "marker": "o", "label": "Vanilla"},
        "Vanilla TE": {"color": "mediumvioletred", "marker": "s", "label": "Vanilla (TE)"},
        "Simba V2":   {"color": "darkcyan", "marker": "p", "label": "Simba V2"},
        "SimbaV2 TE": {"color": "royalblue", "marker": "^", "label": "SimbaV2 (TE)"},
    }

    # --- 3. CONFIGURE PARAMETER MAPPING ---
    WIDTH_TO_PARAMS = {
        0: 100_000,
        1: 1_000_000,
        2: 4_000_000,
        3: 16_000_000,
        4: 64_000_000,
        5: 256_000_000,
    }

    # --- 4. CONFIGURE RUN MAPPING ---
    RUN_MAPPING = {
        "MT10 rand": {
            "Vanilla":    "ppo_vanilla_mt10_rand_scaling",
            "Vanilla TE": "ppo_vanilla_TE_mt10_rand_scaling",
            "Simba V2":   "ppo_simbav2_mt10_rand_scaling",
            "SimbaV2 TE": "ppo_simbav2_TE_mt10_rand_scaling",
        },
        "MT50 rand": {
            "Vanilla":    "ppo_vanilla_mt50_rand_scaling",
            "Vanilla TE": "ppo_vanilla_TE_mt50_rand_scaling",
            "Simba V2":   "ppo_simbav2_mt50_rand_scaling",
            "SimbaV2 TE": "ppo_simbav2_TE_mt50_rand_scaling",
        }
    }

    # --- Find, process, and calculate CIs ---
    tasks = []
    print("Finding event files...")
    for setting, variants in RUN_MAPPING.items():
        for variant_name, base_folder in variants.items():
            multiplier = FRAME_MULTIPLIERS.get(variant_name)
            if multiplier is None:
                print(f"Warning: No frame multiplier found for '{variant_name}'. Skipping.")
                continue

            for width_index, param_count in WIDTH_TO_PARAMS.items():
                search_pattern = os.path.join(LOG_DIR, base_folder, f"*_width_{width_index}_*")
                run_dirs = glob.glob(search_pattern)
                if not run_dirs:
                    continue
                for run_dir in run_dirs:
                    event_files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
                    if event_files:
                        tasks.append((setting, variant_name, param_count, sorted(event_files)[-1], SUCCESS_TAG, TARGET_FRAME, multiplier))

    if not tasks:
        print("\nError: No event files were found.")
    else:
        results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        print(f"\nLoading data from {len(tasks)} runs...")
        with multiprocessing.Pool() as pool:
            pbar = tqdm(pool.imap_unordered(load_performance_at_frame, tasks), total=len(tasks))
            for setting, variant, params, final_value in pbar:
                if not np.isnan(final_value): results[setting][variant][params].append(final_value)

        print("\n" + "="*80)
        print("--- Data Loading Summary ---")
        if TARGET_FRAME:
            print(f"Evaluation performed at frame: {TARGET_FRAME:,}")
        else:
            print("Evaluation performed at final frame of each run.")
        print("="*80)

        for setting in sorted(results.keys()):
            for variant in sorted(results[setting].keys()):
                print(f"\n[{setting}] - [{variant}]")
                for params in sorted(results[setting][variant].keys()):
                    values = results[setting][variant][params]
                    mean_val = np.mean(values)
                    param_str = format_param_count(params)
                    print(f"  - {param_str} ({len(values)} runs): Mean={mean_val:.3f}, Values={np.round(values, 3)}")
        print("="*80 + "\n")

        processed_data = defaultdict(lambda: defaultdict(dict))
        print("Calculating confidence intervals...")
        for setting, variants in results.items():
            for variant, param_configs in variants.items():
                for params, values in param_configs.items():
                    if values:
                        mean, ci_low, ci_high = bootstrap_confidence_interval(values)
                        processed_data[setting][variant][params] = {'mean': mean, 'low': ci_low, 'high': ci_high}

        if processed_data:
            # Generate the plot
            plot_performance_curves(processed_data, VARIANT_STYLES, OUTPUT_DIR)

            # Generate the LaTeX table
            generate_and_save_latex_table(processed_data, RUN_MAPPING, WIDTH_TO_PARAMS, VARIANT_STYLES, OUTPUT_DIR)
        else:
            print("\nError: No valid data was processed.")