import os
from collections import defaultdict
import yaml

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import combinations


def create_hyperparameter_analysis(df, metric='average_task_success_rate', output_dir="./debug"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots for each hyperparameter pair
    hyper_params = ['nstep', 'gradient_steps_per_itr', 'batch_size', 'critic_tau', 'replay_buffer_size']
    param_pairs = list(combinations(hyper_params, 2))
    n_pairs = len(param_pairs)
    
    # Calculate grid dimensions
    n_rows = int(np.ceil(np.sqrt(n_pairs)))
    n_cols = int(np.ceil(n_pairs / n_rows))
    
    # Create heatmap matrix figure
    heatmap_fig = plt.figure(figsize=(5*n_cols, 4*n_rows))
    heatmap_fig.suptitle('Hyperparameter Impact on Success Rate', fontsize=16, y=1.02)
    
    for idx, (param1, param2) in enumerate(param_pairs, 1):
        plt.subplot(n_rows, n_cols, idx)
        
        # Create pivot table for heatmap
        pivot_data = df.groupby([param1, param2])[metric].mean().reset_index()
        pivot_table = pivot_data.pivot(index=param1, columns=param2, values=metric)
        
        # Create heatmap
        sns.heatmap(pivot_table, 
                   cmap='viridis',
                   annot=True, 
                   fmt='.2f',
                   cbar_kws={'label': metric})
        
        plt.title(f'{param1} vs {param2}')
        plt.tight_layout()
    
    # Save heatmap matrix
    heatmap_path = os.path.join(output_dir, 'hyperparameter_heatmaps.png')
    heatmap_fig.savefig(heatmap_path, bbox_inches='tight', dpi=300)
    plt.close(heatmap_fig)
    
    # Create parameter importance figure
    importance_fig = plt.figure(figsize=(10, 6))
    param_importance = {}
    
    for param in hyper_params:
        correlation = df.groupby(param)[metric].mean()
        param_importance[param] = correlation.std()
    
    # Plot parameter importance
    importance_series = pd.Series(param_importance)
    importance_series.sort_values().plot(kind='barh')
    plt.title('Hyperparameter Importance\n(Higher variance indicates more impact)')
    plt.xlabel('Standard Deviation of Success Rate')
    plt.tight_layout()
    
    # Save parameter importance plot
    importance_path = os.path.join(output_dir, 'parameter_importance.png')
    importance_fig.savefig(importance_path, bbox_inches='tight', dpi=300)
    plt.close(importance_fig)
    
    # Get top 3 configurations
    top_3_idx = df[metric].nlargest(3).index
    
    print("\nTop 3 Configurations:")
    print("-" * 50)
    for rank, idx in enumerate(top_3_idx, 1):
        config = {param: df.loc[idx, param] for param in hyper_params}
        performance = df.loc[idx, metric]
        print(f"\nRank {rank} (Success Rate: {performance:.3f}):")
        for param, value in config.items():
            print(f"{param}: {value}")

    print(f"\nFigures saved to:")
    print(f"- {heatmap_path}")
    print(f"- {importance_path}")

if __name__ == "__main__":
    log_dir = "runs/"
    runname_to_exps = {
        "MTSAC": [f for f in os.listdir(log_dir) if f>"mt_sac_14-12" and "mt_sac" in f],
    }
    # Create DataFrame combining config and performance data
    runs_data = []
    for i in range(0, len(runname_to_exps["MTSAC"]), 2):
        config_path = os.path.join(log_dir, runname_to_exps["MTSAC"][i], "config.yaml")
        config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
        
        # Extract hyperparameters
        hyperparams = {
            'nstep': config['nstep'],
            'gradient_steps_per_itr': config['gradient_steps_per_itr'],
            'batch_size': config['batch_size'],
            'critic_tau': config['critic_tau'],
            'replay_buffer_size': config['replay_buffer_size']
        }
        
        # Get performance metric
        summaries_path = os.path.join(log_dir, runname_to_exps["MTSAC"][i+1])
        ef = [f for f in os.listdir(summaries_path) if f.startswith("events.out.tfevents")][0]
        ea = event_accumulator.EventAccumulator(os.path.join(summaries_path, ef))
        ea.Reload()

        tags = ea.Tags()["scalars"]
        data = {k: ea.Scalars(k) for k in tags if k.startswith("Episode")}
        df = pd.DataFrame({k: [x.value for x in v] for k, v in data.items()})
        
        # Assuming df contains the performance metrics
        success_rate = np.mean(df['Episode/average_task_success_rate'].iloc[-3:])
        
        runs_data.append({**hyperparams, 'average_task_success_rate': success_rate})

        print(f"Processing MTSAC run: {runname_to_exps['MTSAC'][i+1]}")

    # Convert to DataFrame
    runs_df = pd.DataFrame(runs_data)

    # Create visualizations
    create_hyperparameter_analysis(runs_df, metric='average_task_success_rate')
    plt.show()