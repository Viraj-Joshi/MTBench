import wandb
import os

# --- Configuration ---
# Replace with your W&B entity (username or team name)
# ENTITY = "your_entity_name" 
# Replace with your W&B project name
PROJECT = "IsaacGym"
# The prefix of the run names you want to download
RUN_NAME_PREFIX = "ppo_famo_mt50_rand_envs_24576_seed_44"
# Directory to save the downloaded run files
DOWNLOAD_DIR = "runs/ppo_vanilla_mt50_rand_scaling/32768"

# --- Initialize W&B API ---
api = wandb.Api()

# --- Create download directory if it doesn't exist ---
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# --- Fetch runs ---
print(f"Fetching runs from project '{PROJECT}'...")
runs = api.runs(f"{PROJECT}")

# --- Filter and download runs ---
found_runs = []
for run in runs:
    if run.name and run.name.startswith(RUN_NAME_PREFIX):
        found_runs.append(run)

if not found_runs:
    print(f"No runs found with names starting with '{RUN_NAME_PREFIX}'.")
else:
    print(f"Found {len(found_runs)} runs matching the prefix. Downloading files...")
    for run in found_runs:
        run_download_path = os.path.join(DOWNLOAD_DIR, run.id)
        os.makedirs(run_download_path, exist_ok=True)
        print(f"  Downloading files for run '{run.name}' (ID: {run.id})...")
        for file in run.files():
            file_path = os.path.join(run_download_path, file.name)
            try:
                file.download(root=run_download_path)
                print(f"    Downloaded: {file.name}")
            except Exception as e:
                print(f"    Error downloading {file.name}: {e}")

    print("Download complete.")