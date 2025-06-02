import os
from collections import defaultdict
import numpy as np

if __name__ == "__main__":
    log_dir = "debug/mt_player"
    runname_to_exps = {
        "PaCO-easy": [f for f in os.listdir(log_dir) if "last_0220_ppo_go1_easy_paco" in f],
        "PaCO-hard": [f for f in os.listdir(log_dir) if "last_0220_ppo_go1_hard_paco" in f],
        "PaCO-hard-cl": [f for f in os.listdir(log_dir) if "last_0220_ppo_go1_hard_cl_paco" in f],
        "MOORE-easy": [f for f in os.listdir(log_dir) if "last_0220_ppo_go1_easy_moore" in f],
        "MOORE-hard": [f for f in os.listdir(log_dir) if "last_0220_ppo_go1_hard_moore" in f],
        "MOORE-hard-cl": [f for f in os.listdir(log_dir) if "last_0220_ppo_go1_hard_cl_moore" in f],
        "Multihead-easy": [f for f in os.listdir(log_dir) if "last_0220_ppo_go1_easy_multihead" in f],
        "Multihead-hard": [f for f in os.listdir(log_dir) if "last_0220_ppo_go1_hard_multihead" in f],
        "Multihead-hard-cl": [f for f in os.listdir(log_dir) if "last_0220_ppo_go1_hard_cl_multihead" in f],
        "Soft-Modularization-easy": [f for f in os.listdir(log_dir) if "last_0220_ppo_go1_easy_soft-modularization" in f],
        "Soft-Modularization-hard": [f for f in os.listdir(log_dir) if "last_0220_ppo_go1_hard_soft-modularization" in f],
        "Soft-Modularization-hard-cl": [f for f in os.listdir(log_dir) if "last_0220_ppo_go1_hard_cl_soft-modularization" in f],
    }

    success_rate = defaultdict(lambda: defaultdict(list))

    for runname in runname_to_exps.keys():
        for exp in runname_to_exps[runname]:
            success_log_file = os.path.join(log_dir, exp, "success_rate.txt")
            successes = open(success_log_file, "r").readlines()
            # append mean success
            successes = [float(s.strip()) for s in successes]
            success_rate[runname]["success"].append(np.mean(successes))

    for runname in success_rate.keys():
        print(f"{runname}: mean {np.mean(success_rate[runname]['success'])}; std {np.std(success_rate[runname]['success'])}; support: {len(success_rate[runname]['success'])}")
        print(success_rate[runname]['success'])
            