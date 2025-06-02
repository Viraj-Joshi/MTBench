import os
import numpy as np

from matplotlib import pyplot as plt

if __name__ == "__main__":
    path = "wandb/run-20241003_210113-uid_0926-mt-non_termination-FAMO_03-21-01-13/files/output.log"

    logits = []
    deltas = []
    losses = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("weights"):
                logits.append(np.array([float(x) for x in line.split("logits [")[1].split("]")[0].split(" ") if x]))
            
            if line.startswith("delta"):
                deltas.append(np.array([float(x) for x in line.split("delta [")[1].split("]")[0].split(" ") if x]))

            if line.startswith("prev_loss"):
                losses.append(np.array([float(x) for x in line.split("prev_loss [")[1].split("]")[0].split(" ") if x]))
                


    logits = np.array(logits)

    #for i in range(logits.shape[1]):
    i = 14
    plt.plot(logits[:, i], label=f"logit {i}")

    plt.legend()
    plt.savefig(f"debug/famo/logits.png", dpi=300, bbox_inches="tight")
    plt.close()

    deltas = np.array(deltas)

    # for i in range(deltas.shape[1]):
    i = 14
    plt.plot(deltas[:, i], label=f"delta {i}")

    plt.legend()
    plt.savefig(f"debug/famo/deltas.png", dpi=300, bbox_inches="tight")
    plt.close()

    losses = np.array(losses)

    # for i in range(losses.shape[1]):
    i = 14
    plt.plot(losses[:, i], label=f"loss {i}")

    plt.legend()
    plt.savefig(f"debug/famo/losses.png", dpi=300, bbox_inches="tight")
    plt.close()

