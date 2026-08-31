import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
import utils
import os
from pubo_solver import pubo_encode_sat_prob
from qubo_solver import qubo_encode_sat_prob
import config

def run_anneal(encode, is_qubo, steps, start_temp, end_temp):
    device = encode["linear"].device
    total_vars = encode["total_num_vars"] if is_qubo else encode["num_vars"]
    group = total_vars // 2
    spins = torch.randint(0, 2, (total_vars,), dtype=torch.float32, device=device)

    history = []
    for step in range(steps):
        history.append(utils.count_unsatisfied_clauses(spins[:encode["num_vars"]], encode["clauses"]))

        temp = start_temp * ((end_temp / start_temp) ** (step / steps))
        if step % 2 == 0:
            all_indices = torch.randperm(total_vars, device=device)
            batch_indices = all_indices[:group]
            complement_indices = all_indices[group:]
        else:
            batch_indices = complement_indices

        if is_qubo:
            grad = utils.qubo_compute_gradient(spins, encode)
        else:
            grad = utils.pubo_compute_gradient(spins, encode)

        noise = (torch.rand(len(batch_indices), device=device) * 2 * temp - temp)
        spins[batch_indices] = (grad[batch_indices] < noise).float()

    return history

if __name__ == "__main__":
    """!!! change this according to the current sat size !!!"""
    sat_size = 100
    print(f"sat_size: {sat_size}")

    files = sorted(glob.glob(config.MULTIPLE_FILES_PATH))[:config.INSTANCE]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qubo_results, pubo_results = [], []

    count = 1
    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()
        pubo_enc = pubo_encode_sat_prob(lines, device, sigma=config.SIGMA)
        qubo_enc = qubo_encode_sat_prob(lines, device, sigma=config.SIGMA, penalty=config.PENALTY)

        for _ in range(config.RUNS_PER_INSTANCE):
            pubo_results.append(run_anneal(pubo_enc, is_qubo=False, steps=config.STEPS, start_temp=config.START_TEMP, end_temp=config.END_TEMP))
            qubo_results.append(run_anneal(qubo_enc, is_qubo=True, steps=config.STEPS, start_temp=config.START_TEMP, end_temp=config.END_TEMP))

        print(f"run {count}/{len(files)} | file: {os.path.basename(file)}")
        count += 1
        
    qubo_mat = np.array(qubo_results)
    pubo_mat = np.array(pubo_results)

    plt.figure(figsize=(8, 6), dpi=300)
    steps_x = np.arange(config.STEPS)

    for name, mat, color in [("QUBO", qubo_mat, "#1f77b4"),("PUBO", pubo_mat, "#ff7f0e")]:
        for p_low, p_high, alpha in [
            (5, 95, 0.15),
            (15, 85, 0.25),
            (25, 75, 0.35),
        ]:
            plt.fill_between(
                steps_x,
                np.percentile(mat, p_low, axis=0),
                np.percentile(mat, p_high, axis=0),
                color=color,
                alpha=alpha,
                linewidth=0,
            )
        plt.plot(
            steps_x,
            np.mean(mat, axis=0),
            color=color,
            linewidth=2.5,
            label=f"{name} PU Mean {sat_size}",
        )

    
    output_dir="./visualizations"
    output_path = os.path.join(output_dir, f"unsats_vs_steps_{sat_size}.png")
    plt.xlabel("Steps", fontsize=11, fontweight="bold")
    plt.ylabel("Unsatisfied Clauses", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()