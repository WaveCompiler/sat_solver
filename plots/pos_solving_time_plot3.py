import os
import sys

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import matplotlib.pyplot as plt
import numpy as np
import torch
import utils

from g2wsat import G2WSAT
from gsat import GSAT
from pubo import (
    PUBO,
    PUBO_SIGMA,
    PUBO_STEPS,
    PUBO_START_TEMP,
    PUBO_END_TEMP,
    PUBO_GROUP_SLICE
)
from walksat import WalkSAT
from adapt_g2wsat import AdaptG2WSATP

DATASETS = {
    "uf20": ("/DATA/FCD_LAB/user1/TH/dataset/uf20-91", "uf20", 10),
    "uf50": ("/DATA/FCD_LAB/user1/TH/dataset/uf50-218", "uf50", 10),
    "uf100": ("/DATA/FCD_LAB/user1/TH/dataset/uf100-430", "uf100", 10),
    "uf150": (
        "/DATA/FCD_LAB/user1/TH/dataset/uf150-645/ai/hoos/Research/SAT/Formulae/UF150.645.100",
        "uf150",
        10,
    ),
    "uf200": ("/DATA/FCD_LAB/user1/TH/dataset/uf200-860", "uf200", 10),
    "uf250": (
        "/DATA/FCD_LAB/user1/TH/dataset/uf250-1065/ai/hoos/Shortcuts/UF250.1065.100",
        "uf250",
        10,
    ),
}

RUNS_PER_FILE = 1
MAX_STEPS = 10000
MAX_TRIES = 1

def calculate_pos_vectorized(steps_list, steps_arr):
    solved_steps = np.sort([s for s in steps_list if s != float("inf")])
    if len(solved_steps) == 0:
        return np.zeros_like(steps_arr, dtype=np.float64)
    counts = np.searchsorted(solved_steps, steps_arr, side="right")
    return counts / RUNS_PER_FILE

def run_benchmark_for_size(size_key):
    base_dir, prefix, num_files = DATASETS[size_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps_arr = np.arange(1, MAX_STEPS + 1)

    all_pubo_pos = []
    all_pubo_escape_walk1_pos = []
    all_pubo_escape_walk2_pos = []
    all_pubo_escape_walk3_pos = []
    all_g2wsat_pos = []

    prob_p = 0.5
    div_prob_dp = 0.01
    
    print(f"\n==================================================")
    print(f" Running Benchmark for Size: {size_key.upper()} ({num_files} files, {RUNS_PER_FILE} runs/file)")
    print(f"==================================================")

    for file_idx in range(1, num_files + 1):
        if file_idx < 10:
            file_name = f"{prefix}-{file_idx:02d}.cnf"  # e.g., uf20-01.cnf
        else:
            file_name = f"{prefix}-{file_idx:03d}.cnf"
        file_path = os.path.join(base_dir, file_name)

        if not os.path.exists(file_path):
            print(f"Warning: File not found -> {file_path}. Skipping.")
            continue

        print(f"[{file_idx}/{num_files}] Processing {file_name}...")

        with open(file_path, "r") as f:
            num_vars, num_clauses, clauses = utils.parse_sat_prob(f)

        pubo_steps = []
        pubo_escape_walk1_steps = []
        pubo_escape_walk2_steps = []
        pubo_escape_walk3_steps = []
        g2wsat_steps = []

        pubo_solver = PUBO(
            num_vars=num_vars,
            num_clauses=num_clauses,
            clauses=clauses,
            device=device,
        )
        for r in range(RUNS_PER_FILE):
            status, _, step, _ = pubo_solver.solve_simulated_annealing(
                max_steps=PUBO_STEPS,
                max_tries=MAX_TRIES,
                start_temp=PUBO_START_TEMP,
                end_temp=PUBO_END_TEMP,
                group_slice=PUBO_GROUP_SLICE,
                sigma=PUBO_SIGMA,
            )
            pubo_steps.append(step if status else float("inf"))
        print(f"pubo_steps:", pubo_steps)

        for r in range(RUNS_PER_FILE):
            status, _, step, _ = pubo_solver.solve_escape_walk(
                max_steps=PUBO_STEPS,
                max_tries=MAX_TRIES,
                start_temp=PUBO_START_TEMP,
                end_temp=PUBO_END_TEMP,
                group_slice=PUBO_GROUP_SLICE,
                sigma=PUBO_SIGMA,
                escape_prob=0.4,
            )
            pubo_escape_walk1_steps.append(step if status else float("inf"))
        print(f"pubo_escape_walk1_steps:", pubo_escape_walk1_steps)

        for r in range(RUNS_PER_FILE):
            status, _, step, _ = pubo_solver.solve_escape_walk(
                max_steps=PUBO_STEPS,
                max_tries=MAX_TRIES,
                start_temp=PUBO_START_TEMP,
                end_temp=PUBO_END_TEMP,
                group_slice=PUBO_GROUP_SLICE,
                sigma=PUBO_SIGMA,
                escape_prob=0.5,
            )
            pubo_escape_walk2_steps.append(step if status else float("inf"))
        print(f"pubo_escape_walk2_steps:", pubo_escape_walk2_steps)

        for r in range(RUNS_PER_FILE):
            status, _, step, _ = pubo_solver.solve_escape_walk(
                max_steps=PUBO_STEPS,
                max_tries=MAX_TRIES,
                start_temp=PUBO_START_TEMP,
                end_temp=PUBO_END_TEMP,
                group_slice=PUBO_GROUP_SLICE,
                sigma=PUBO_SIGMA,
                escape_prob=0.6,
            )
            pubo_escape_walk3_steps.append(step if status else float("inf"))
        print(f"pubo_escape_walk3_steps:", pubo_escape_walk3_steps)

        g2wsat_solver = G2WSAT(
            num_vars=num_vars, num_clauses=num_clauses, clauses=clauses
        )
        for r in range(RUNS_PER_FILE):
            status, _, step, _ = g2wsat_solver.solve_novelty_plus_plus(
                max_steps=MAX_STEPS,
                max_tries=MAX_TRIES,
                prob_p=prob_p,
                div_prob_dp=div_prob_dp,
            )
            g2wsat_steps.append(step if status else float("inf"))
        print(f"g2wsat_steps:", g2wsat_steps)

        all_pubo_pos.append(calculate_pos_vectorized(pubo_steps, steps_arr))
        all_pubo_escape_walk1_pos.append(calculate_pos_vectorized(pubo_escape_walk1_steps, steps_arr))
        all_pubo_escape_walk2_pos.append(calculate_pos_vectorized(pubo_escape_walk2_steps, steps_arr))
        all_pubo_escape_walk3_pos.append(calculate_pos_vectorized(pubo_escape_walk3_steps, steps_arr))
        all_g2wsat_pos.append(calculate_pos_vectorized(g2wsat_steps, steps_arr))

    if len(all_pubo_pos) == 0:
        print(f"No valid files processed for {size_key}. Skipping plot.")
        return

    avg_pubo_pos = np.mean(all_pubo_pos, axis=0)
    avg_pubo_escape_walk1_pos = np.mean(all_pubo_escape_walk1_pos, axis=0)
    avg_pubo_escape_walk2_pos = np.mean(all_pubo_escape_walk2_pos, axis=0)
    avg_pubo_escape_walk3_pos = np.mean(all_pubo_escape_walk3_pos, axis=0)
    avg_g2wsat_pos = np.mean(all_g2wsat_pos, axis=0)

    # Plotting
    plt.figure(figsize=(10, 6), dpi=300)

    plt.plot(
        steps_arr,
        avg_pubo_pos,
        label="pubo parallel update",
        color="#d62728",
        linewidth=2.0,
        alpha=0.85,
        zorder=5,
    )
    plt.plot(
        steps_arr,
        avg_pubo_escape_walk1_pos,
        label="pubo_escape_walk (escape_prob=0.4)",
        color="#efe30a",
        linewidth=2.0,
        alpha=0.85,
        zorder=4,
    )
    plt.plot(
        steps_arr,
        avg_pubo_escape_walk2_pos,
        label="pubo_escape_walk (escape_prob=0.5)",
        color="#ff7f0e",
        linewidth=2.0,
        alpha=0.85,
        zorder=3,
    )
    plt.plot(
        steps_arr,
        avg_pubo_escape_walk3_pos,
        label="pubo_escape_walk (escape_prob=0.6)",
        color="#2ca02c",
        linewidth=2.0,
        alpha=0.85,
        zorder=2,
    )
    plt.plot(
        steps_arr,
        avg_g2wsat_pos,
        label="G2wSAT (Novelty++)",
        color="#1f77b4",
        linewidth=2.0,
        alpha=0.85,
        zorder=1,
    )

    plt.xscale("log")
    plt.xlabel("Steps (Log Scale)", fontsize=12)
    plt.ylabel("Avg PoS", fontsize=12)
    plt.title(
        f"Avg PoS vs. Steps ({size_key.upper()}, {len(all_pubo_pos)} Instances × {RUNS_PER_FILE} Runs)",
        fontsize=13,
    )
    plt.ylim(-0.02, 1.02)
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend(loc="lower right", fontsize=11)

    os.makedirs("./visualizations", exist_ok=True)
    output_file = f"./visualizations/avg_pos_escape_walk_test_{size_key}_log.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Successfully saved plot for {size_key.upper()} to {output_file}")

def run_all_benchmarks():
    for size_key in DATASETS.keys():
        run_benchmark_for_size(size_key)

if __name__ == "__main__":
    run_all_benchmarks()