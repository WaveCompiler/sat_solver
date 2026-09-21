import glob
import re
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pubo_success_eval
import qubo_success_eval
import torch

DATASET_PATHS = {
    20: "/home/taehy/sat/sat_problem_dataset/uf20-91.tar/*.cnf",
    # 50: "/home/taehy/sat/sat_problem_dataset/uf50-218.tar/*.cnf",
    # 100: "/home/taehy/sat/sat_problem_dataset/uf100-430.tar/*.cnf",
    # 150: "/home/taehy/sat/sat_problem_dataset/uf150-645.tar/ai/hoos/Research/SAT/Formulae/UF150.645.100/*.cnf",
    200: "/home/taehy/sat/sat_problem_dataset/uf200-860.tar/uf200-860/*.cnf",
}

INSTANCE_RANGES = {
    20: (901, 1000),
    50: (901, 1000),
    100: (901, 1000),
    150: (1, 100),
    200: (1, 100),
}

TARGET_SUCCESS_RATE = 0.99

# pubo
PUBO_SIGMA = 2 ** -8
PUBO_STEPS = 1000
PUBO_START_TEMP = 1.0
PUBO_END_TEMP = 0.1
PUBO_LOG_INTERVAL = 100

# qubo
QUBO_SIGMA = 2 ** -8
QUBO_STEPS = 3000
QUBO_START_TEMP = 1.0
QUBO_END_TEMP = 0.1
QUBO_LOG_INTERVAL = 500
QUBO_PENALTY = 0.5

RUNS_PER_INSTANCE = 100
PUBO_BATCH_SIZE = 100
QUBO_BATCH_SIZE = 100

MAX_PENALTY_FACTOR = 1e3

# hardware constants
PUBO_TPI_PU = 1.5e-6
QUBO_TPI_PU = 1.0e-6

BASE_VARS = 20.0
PUBO_GROUP_SLICE = [2, 2, 2, 2, 2]
QUBO_GROUP_SLICE = [2, 2, 2, 2, 2]

def extract_file_index(filepath):
    match = re.search(r'-0*(\d+)\.cnf$', filepath)
    return int(match.group(1)) if match else -1

def load_sat_instances(mode):
    dataset_files = {}
    for sat_size, glob_pattern in DATASET_PATHS.items():
        raw_files = glob.glob(glob_pattern, recursive=True)
        sorted_files = sorted(raw_files, key=extract_file_index)
        start_idx, end_idx = INSTANCE_RANGES[sat_size]
        valid_files = [f for f in sorted_files if start_idx <= extract_file_index(f) <= end_idx]
        
        if mode == "optimization":
            dataset_files[sat_size] = valid_files[:20]
        elif mode == "evaluation":
            dataset_files[sat_size] = valid_files[20:100]
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'optimization' or 'evaluation'.")
    return dataset_files

if __name__ == "__main__":
    # select mode: "optimization" or "evaluation"
    mode = "optimization"
    # mode = "evaluation"
    selected_dataset = load_sat_instances(mode=mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    for sat_size, files in selected_dataset.items():
        print(f"Size {sat_size} ({mode} mode): loaded {len(files)} files")
        for file in files:
            print(file)

        pubo_obj_arr = []
        qubo_obj_arr = []
        log_failure_allowance_level = math.log(1 - TARGET_SUCCESS_RATE)

        for file_idx, file in enumerate(files):
            print(f"process file: {file}...")
            
            with open(file, "r") as f:
                lines = f.readlines()

            pubo_enc = pubo_success_eval.pubo_encode_sat_prob(lines, device, sigma=PUBO_SIGMA)
            qubo_enc = qubo_success_eval.qubo_encode_sat_prob(lines, device, sigma=QUBO_SIGMA, penalty=QUBO_PENALTY)

            size_keys = list(DATASET_PATHS.keys())
            size_idx = size_keys.index(sat_size)
            pubo_slice = PUBO_GROUP_SLICE[size_idx]
            qubo_slice = QUBO_GROUP_SLICE[size_idx]
            print(f"pubo_slice: {pubo_slice}")
            print(f"qubo_slice: {qubo_slice}")
            
            pubo_num_satisfied, pubo_mean_solved_steps = pubo_success_eval.pubo_subgroup_update_simulated_annealing(pubo_enc, device, PUBO_STEPS, PUBO_START_TEMP, PUBO_END_TEMP, pubo_slice, PUBO_LOG_INTERVAL, PUBO_BATCH_SIZE)
            print(f"pubo_num_satisfied: {pubo_num_satisfied}")
            print(f"pubo_mean_solved_steps: {pubo_mean_solved_steps}")
            qubo_num_satisfied, qubo_mean_solved_steps = qubo_success_eval.qubo_subgroup_update_simulated_annealing(qubo_enc, device, QUBO_STEPS, QUBO_START_TEMP, QUBO_END_TEMP, qubo_slice, QUBO_LOG_INTERVAL, QUBO_BATCH_SIZE)
            print(f"qubo_num_satisfied: {qubo_num_satisfied}")
            print(f"qubo_mean_solved_steps: {qubo_mean_solved_steps}")

            pubo_success_count = pubo_num_satisfied
            qubo_success_count = qubo_num_satisfied
            
            pubo_pos = pubo_success_count / RUNS_PER_INSTANCE
            qubo_pos = qubo_success_count / RUNS_PER_INSTANCE

            if pubo_pos >= 1.0:
                pubo_its = pubo_mean_solved_steps
            elif pubo_pos <= 0.0:
                pubo_its = PUBO_STEPS * MAX_PENALTY_FACTOR
            else:
                pos_log_failure_rate = math.log(1 - pubo_pos)
                pos_perc_required_runs = log_failure_allowance_level / pos_log_failure_rate
                pubo_its = pubo_mean_solved_steps * pos_perc_required_runs
            
            if qubo_pos >= 1.0:
                qubo_its = qubo_mean_solved_steps
            elif qubo_pos <= 0.0:
                qubo_its = QUBO_STEPS * MAX_PENALTY_FACTOR
            else:
                qos_log_failure_rate = math.log(1 - qubo_pos)
                qos_perc_required_runs = log_failure_allowance_level / qos_log_failure_rate
                qubo_its = qubo_mean_solved_steps * qos_perc_required_runs

            pubo_obj = (pubo_pos, pubo_its)
            qubo_obj = (qubo_pos, qubo_its)
            
            pubo_obj_arr.append(pubo_obj)
            qubo_obj_arr.append(qubo_obj)
            print(f"file [{file_idx+1}/{len(files)}] | PUBO PoS={pubo_pos:.4f}, ITS={pubo_its:.4f} | QUBO PoS={qubo_pos:.4f}, ITS={qubo_its:.4f}")

        pubo_obj_arr_np = np.array(pubo_obj_arr)
        qubo_obj_arr_np = np.array(qubo_obj_arr)

        pubo_pos_vals = pubo_obj_arr_np[:, 0]
        pubo_tts_vals = pubo_obj_arr_np[:, 1] * PUBO_TPI_PU

        qubo_pos_vals = qubo_obj_arr_np[:, 0]
        qubo_tts_vals = qubo_obj_arr_np[:, 1] * QUBO_TPI_PU

        # plot
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        ax.scatter(
            qubo_tts_vals,
            qubo_pos_vals,
            color="#1f77b4",
            edgecolors="black",
            linewidths=0.5,
            label="QUBO-PU",
            alpha=0.6,
            s=45,
        )
        ax.scatter(
            pubo_tts_vals,
            pubo_pos_vals,
            color="#ff7f0e",
            edgecolors="black",
            linewidths=0.5,
            label="PUBO-PU",
            marker="s",
            alpha=0.6,
            s=45,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\mathrm{TTS}_{0.99}$ in seconds (log scale)", fontsize=11)
        ax.set_ylabel("Probability of Success (PoS)", fontsize=11)
        ax.set_title(
            f"PoS vs. TTS for SAT Size {sat_size} ({mode.capitalize()} Mode)",
            fontsize=12,
        )
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(loc="best")

        plt.tight_layout()
        os.makedirs("./visualizations", exist_ok=True)
        save_path = f"./visualizations/pos_tts_plot_size_{sat_size}.png"
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
