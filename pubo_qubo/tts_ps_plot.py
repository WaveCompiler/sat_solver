import glob
import re
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import sat_solver.pubo.pubo_success_eval as pubo_success_eval
import sat_solver.pubo.qubo_success_eval as qubo_success_eval
import torch

DATASET_PATHS = {
    20: "/home/taehy/sat/sat_problem_dataset/uf20-91.tar/*.cnf",
    50: "/home/taehy/sat/sat_problem_dataset/uf50-218.tar/*.cnf",
    100: "/home/taehy/sat/sat_problem_dataset/uf100-430.tar/*.cnf",
    150: "/home/taehy/sat/sat_problem_dataset/uf150-645.tar/ai/hoos/Research/SAT/Formulae/UF150.645.100/*.cnf",
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
PUBO_MAX_STEPS = 1000
PUBO_START_TEMP = 1.0
PUBO_END_TEMP = 0.1
PUBO_LOG_INTERVAL = 100

# qubo
QUBO_SIGMA = 2 ** -8
QUBO_MAX_STEPS = 3000
QUBO_START_TEMP = 1.0
QUBO_END_TEMP = 0.1
QUBO_LOG_INTERVAL = 500
QUBO_PENALTY = 0.5

RUNS_PER_INSTANCE = 100
PUBO_BATCH_SIZE = 100
QUBO_BATCH_SIZE = 100

PENALTY_FACTOR = 1e3

# hardware constants
PUBO_TPI_PU = 2.0e-9
QUBO_TPI_PU = 2.0e-9
PUBO_EPI_PU = 5.0e-9
QUBO_EPI_PU = 1.0e-8

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

    sizes_evaluated = []
    tts_qubo_means, tts_qubo_errors = [], []
    tts_pubo_means, tts_pubo_errors = [], []
    ets_qubo_means, ets_qubo_errors = [], []
    ets_pubo_means, ets_pubo_errors = [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    for sat_size, files in selected_dataset.items():
        print(f"Size {sat_size} ({mode} mode): loaded {len(files)} files")
        for file in files:
            print(file)

        sizes_evaluated.append(sat_size)
        pubo_its_arr = []
        qubo_its_arr = []
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
            
            pubo_num_satisfied, pubo_mean_solved_steps = pubo_success_eval.pubo_subgroup_update_simulated_annealing(pubo_enc, device, PUBO_MAX_STEPS, PUBO_START_TEMP, PUBO_END_TEMP, pubo_slice, PUBO_LOG_INTERVAL, PUBO_BATCH_SIZE)
            print(f"pubo_num_satisfied: {pubo_num_satisfied}")
            print(f"pubo_mean_solved_steps: {pubo_mean_solved_steps}")
            qubo_num_satisfied, qubo_mean_solved_steps = qubo_success_eval.qubo_subgroup_update_simulated_annealing(qubo_enc, device, QUBO_MAX_STEPS, QUBO_START_TEMP, QUBO_END_TEMP, qubo_slice, QUBO_LOG_INTERVAL, QUBO_BATCH_SIZE)
            print(f"qubo_num_satisfied: {qubo_num_satisfied}")
            print(f"qubo_mean_solved_steps: {qubo_mean_solved_steps}")

            pubo_success_count = pubo_num_satisfied
            qubo_success_count = qubo_num_satisfied
            
            pubo_pos = pubo_success_count / RUNS_PER_INSTANCE
            qubo_pos = qubo_success_count / RUNS_PER_INSTANCE

            if pubo_pos >= 1.0:
                pubo_its = pubo_mean_solved_steps
            elif pubo_pos <= 0.0:
                pubo_its = PUBO_MAX_STEPS * PENALTY_FACTOR
            else:
                pos_log_failure_rate = math.log(1 - pubo_pos)
                pos_perc_required_runs = log_failure_allowance_level / pos_log_failure_rate
                pubo_its = pubo_mean_solved_steps * pos_perc_required_runs
            
            if qubo_pos >= 1.0:
                qubo_its = qubo_mean_solved_steps
            elif qubo_pos <= 0.0:
                qubo_its = QUBO_MAX_STEPS * PENALTY_FACTOR
            else:
                qos_log_failure_rate = math.log(1 - qubo_pos)
                qos_perc_required_runs = log_failure_allowance_level / qos_log_failure_rate
                qubo_its = qubo_mean_solved_steps * qos_perc_required_runs
            
            pubo_its_arr.append(pubo_its)
            qubo_its_arr.append(qubo_its)

            print(f"file [{file_idx+1}/{len(files)}] | PUBO PoS={pubo_pos:.4f}, ITS={pubo_its:.4f} | QUBO PoS={qubo_pos:.4f}, ITS={qubo_its:.4f}")

        pubo_its_arr_np = np.array(pubo_its_arr)
        qubo_its_arr_np = np.array(qubo_its_arr)

        pubo_ets_size_growth = pubo_enc["num_vars"] / BASE_VARS
        qubo_ets_size_growth = qubo_enc["total_num_vars"] / BASE_VARS

        pubo_tts = pubo_its_arr_np * PUBO_TPI_PU 
        qubo_tts = qubo_its_arr_np * QUBO_TPI_PU
        pubo_ets = pubo_its_arr_np * PUBO_EPI_PU * pubo_ets_size_growth
        qubo_ets = qubo_its_arr_np * QUBO_EPI_PU * qubo_ets_size_growth

        tts_pubo_means.append(np.mean(pubo_tts))
        tts_pubo_errors.append(np.std(pubo_tts) / np.sqrt(len(pubo_tts)))
        tts_qubo_means.append(np.mean(qubo_tts))
        tts_qubo_errors.append(np.std(qubo_tts) / np.sqrt(len(qubo_tts)))
        ets_pubo_means.append(np.mean(pubo_ets))
        ets_pubo_errors.append(np.std(pubo_ets) / np.sqrt(len(pubo_ets)))
        ets_qubo_means.append(np.mean(qubo_ets))
        ets_qubo_errors.append(np.std(qubo_ets) / np.sqrt(len(qubo_ets)))

    # -------------------------------------------------------------
    # PLOTTING FIGURE 3
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    # (a) TTS
    ax1.errorbar(
        sizes_evaluated,
        tts_qubo_means,
        yerr=tts_qubo_errors,
        fmt="-o",
        capsize=3,
        color="#1f77b4",
        label="QUBO-PU",
    )
    ax1.errorbar(
        sizes_evaluated,
        tts_pubo_means,
        yerr=tts_pubo_errors,
        fmt="-s",
        capsize=3,
        color="#ff7f0e",
        label="PUBO-PU",
    )

    ax1.set_yscale("log")
    ax1.set_xlabel("Problem Size", fontsize=11)
    ax1.set_ylabel(r"$\mathrm{TTS}_{0.99}$ in seconds", fontsize=11)
    ax1.set_xticks(sizes_evaluated)
    ax1.set_title("(a)", fontsize=12)
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)
    ax1.legend()

    # (b) ETS
    ax2.errorbar(
        sizes_evaluated,
        ets_qubo_means,
        yerr=ets_qubo_errors,
        fmt="-o",
        capsize=3,
        color="#1f77b4",
        label="QUBO-PU",
    )
    ax2.errorbar(
        sizes_evaluated,
        ets_pubo_means,
        yerr=ets_pubo_errors,
        fmt="-s",
        capsize=3,
        color="#ff7f0e",
        label="PUBO-PU",
    )

    ax2.set_yscale("log")
    ax2.set_xlabel("Problem Size", fontsize=11)
    ax2.set_ylabel(r"$\mathrm{ETS}_{0.99}$ in joules", fontsize=11)
    ax2.set_xticks(sizes_evaluated)
    ax2.set_title("(b)", fontsize=12)
    ax2.grid(True, which="both", linestyle="--", alpha=0.3)

    plt.tight_layout()
    os.makedirs("./visualizations", exist_ok=True)
    save_path = "./visualizations/tts_ps_plot.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"\nSaved scaling plot successfully to {save_path}")