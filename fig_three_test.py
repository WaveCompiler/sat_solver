import glob
import re
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pubo_success_eval
import qubo_success_eval
import torch

# Map problem size to its corresponding directory glob pattern
DATASET_PATHS = {
    20: "/home/taehy/sat/sat_problem_dataset/uf20-91.tar/*.cnf",  # Add your size 20 path if available
    50: "/home/taehy/sat/sat_problem_dataset/uf50-218.tar/*.cnf",  # Add your size 50 path if available
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

# Hyperparameters
NUM_FILES_PER_SIZE = 20  # Limit to first 20 instances per problem size
RUNS_PER_INSTANCE = 100
TARGET_SUCCESS_RATE = 0.99

# pubo
PUBO_SIGMA = 2 ** -3
PUBO_STEPS = 500
PUBO_START_TEMP = 1.0
PUBO_END_TEMP = 0.01
PUBO_LOG_INTERVAL = 100
PUBO_BATCH_SIZE = 100

# qubo
QUBO_SIGMA = 2 ** -7
QUBO_STEPS = 5000
QUBO_START_TEMP = 1.0
QUBO_END_TEMP = 0.1
QUBO_LOG_INTERVAL = 250
QUBO_PENALTY = 0.5
QUBO_BATCH_SIZE = 100

# Hardware Constants
# TPI_QUBO_PU = 1.0e-6
# TPI_PUBO_PU = 1.5e-6
# EPI_QUBO_PU = 1.0e-8
# EPI_PUBO_PU = 5.0e-9

TPI_QUBO_PU = 1
TPI_PUBO_PU = 1
EPI_QUBO_PU = 1
EPI_PUBO_PU = 1

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
            dataset_files[sat_size] = valid_files[:20]  #
        elif mode == "evaluation":
            dataset_files[sat_size] = valid_files[20:100]  #
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'optimization' or 'evaluation'.")

    return dataset_files

if __name__ == "__main__":
    # select mode: "optimization" or "evaluation"
    # mode = "optimization"
    mode = "evaluation"
    selected_dataset = load_sat_instances(mode=mode)

    # Container for final metrics per size
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

        arr_pubo_its = []
        arr_qubo_its = []
        
        log_failure_allowance_level = math.log(1 - TARGET_SUCCESS_RATE)

        for file_idx, file in enumerate(files):
            print(f"process file: {file}...")
            
            with open(file, "r") as f:
                lines = f.readlines()

            pubo_enc = pubo_success_eval.pubo_encode_sat_prob(lines, device, sigma=PUBO_SIGMA)
            qubo_enc = qubo_success_eval.qubo_encode_sat_prob(lines, device, sigma=QUBO_SIGMA, penalty=QUBO_PENALTY)

            pubo_group = pubo_enc["num_vars"] // 2
            qubo_group = qubo_enc["total_num_vars"] // 2

            pubo_success_count = 0
            qubo_success_count = 0

            # Run independent annealing attempts
            pubo_rates = pubo_success_eval.pubo_subgroup_update_simulated_annealing(pubo_enc, device, PUBO_STEPS, PUBO_START_TEMP, PUBO_END_TEMP, pubo_group, PUBO_LOG_INTERVAL, PUBO_BATCH_SIZE)
            qubo_rates = qubo_success_eval.qubo_subgroup_update_simulated_annealing(qubo_enc, device, QUBO_STEPS, QUBO_START_TEMP, QUBO_END_TEMP, qubo_group, QUBO_LOG_INTERVAL, QUBO_BATCH_SIZE)
            
            pubo_successful_runs = pubo_rates >= TARGET_SUCCESS_RATE 
            pubo_success_count = pubo_successful_runs.sum().item()
            qubo_successful_runs = qubo_rates >= TARGET_SUCCESS_RATE 
            qubo_success_count = qubo_successful_runs.sum().item()
            
            # Compute Single-Cycle Probabilities
            pos_pubo = pubo_success_count / RUNS_PER_INSTANCE
            pos_qubo = qubo_success_count / RUNS_PER_INSTANCE

            # Calculate PUBO ITS
            if pos_pubo >= 1.0:
                its_pubo = PUBO_STEPS
            elif pos_pubo <= 0.0:
                its_pubo = np.nan
            else:
                pos_log_failure_rate = math.log(1 - pos_pubo)
                pos_perc_required_runs = log_failure_allowance_level / pos_log_failure_rate
                its_pubo = PUBO_STEPS * pos_perc_required_runs

            # Calculate QUBO ITS
            if pos_qubo == 1.0:
                its_qubo = QUBO_STEPS
            elif pos_qubo == 0.0:
                its_qubo = np.nan
            else:
                qos_log_failure_rate = math.log(1 - pos_qubo)
                qos_perc_required_runs = log_failure_allowance_level / qos_log_failure_rate
                its_qubo = QUBO_STEPS * qos_perc_required_runs
            
            arr_pubo_its.append(its_pubo)
            arr_qubo_its.append(its_qubo)

            print(f"file [{file_idx+1}/{len(files)}] | PUBO PoS={pos_pubo:.4f}, ITS={its_pubo:.4f} | QUBO PoS={pos_qubo:.4f}, ITS={its_qubo:.4f}")

        # Convert ITS to TTS and ETS
        its_pubo_arr = np.array(arr_pubo_its)
        its_qubo_arr = np.array(arr_qubo_its)

        tts_pubo = its_pubo_arr * TPI_PUBO_PU
        tts_qubo = its_qubo_arr * TPI_QUBO_PU
        ets_pubo = its_pubo_arr * EPI_PUBO_PU
        ets_qubo = its_qubo_arr * EPI_QUBO_PU

        # Filter non-finite values
        v_tts_pubo = tts_pubo[np.isfinite(tts_pubo)]
        v_tts_qubo = tts_qubo[np.isfinite(tts_qubo)]
        v_ets_pubo = ets_pubo[np.isfinite(ets_pubo)]
        v_ets_qubo = ets_qubo[np.isfinite(ets_qubo)]

        # Compute Means and Standard Errors safely
        tts_pubo_means.append(np.mean(v_tts_pubo) if len(v_tts_pubo) > 0 else np.nan)
        tts_pubo_errors.append(np.std(v_tts_pubo) / np.sqrt(len(v_tts_pubo)) if len(v_tts_pubo) > 0 else np.nan)
        
        tts_qubo_means.append(np.mean(v_tts_qubo) if len(v_tts_qubo) > 0 else np.nan)
        tts_qubo_errors.append(np.std(v_tts_qubo) / np.sqrt(len(v_tts_qubo)) if len(v_tts_qubo) > 0 else np.nan)

        ets_pubo_means.append(np.mean(v_ets_pubo) if len(v_ets_pubo) > 0 else np.nan)
        ets_pubo_errors.append(np.std(v_ets_pubo) / np.sqrt(len(v_ets_pubo)) if len(v_ets_pubo) > 0 else np.nan)

        ets_qubo_means.append(np.mean(v_ets_qubo) if len(v_ets_qubo) > 0 else np.nan)
        ets_qubo_errors.append(np.std(v_ets_qubo) / np.sqrt(len(v_ets_qubo)) if len(v_ets_qubo) > 0 else np.nan)

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
    save_path = "./visualizations/fig3_tts_ets_scaling.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"\nSaved scaling plot successfully to {save_path}")