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
PUBO_STEPS = 500
PUBO_START_TEMP = 1.0
PUBO_END_TEMP = 0.01
PUBO_LOG_INTERVAL = 100

# qubo
QUBO_STEPS = 5000
QUBO_START_TEMP = 1.0
QUBO_END_TEMP = 0.1
QUBO_LOG_INTERVAL = 250
QUBO_PENALTY = 0.5

RUNS_PER_INSTANCE = 100
PUBO_BATCH_SIZE = 100
QUBO_BATCH_SIZE = 100

PUBO_MAX_ITS_PENALTY = PUBO_STEPS * 1e5
QUBO_MAX_ITS_PENALTY = QUBO_STEPS * 1e5

# hardware constants
TPI_QUBO_PU = 1.0e-6
TPI_PUBO_PU = 1.5e-6
EPI_QUBO_PU = 1.0e-8
EPI_PUBO_PU = 5.0e-9

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

def pubo_calculate_sigma(ebr, pubo_clean_enc):
    linear = pubo_clean_enc["linear"]
    quadratic = pubo_clean_enc["quadratic"]
    cubic = pubo_clean_enc["cubic"]
    
    all_weights = torch.cat([linear[linear != 0], quadratic[quadratic != 0], cubic[cubic != 0]])

    if len(all_weights) == 0:
        return 0.0

    dynamic_range = (torch.max(all_weights) - torch.min(all_weights)).item()
    print(f"pubo dynamic_range: {dynamic_range}")
    sigma = dynamic_range / ((2.0 ** ebr) * math.sqrt(12.0))
    return sigma

def qubo_calculate_sigma(ebr, qubo_clean_enc):
    linear = qubo_clean_enc["linear"]
    quadratic = qubo_clean_enc["quadratic"]
    
    all_weights = torch.cat([linear[linear != 0], quadratic[quadratic != 0]])

    dynamic_range = 0.0
    sigma = 0
    if len(all_weights) == 0:
        return sigma
    
    dynamic_range = (torch.max(all_weights) - torch.min(all_weights)).item()
    print(f"qubo dynamic_range: {dynamic_range}")
    sigma = dynamic_range / ((2.0 ** ebr) * math.sqrt(12.0))
    return sigma

def plot_ebr_vs_tts_for_size(sat_size, ebrs, pubo_means, pubo_errors, qubo_means, qubo_errors, save_dir="./visualizations"):
    print(f"DEBUG: len(ebrs) = {len(ebrs)}")
    print(f"DEBUG: len(pubo_means) = {len(pubo_means)}")
    print(f"DEBUG: len(qubo_means) = {len(qubo_means)}")

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

    ax.errorbar(
        ebrs,
        qubo_means,
        yerr=qubo_errors,
        fmt="--o",
        capsize=3,
        color="#1f77b4",
        linewidth=1.5,
        label="QUBO-PU",
    )
    
    ax.errorbar(
        ebrs,
        pubo_means,
        yerr=pubo_errors,
        fmt="-s",
        capsize=3,
        color="#ff7f0e",
        linewidth=2.0,
        label="PUBO-PU",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Effective Bit Resolution", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$\mathrm{TTS}_{0.99}$", fontsize=11, fontweight="bold")
    ax.set_title(f"3-SAT Benchmark Scaling (N={sat_size})", fontsize=12)
    
    # Invert X-axis from 8 down to 0
    ax.set_xlim(8.5, -0.5)
    ax.set_xticks(ebrs)

    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    filename = f"fig5_ebr_vs_tts_size_{sat_size}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Free memory for next iteration

    print(f"Saved plot for size {sat_size} to: {save_path}")

if __name__ == "__main__":
    ebrs = [8, 7, 6, 5, 4, 3, 2, 1, 0]

    # select mode: "optimization" or "evaluation"
    mode = "optimization"
    # mode = "evaluation"
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
        
        log_failure_allowance_level = math.log(1 - TARGET_SUCCESS_RATE)

        tts_qubo_means, tts_qubo_errors = [], []
        tts_pubo_means, tts_pubo_errors = [], []
        ets_qubo_means, ets_qubo_errors = [], []
        ets_pubo_means, ets_pubo_errors = [], []

        for ebr in ebrs:
            print(f"ebr: {ebr}")

            arr_pubo_its = []
            arr_qubo_its = []

            for file_idx, file in enumerate(files):
                print(f"process file: {file}...")

                with open(file, "r") as f:
                    lines = f.readlines()
                
                pubo_clean_enc = pubo_success_eval.pubo_encode_sat_prob(lines, device, 0)
                qubo_clean_enc = qubo_success_eval.qubo_encode_sat_prob(lines, device, 0, QUBO_PENALTY)
                pubo_sigma = pubo_calculate_sigma(ebr, pubo_clean_enc)
                qubo_sigma = qubo_calculate_sigma(ebr, qubo_clean_enc)
                print(f"pubo_sigma: {pubo_sigma}")
                print(f"qubo_sigma: {qubo_sigma}")
                
                pubo_enc = pubo_success_eval.pubo_encode_sat_prob(lines, device, pubo_sigma)
                qubo_enc = qubo_success_eval.qubo_encode_sat_prob(lines, device, qubo_sigma, QUBO_PENALTY)

                pubo_group = pubo_enc["num_vars"] // 2
                qubo_group = qubo_enc["total_num_vars"] // 2

                pubo_rates = pubo_success_eval.pubo_subgroup_update_simulated_annealing(pubo_enc, device, PUBO_STEPS, PUBO_START_TEMP, PUBO_END_TEMP, pubo_group, PUBO_LOG_INTERVAL, PUBO_BATCH_SIZE)
                qubo_rates = qubo_success_eval.qubo_subgroup_update_simulated_annealing(qubo_enc, device, QUBO_STEPS, QUBO_START_TEMP, QUBO_END_TEMP, qubo_group, QUBO_LOG_INTERVAL, QUBO_BATCH_SIZE)
                            
                pubo_successful_runs = pubo_rates >= TARGET_SUCCESS_RATE
                pubo_success_count = pubo_successful_runs.sum().item()
                qubo_successful_runs = qubo_rates >= TARGET_SUCCESS_RATE
                qubo_success_count = qubo_successful_runs.sum().item()
                
                pos_pubo = pubo_success_count / RUNS_PER_INSTANCE
                pos_qubo = qubo_success_count / RUNS_PER_INSTANCE

                # Calculate PUBO ITS
                if pos_pubo >= 1.0:
                    its_pubo = PUBO_STEPS
                elif pos_pubo <= 0.0:
                    its_pubo = PUBO_MAX_ITS_PENALTY
                else:
                    pos_log_failure_rate = math.log(1 - pos_pubo)
                    pos_perc_required_runs = log_failure_allowance_level / pos_log_failure_rate
                    its_pubo = PUBO_STEPS * pos_perc_required_runs

                # Calculate QUBO ITS
                if pos_qubo == 1.0:
                    its_qubo = QUBO_STEPS
                elif pos_qubo == 0.0:
                    its_qubo = QUBO_MAX_ITS_PENALTY
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

            # Compute Means and Standard Errors
            tts_pubo_means.append(np.mean(tts_pubo))
            tts_pubo_errors.append(np.std(tts_pubo) / np.sqrt(len(tts_pubo)))
            tts_qubo_means.append(np.mean(tts_qubo))
            tts_qubo_errors.append(np.std(tts_qubo) / np.sqrt(len(tts_qubo)))
            ets_pubo_means.append(np.mean(ets_pubo))
            ets_pubo_errors.append(np.std(ets_pubo) / np.sqrt(len(ets_pubo)))
            ets_qubo_means.append(np.mean(ets_qubo))
            ets_qubo_errors.append(np.std(ets_qubo) / np.sqrt(len(ets_qubo)))

        plot_ebr_vs_tts_for_size(
            sat_size=sat_size,
            ebrs=ebrs,
            pubo_means=tts_pubo_means,
            pubo_errors=tts_pubo_errors,
            qubo_means=tts_qubo_means,
            qubo_errors=tts_qubo_errors,
        )
