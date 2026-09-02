import glob
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
    # 100: "/home/taehy/sat/sat_problem_dataset/uf100-430.tar/*.cnf",
    # 150: "/home/taehy/sat/sat_problem_dataset/uf150-645.tar/ai/hoos/Research/SAT/Formulae/UF150.645.100/*.cnf",
    # 200: "/home/taehy/sat/sat_problem_dataset/uf200-860.tar/uf200-860/*.cnf",
}

# Hyperparameters
NUM_FILES_PER_SIZE = 10  # Limit to first 20 instances per problem size
RUNS_PER_INSTANCE = 100
STEPS = 100
SIGMA = 2**-5
START_TEMP = 1.0
END_TEMP = 0.001
LOG_INTERVAL = 10
PENALTY = 1
TARGET_SUCCESS_RATE = 0.99

# Hardware Constants (Replace with your hardware's actual values)
TPI_QUBO_PU = 1.0e-6
TPI_PUBO_PU = 1.5e-6
EPI_QUBO_PU = 1.0e-8
EPI_PUBO_PU = 5.0e-9

# Failure allowance constant for q = 0.99
failure_allowance_level = math.log(1 - TARGET_SUCCESS_RATE)

# Container for final metrics per size
sizes_evaluated = []
tts_qubo_means, tts_qubo_errors = [], []
tts_pubo_means, tts_pubo_errors = [], []
ets_qubo_means, ets_qubo_errors = [], []
ets_pubo_means, ets_pubo_errors = [], []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# Main Loop over Problem Sizes
for sat_size, glob_pattern in DATASET_PATHS.items():
    files = sorted(glob.glob(glob_pattern))[:NUM_FILES_PER_SIZE]

    if not files:
        print(f"Skipping size {sat_size}: No files found matching pattern.")
        continue

    print(f"\n--- Processing Size: {sat_size} ({len(files)} files) ---")
    sizes_evaluated.append(sat_size)

    arr_pubo_its = []
    arr_qubo_its = []

    for file_idx, file in enumerate(files):
        with open(file, "r") as f:
            lines = f.readlines()

        pubo_enc = pubo_success_eval.pubo_encode_sat_prob(
            lines, device, sigma=SIGMA
        )
        qubo_enc = qubo_success_eval.qubo_encode_sat_prob(
            lines, device, sigma=SIGMA, penalty=PENALTY
        )

        pubo_group = pubo_enc["num_vars"] // 2
        qubo_group = qubo_enc["total_num_vars"] // 2

        pubo_success_count = 0
        qubo_success_count = 0

        # Run independent annealing attempts
        for _ in range(RUNS_PER_INSTANCE):
            _, pubo_rate = (
                pubo_success_eval.pubo_subgroup_update_simulated_annealing(
                    pubo_enc,
                    device,
                    STEPS,
                    START_TEMP,
                    END_TEMP,
                    pubo_group,
                    LOG_INTERVAL,
                )
            )
            _, qubo_rate = (
                qubo_success_eval.qubo_subgroup_update_simulated_annealing(
                    qubo_enc,
                    device,
                    STEPS,
                    START_TEMP,
                    END_TEMP,
                    qubo_group,
                    LOG_INTERVAL,
                )
            )

            # Check if all clauses are satisfied (0 unsat clauses)
            if pubo_rate == 0:
                pubo_success_count += 1
            if qubo_rate == 0:
                qubo_success_count += 1

        # Compute Single-Cycle Probabilities
        pos_pubo = pubo_success_count / RUNS_PER_INSTANCE
        pos_qubo = qubo_success_count / RUNS_PER_INSTANCE

        # Calculate PUBO ITS
        if pos_pubo == 1.0:
            its_pubo = STEPS
        elif pos_pubo == 0.0:
            its_pubo = np.nan
        else:
            its_pubo = STEPS * (failure_allowance_level / math.log(1 - pos_pubo))

        # Calculate QUBO ITS
        if pos_qubo == 1.0:
            its_qubo = STEPS
        elif pos_qubo == 0.0:
            its_qubo = np.nan
        else:
            its_qubo = STEPS * (failure_allowance_level / math.log(1 - pos_qubo))

        arr_pubo_its.append(its_pubo)
        arr_qubo_its.append(its_qubo)

        print(
            f"  File [{file_idx+1}/{len(files)}]: "
            f"PUBO PoS={pos_pubo:.2f} (ITS={its_pubo:.1f}) | "
            f"QUBO PoS={pos_qubo:.2f} (ITS={its_qubo:.1f})"
        )

    # Convert ITS to TTS and ETS
    its_pubo_arr = np.array(arr_pubo_its)
    its_qubo_arr = np.array(arr_qubo_its)

    tts_pubo = its_pubo_arr * TPI_PUBO_PU
    tts_qubo = its_qubo_arr * TPI_QUBO_PU
    ets_pubo = its_pubo_arr * EPI_PUBO_PU
    ets_qubo = its_qubo_arr * EPI_QUBO_PU

    # Filter non-finite values for statistical calculation
    v_tts_pubo = tts_pubo[np.isfinite(tts_pubo)]
    v_tts_qubo = tts_qubo[np.isfinite(tts_qubo)]
    v_ets_pubo = ets_pubo[np.isfinite(ets_pubo)]
    v_ets_qubo = ets_qubo[np.isfinite(ets_qubo)]

    # Compute Means and Standard Errors
    tts_pubo_means.append(np.mean(v_tts_pubo))
    tts_pubo_errors.append(np.std(v_tts_pubo) / np.sqrt(len(v_tts_pubo)))
    tts_qubo_means.append(np.mean(v_tts_qubo))
    tts_qubo_errors.append(np.std(v_tts_qubo) / np.sqrt(len(v_tts_qubo)))

    ets_pubo_means.append(np.mean(v_ets_pubo))
    ets_pubo_errors.append(np.std(v_ets_pubo) / np.sqrt(len(v_ets_pubo)))
    ets_qubo_means.append(np.mean(v_ets_qubo))
    ets_qubo_errors.append(np.std(v_ets_qubo) / np.sqrt(len(v_ets_qubo)))

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
plt.savefig("./visualizations/fig3_tts_ets_scaling.png", dpi=300)
plt.show()