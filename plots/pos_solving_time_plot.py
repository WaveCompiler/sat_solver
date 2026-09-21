import glob
import re
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pubo_success_eval
import qubo_success_eval
import torch
import utils

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
PUBO_TPI_PU = 2.0e-9
QUBO_TPI_PU = 2.0e-9

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
            dataset_files[sat_size] = valid_files[:10]
        elif mode == "evaluation":
            dataset_files[sat_size] = valid_files[20:100]
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'optimization' or 'evaluation'.")
    return dataset_files

def pubo_subgroup_update_simulated_annealing(encode, device, steps, start_temp, end_temp, group_slice, log_interval, batch_size):
    num_vars = encode["num_vars"]
    clauses = encode["clauses"]
    spins = torch.randint(0, 2, (batch_size, num_vars), dtype=torch.float32, device=device)
    all_indices = None
    chunks = None
    cooling_ratio = end_temp / start_temp

    solved_steps = torch.full((batch_size,), steps, dtype=torch.long, device=device)
    solved_mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    unsatisfied = utils.count_unsatisfied_clauses(spins, clauses)
    
    initial_zero_mask = (unsatisfied == 0)
    solved_steps[initial_zero_mask] = 0
    solved_mask[initial_zero_mask] = True

    for step in range(steps):
        if (unsatisfied == 0).all():
            break

        progress = step / steps
        decay_factor = cooling_ratio ** progress
        current_temp = start_temp * decay_factor

        if step % group_slice == 0:
            all_indices = torch.randperm(num_vars, device=device)
            chunks = torch.tensor_split(all_indices, group_slice)

        group_idx = step % group_slice
        batch_indices = chunks[group_idx]

        gradient = utils.pubo_compute_gradient(spins, encode)

        scale_noise = torch.rand((batch_size, len(batch_indices)), device=device) * 2 * current_temp
        shift_left = current_temp
        random_noise = scale_noise - shift_left

        grad_sub = gradient[:, batch_indices]
        new_spins_sub = (grad_sub < random_noise).float()

        active_mask = ~solved_mask 
        active_mask_sub = active_mask.unsqueeze(1).expand(-1, len(batch_indices))
        current_spins_sub = spins[:, batch_indices]
        spins[:, batch_indices] = torch.where(active_mask_sub, new_spins_sub, current_spins_sub)
        
        unsatisfied = utils.count_unsatisfied_clauses(spins, clauses)
        
        newly_solved = (unsatisfied == 0) & (~solved_mask)
        solved_steps[newly_solved] = step + 1
        solved_mask[newly_solved] = True

        if step % log_interval == 0 or step == steps - 1:
            utils.print_solver_metrics(step, steps, current_temp, spins, clauses, unsatisfied, solved_mask)

    solved_steps = solved_steps[solved_mask]
    num_satisfied = solved_mask.sum().item()
    return num_satisfied, solved_steps

def qubo_subgroup_update_simulated_annealing(encode, device, steps, start_temp, end_temp, group_slice, log_interval, batch_size):
    total_num_vars = encode["total_num_vars"]
    num_vars = encode["num_vars"]
    clauses = encode["clauses"]
    spins = torch.randint(0, 2, (batch_size, total_num_vars), dtype=torch.float32, device=device)
    all_indices = None
    chunks = None
    cooling_ratio = end_temp / start_temp

    solved_steps = torch.full((batch_size,), steps, dtype=torch.long, device=device)
    solved_mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    unsatisfied = utils.count_unsatisfied_clauses(spins[:, :num_vars], clauses)
    
    initial_zero_mask = (unsatisfied == 0)
    solved_steps[initial_zero_mask] = 0
    solved_mask[initial_zero_mask] = True

    for step in range(steps):
        if (unsatisfied == 0).all():
            break

        progress = step / steps
        decay_factor = cooling_ratio ** progress
        current_temp = start_temp * decay_factor

        if step % group_slice == 0:
            all_indices = torch.randperm(total_num_vars, device=device)
            chunks = torch.tensor_split(all_indices, group_slice)

        group_idx = step % group_slice
        batch_indices = chunks[group_idx]

        gradient = utils.qubo_compute_gradient(spins, encode)

        scale_noise = torch.rand((batch_size, len(batch_indices)), device=device) * 2 * current_temp
        shift_left = current_temp
        random_noise = scale_noise - shift_left

        grad_sub = gradient[:, batch_indices]
        new_spins_sub = (grad_sub < random_noise).float()
        active_mask = ~solved_mask
        active_mask_sub = active_mask.unsqueeze(1).expand(-1, len(batch_indices))
        current_spins_sub = spins[:, batch_indices]
        spins[:, batch_indices] = torch.where(active_mask_sub, new_spins_sub, current_spins_sub)

        unsatisfied = utils.count_unsatisfied_clauses(spins[:, :num_vars], clauses)
        
        newly_solved = (unsatisfied == 0) & (~solved_mask)
        solved_steps[newly_solved] = step + 1
        solved_mask[newly_solved] = True

        if step % log_interval == 0 or step == steps - 1:
            utils.print_solver_metrics(step, steps, current_temp, spins, clauses, unsatisfied, solved_mask)

    solved_steps = solved_steps[solved_mask]
    num_satisfied = solved_mask.sum().item()
    return num_satisfied, solved_steps

if __name__ == "__main__":
    # select mode: "optimization" or "evaluation"
    mode = "optimization"
    # mode = "evaluation"
    selected_dataset = load_sat_instances(mode=mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # x-bar
    pubo_solving_time_arr = []
    qubo_solving_time_arr = []
    for t_step in range(PUBO_STEPS):
        pubo_solving_time_arr.append(t_step * PUBO_TPI_PU)
    for t_step in range(QUBO_STEPS):
        qubo_solving_time_arr.append(t_step * QUBO_TPI_PU)

    for sat_size, files in selected_dataset.items():
        print(f"Size {sat_size} ({mode} mode): loaded {len(files)} files")
        for file in files:
            print(file)

        all_pubo_pos = []
        all_qubo_pos = []

        size_keys = list(DATASET_PATHS.keys())
        size_idx = size_keys.index(sat_size)
        pubo_slice = PUBO_GROUP_SLICE[size_idx]
        qubo_slice = QUBO_GROUP_SLICE[size_idx]
        print(f"pubo_slice: {pubo_slice}")
        print(f"qubo_slice: {qubo_slice}")

        for file_idx, file in enumerate(files):
            print(f"process file: {file}...")
            
            with open(file, "r") as f:
                lines = f.readlines()

            pubo_enc = pubo_success_eval.pubo_encode_sat_prob(lines, device, sigma=PUBO_SIGMA)
            qubo_enc = qubo_success_eval.qubo_encode_sat_prob(lines, device, sigma=QUBO_SIGMA, penalty=QUBO_PENALTY)
            
            pubo_num_satisfied, pubo_solved_steps = pubo_subgroup_update_simulated_annealing(pubo_enc, device, PUBO_STEPS, PUBO_START_TEMP, PUBO_END_TEMP, pubo_slice, PUBO_LOG_INTERVAL, PUBO_BATCH_SIZE)
            print(f"pubo_num_satisfied: {pubo_num_satisfied}")
            print(f"pubo_solved_steps: {pubo_solved_steps}")
            qubo_num_satisfied, qubo_solved_steps = qubo_subgroup_update_simulated_annealing(qubo_enc, device, QUBO_STEPS, QUBO_START_TEMP, QUBO_END_TEMP, qubo_slice, QUBO_LOG_INTERVAL, QUBO_BATCH_SIZE)
            print(f"qubo_num_satisfied: {qubo_num_satisfied}")
            print(f"qubo_solved_steps: {qubo_solved_steps}")

            # y-bar
            pubo_pos_arr = []
            qubo_pos_arr = []
            for t_step in range(PUBO_STEPS):
                pubo_pos_at_t_step = (pubo_solved_steps <= t_step).sum().item() / PUBO_BATCH_SIZE
                pubo_pos_arr.append(pubo_pos_at_t_step)
            for t_step in range(QUBO_STEPS):
                qubo_pos_at_t_step = (qubo_solved_steps <= t_step).sum().item() / QUBO_BATCH_SIZE
                qubo_pos_arr.append(qubo_pos_at_t_step)
            all_pubo_pos.append(pubo_pos_arr)
            all_qubo_pos.append(qubo_pos_arr)

            if file_idx < 3 or file_idx == 9:
                # plot
                fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
                ax.plot(
                    qubo_solving_time_arr[1:],
                    qubo_pos_arr[1:],
                    color="#1f77b4",
                    linestyle="-",
                    linewidth=1.5,
                    marker="o",
                    markersize=3,
                    markeredgecolor="black",
                    markeredgewidth=0.3,
                    label="QUBO-PU",
                    alpha=0.8,
                )
                ax.plot(
                    pubo_solving_time_arr[1:],
                    pubo_pos_arr[1:],
                    color="#ff7f0e",
                    linestyle="-",
                    linewidth=1.5,
                    marker="s",
                    markersize=3,
                    markeredgecolor="black",
                    markeredgewidth=0.3,
                    label="PUBO-PU",
                    alpha=0.8,
                )
        
                ax.set_xscale("log")
                ax.set_xlabel("Solving Time (s) [Log Scale]", fontsize=11)
                ax.set_ylabel("Probability of Success (PoS)", fontsize=11)
                ax.set_title(f"PoS vs. Solving Time (s) for SAT Size {sat_size} ({mode.capitalize()} Mode)", fontsize=12)
                ax.grid(True, which="both", linestyle="--", alpha=0.3)
                ax.legend(loc="best")
        
                plt.tight_layout()
                os.makedirs("./visualizations", exist_ok=True)
                save_path = f"./visualizations/pos_solving_time_plot_sat_size_{sat_size}_file_{file_idx}.png"
                plt.savefig(save_path, dpi=300)
                plt.close(fig)

        avg_pubo_pos = np.mean(all_pubo_pos, axis=0)
        avg_qubo_pos = np.mean(all_qubo_pos, axis=0)

        # plot
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        ax.plot(
            qubo_solving_time_arr[1:],
            avg_qubo_pos[1:],
            color="#1f77b4",
            linestyle="-",
            linewidth=1.5,
            marker="o",
            markersize=3,
            markeredgecolor="black",
            markeredgewidth=0.3,
            label="QUBO-PU (Avg)",
            alpha=0.8,
        )
        ax.plot(
            pubo_solving_time_arr[1:],
            avg_pubo_pos[1:],
            color="#ff7f0e",
            linestyle="-",
            linewidth=1.5,
            marker="s",
            markersize=3,
            markeredgecolor="black",
            markeredgewidth=0.3,
            label="PUBO-PU (Avg)",
            alpha=0.8,
        )

        ax.set_xscale("log")
        ax.set_xlabel("Solving Time (s) [Log Scale]", fontsize=11)
        ax.set_ylabel("Avg Probability of Success (PoS)", fontsize=11)
        ax.set_title(f"Avg PoS vs. Solving Time (s) for SAT Size {sat_size} ({mode.capitalize()} Mode)", fontsize=12)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(loc="best")

        plt.tight_layout()
        os.makedirs("./visualizations", exist_ok=True)
        save_path = f"./visualizations/avg_pos_solving_time_plot_sat_size_{sat_size}.png"
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        