import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import config

# compute
def parse_sat_prob(sat_prob):
    clauses = []
    num_vars = 0
    num_clauses = 0
    for line in sat_prob:
        line = line.strip()
        if not line or line.startswith('c') or line.startswith('%'):
            continue
        if line.startswith('0'):
            break

        if line.startswith('p cnf'):
            parts = line.split()
            num_vars = int(parts[2])
            num_clauses = int(parts[3])
            continue

        clause = line.split()
        x_idxs = []
        for x_idx in clause:
            if x_idx == '0':
                break
            x_idxs.append(int(x_idx))

        if x_idxs:
            clauses.append(x_idxs)

    return num_vars, num_clauses, clauses

def pubo_compute_gradient(spins, encode):
    linear = encode["linear"]
    sym_quadratic = encode["sym_quadratic"]
    sym_cubic = encode["sym_cubic"]

    gradient = linear.unsqueeze(0).expand(spins.shape[0], -1).clone()
    gradient += torch.matmul(spins, sym_quadratic)
    cubic_gradient_term = 'ijk,bj,bk->bi'
    cubic_term = torch.einsum(cubic_gradient_term, sym_cubic, spins, spins)
    gradient += 0.5 * cubic_term
    return gradient

def qubo_compute_gradient(spins, encode):
    linear = encode["linear"]                # Shape: (total_num_vars,)
    sym_quadratic = encode["sym_quadratic"]  # Shape: (total_num_vars, total_num_vars)
    
    gradient = linear.unsqueeze(0).expand(spins.shape[0], -1).clone()
    gradient += torch.matmul(spins, sym_quadratic)
    return gradient

def pubo_compute_energy(spins, encode):
    bias = encode["bias"]
    linear = encode["linear"]
    quadratic = encode["quadratic"]
    cubic = encode["cubic"]
    
    energy = bias + torch.dot(linear, spins)
    energy += torch.sum(quadratic * torch.outer(spins, spins))
    cubic_summation_term = 'i,j,k->ijk'
    energy += torch.sum(cubic * torch.einsum(cubic_summation_term, spins, spins, spins))
    return energy

def qubo_compute_energy(spins, encode):
    bias = encode["bias"]
    linear = encode["linear"]
    quadratic = encode["quadratic"]

    linear_energy = torch.matmul(spins, linear)
    quad_interaction = torch.matmul(spins, quadratic)
    quad_energy = (spins * quad_interaction).sum(dim=1)

    total_energy = bias + linear_energy + quad_energy
    return total_energy

def solution_found(spins, encode):
    clauses = encode["clauses"]
    is_satisfied = count_unsatisfied_clauses(spins, clauses) == 0
    return is_satisfied

def print_solver_metrics(step, current_temp, current_energy, unsatisfied):
    print(f"step {step:4d} | current_temp: {current_temp:.4f} | current_energy: {current_energy:6.2f} | unsatisfied clauses: {unsatisfied:3d}")

def verify_and_print_clauses(spins, clauses):
    spin_list = [int(v) for v in spins.cpu().tolist()]
    
    print("="*80)
    print(f"{'Clause':<8} | {'Conditions':<25} | {'Spins':<25} | {'Status'}")
    print("="*80)
    
    satisfied_count = 0
    total_clauses = len(clauses)
    
    for idx, clause in enumerate(clauses, start=1):
        cond_str = " or ".join([f"x{x}" if x > 0 else f"-x{abs(x)}" for x in clause])
        spins_str = ", ".join([f"x{abs(x)}={spin_list[abs(x)-1]}" for x in clause])
        
        is_satisfied = False
        for x_idx in clause:
            var_idx = abs(x_idx) - 1
            val = spin_list[var_idx]
            if (x_idx > 0 and val == 1) or (x_idx < 0 and val == 0):
                is_satisfied = True
                break
                
        if is_satisfied:
            satisfied_count += 1
            status = "satisfied!"
        else:
            status = "UNSATISFIED"
            
        print(f"{idx:<8} | {cond_str:<25} | {spins_str:<25} | {status}")
        
    print("="*80)
    print(f"Summary: {satisfied_count}/{total_clauses} clauses satisfied.")
    print("="*80)

def count_unsatisfied_clauses(spins, clauses):
    batch_size = spins.shape[0]
    device = spins.device

    unsatisfied_counts = torch.zeros(batch_size, device=device)
    for clause in clauses:
        clause_satisfied = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for idx in clause:
            pos_idx = abs(idx) - 1
            if idx > 0:
                clause_satisfied |= (spins[:, pos_idx] == 1.0)
            else:
                clause_satisfied |= (spins[:, pos_idx] == 0.0)

        unsatisfied_counts += (~clause_satisfied).float()
    return unsatisfied_counts

def get_success_rates(spins, encode, len_clauses):
    clauses = encode["clauses"]
    unsatisfied = count_unsatisfied_clauses(spins, clauses)
    satisfied = len_clauses - unsatisfied
    success_rates = satisfied / len_clauses
    return success_rates

def print_solver_metrics(step, total_steps, current_temp, spins, encode, len_clauses):
    """
    Prints solver execution metrics aggregated across the current batch.
    """
    # Compute batched metrics on GPU
    energies = qubo_compute_energy(spins, encode)
    rates = get_success_rates(spins[:, :encode["num_vars"]], encode, len_clauses)
    unsatisfied = count_unsatisfied_clauses(spins[:, :encode["num_vars"]], encode["clauses"])
    satisfied = len_clauses - unsatisfied

    # Calculate statistics across the batch
    mean_energy = energies.mean().item()
    min_energy = energies.min().item()
    mean_satisfied = satisfied.mean().item()
    max_satisfied = satisfied.max().item()
    success_count = (rates >= config.TARGET_SUCCESS_RATE).sum().item()
    batch_size = spins.shape[0]

    # Format step padding dynamically
    step_fmt = f"{step:>{len(str(total_steps))}}/{total_steps}"

    print(
        f"Step {step_fmt} | Temp: {current_temp:.4f} | "
        f"Sat Clauses (Best/Avg): {int(max_satisfied)}/{mean_satisfied:.1f} of {len_clauses} | "
        f"Energy (Min/Avg): {min_energy:.2f}/{mean_energy:.2f} | "
        f"Solved Runs: {success_count}/{batch_size}"
    )

# graph
def pubo_visualize_1d_linear(encode, output_dir="./visualizations"):
    """
    1D Linear Term: Plots single-variable energy biases.
    """
    os.makedirs(output_dir, exist_ok=True)
    linear = encode["linear"].detach().cpu().numpy()
    num_vars = encode["num_vars"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]})

    # 1. Stem Plot (Spike plot showing magnitude per variable)
    markerline, stemlines, baseline = ax1.stem(
        range(num_vars), linear, linefmt='b-', markerfmt='bo', basefmt='r-'
    )
    plt.setp(markerline, markersize=3)
    plt.setp(stemlines, linewidth=0.8)
    ax1.set_title(f"1D Linear Bias Terms Across {num_vars} Variables", fontsize=14)
    ax1.set_ylabel("Linear Coefficient Weight")
    ax1.set_xlim([0, num_vars])
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Horizontal Heatmap Strip (Quick visual scan)
    sns.heatmap(
        linear.reshape(1, -1), 
        cmap="coolwarm", 
        center=0, 
        cbar=True, 
        ax=ax2, 
        xticklabels=50 if num_vars >= 100 else 10,
        yticklabels=False,
        cbar_kws={'orientation': 'horizontal', 'pad': 0.4, 'label': 'Bias Weight'}
    )
    ax2.set_xlabel("Variable Index i", fontsize=12)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "01_linear_1d_terms.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [Saved 1D Linear Plot] -> {output_path}")

def pubo_visualize_2d_quadratic(encode, output_dir="./visualizations"):
    """
    2D Quadratic Matrix: Plots pair-wise variable coupling weights.
    """
    os.makedirs(output_dir, exist_ok=True)
    sym_quad = encode["sym_quadratic"].detach().cpu().numpy()
    num_vars = encode["num_vars"]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sym_quad, 
        cmap="coolwarm", 
        center=0, 
        cbar_kws={'label': 'Symmetric Quadratic Coupling Weight ($Q_{ij}$)'}
    )
    plt.title(f"2D Symmetric Quadratic Matrix ({num_vars}x{num_vars})", fontsize=14)
    plt.xlabel("Variable Index j", fontsize=12)
    plt.ylabel("Variable Index i", fontsize=12)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "02_quadratic_2d_matrix.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [Saved 2D Quadratic Plot] -> {output_path}")

def pubo_visualize_3d_cubic(encode, output_dir="./visualizations", camera_angles=[(30, 45), (60, 120)]):
    """
    3D Cubic Matrix: Plots full 3-way variable clause interactions in 3D spatial coordinates.
    Generates multiple viewing angles for clear perspective.
    """
    os.makedirs(output_dir, exist_ok=True)
    sym_cubic = encode["sym_cubic"]
    
    # Extract non-zero coordinates (active 3-literal clauses)
    nonzero_indices = torch.nonzero(sym_cubic)
    if nonzero_indices.numel() == 0:
        print("  [Skipped 3D Cubic Plot] Cubic tensor contains no non-zero entries.")
        return

    i_coords = nonzero_indices[:, 0].cpu().numpy()
    j_coords = nonzero_indices[:, 1].cpu().numpy()
    k_coords = nonzero_indices[:, 2].cpu().numpy()
    weights = sym_cubic[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]].cpu().numpy()
    
    num_vars = encode["num_vars"]

    # Generate multi-angle views to prevent 3D occlusion
    for idx, (elev, azim) in enumerate(camera_angles, 1):
        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot where marker size scales with absolute coupling strength
        sc = ax.scatter(
            i_coords, 
            j_coords, 
            k_coords, 
            c=weights, 
            cmap='coolwarm', 
            s=abs(weights) * 20 + 5,  # Variable size for intuitive depth
            alpha=0.5,
            edgecolors='none'
        )

        ax.set_xlim([0, num_vars])
        ax.set_ylim([0, num_vars])
        ax.set_zlim([0, num_vars])

        ax.set_xlabel("Variable i", labelpad=8)
        ax.set_ylabel("Variable j", labelpad=8)
        ax.set_zlabel("Variable k", labelpad=8)
        ax.set_title(f"3D Cubic Tensor Map ({num_vars}x{num_vars}x{num_vars}) - View {idx}", pad=15, fontsize=14)
        
        # Set camera angle
        ax.view_init(elev=elev, azim=azim)

        cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
        cbar.set_label('Cubic Weight ($C^{\\text{sym}}_{ijk}$)')

        output_path = os.path.join(output_dir, f"03_cubic_3d_matrix_view_{idx}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"  [Saved 3D Cubic Plot] -> {output_path}")

def pubo_generate_all_visualizations(encode, output_dir="./visualizations"):
    print(f"generating 1D, 2D, and 3D tensor visualizations in '{output_dir}'...")
    pubo_visualize_1d_linear(encode, output_dir)
    pubo_visualize_2d_quadratic(encode, output_dir)
    pubo_visualize_3d_cubic(encode, output_dir)
    print("all visualizations complete...")

