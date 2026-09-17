import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import sat_solver.pubo.config as config

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

def print_solver_metrics(step, total_steps, current_temp, spins, clauses, unsatisfied, solved_mask):
    len_clauses = len(clauses)
    mean_satisfied = (len_clauses - unsatisfied).mean().item()
    success_count = solved_mask.sum().item()
    batch_size = spins.shape[0]

    step_fmt = f"{step:>{len(str(total_steps))}}/{total_steps}"
    print(f"Step {step_fmt} | Temp: {current_temp:.4f} | Sat Clauses (Mean/Total): {mean_satisfied:.2f} of {len_clauses} | Solved Runs: {success_count}/{batch_size}")

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

