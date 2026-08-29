import torch
import pubo_util as pu

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

def encode_sat_prob(sat_prob, device):
    num_vars, num_clauses, clauses = parse_sat_prob(sat_prob)

    bias = 0.0
    linear = torch.zeros(num_vars, dtype=torch.float32, device=device)
    quadratic = torch.zeros((num_vars, num_vars), dtype=torch.float32, device=device)
    cubic = torch.zeros((num_vars, num_vars, num_vars), dtype=torch.float32, device=device)

    for clause in clauses:
        terms = []
        for x_idx in clause:
            idx = abs(x_idx) - 1
            if x_idx > 0:
                terms.append((idx, 1.0, -1.0))
            else:
                terms.append((idx, 0.0, 1.0))

        terms.sort(key=lambda item: item[0])

        (i, c1, x1), (j, c2, x2), (k, c3, x3) = terms
        bias += c1 * c2 * c3
        linear[i] += x1 * c2 * c3
        linear[j] += c1 * x2 * c3
        linear[k] += c1 * c2 * x3
        quadratic[i, j] += x1 * x2 * c3
        quadratic[i, k] += x1 * c2 * x3
        quadratic[j, k] += c1 * x2 * x3
        cubic[i, j, k] += x1 * x2 * x3

    # pre-symmetrize tensors to optimize GPU compute_gradient
    sym_quadratic = quadratic + quadratic.T
    sym_cubic = cubic + cubic.permute(1, 0, 2) + cubic.permute(2, 1, 0) + cubic.permute(0, 2, 1) + cubic.permute(1, 2, 0) + cubic.permute(2, 0, 1)

    return {
        "bias": bias,
        "linear": linear,
        "quadratic": quadratic,
        "cubic": cubic,
        "sym_quadratic": sym_quadratic,
        "sym_cubic": sym_cubic,
        "num_vars": num_vars,
        "num_clauses": num_clauses,
        "clauses": clauses
    }

def solution_found(s, encode):
    return pu.compute_energy(s, encode).item() == 0.0

def compute_gradient(spins, encode):
    linear = encode["linear"]
    sym_quadratic = encode["sym_quadratic"]
    sym_cubic = encode["sym_cubic"]
    
    gradient = linear.clone()
    gradient += torch.matmul(sym_quadratic, spins)
    cubic_gradient_term = 'ijk,j,k->i'
    symmetry_factor = 2.0  # accounts for symmetric derivatives of 3-variable terms
    gradient += torch.einsum(cubic_gradient_term, sym_cubic, spins, spins) / symmetry_factor
    return gradient

def subgroup_update_simulated_annealing(encode, device, steps, start_temp, end_temp, group, log_interval):
    print("-"*80)
    num_vars = encode["num_vars"]
    clauses = encode["clauses"]
        
    spins = torch.randint(0, 2, (num_vars,), dtype=torch.float32, device=device)
    
    for step in range(steps):
        # temperature cooling process
        cooling_ratio = end_temp / start_temp
        progress = step / steps
        decay_factor = cooling_ratio ** progress
        current_temp = start_temp * decay_factor

        # choose batch indices
        if step % 2 == 0:
            # permutation 2-step group(batch_indices) selection strategy
            all_indices = torch.randperm(num_vars, device=device)
            batch_indices = all_indices[:group]
        else:
            mask = torch.ones(num_vars, dtype=torch.bool, device=device)
            mask[batch_indices] = False
            batch_indices = torch.arange(num_vars, device=device)[mask]
            
        gradient = compute_gradient(spins, encode)

        # add randomness for gradient descent
        scale_noise = torch.rand(len(batch_indices), device=device) * 2 * current_temp
        shift_left = current_temp
        random_noise = scale_noise - shift_left
        spins[batch_indices] = (gradient[batch_indices] < random_noise).float()

        if step % log_interval == 0 or step == steps - 1:
            current_energy = pu.compute_energy(spins, encode).item()
            unsatisfied = pu.count_unsatisfied_clauses(spins, clauses)
            pu.print_solver_metrics(step, current_temp, current_energy, unsatisfied)

        is_satisfied = solution_found(spins, encode)
        if is_satisfied:
            current_energy = pu.compute_energy(spins, encode).item()
            unsatisfied = pu.count_unsatisfied_clauses(spins, clauses)
            pu.print_solver_metrics(step, current_temp, current_energy, unsatisfied)
            print(f"solution found at step {step}!")
            break

    print("-"*80)
    
    return spins, is_satisfied

if __name__ == "__main__":
    print("="*80)
    print("start encoding...")

    # file_path = "/home/taehy/sat/sat_problem_dataset/uf100-430.tar/uf100-01.cnf"
    # file_path = "/home/taehy/sat/sat_problem_dataset/uuf100-430.tar/UUF100.430.1000/uuf100-01.cnf"
    # file_path = "/home/taehy/sat/sat_problem_dataset/uf150-645.tar/ai/hoos/Research/SAT/Formulae/UF150.645.100/uf150-01.cnf"
    # file_path = "/home/taehy/sat/sat_problem_dataset/uf200-860.tar/uf200-860/uf200-01.cnf"
    file_path = "/home/taehy/sat/sat_problem_dataset/uf250-1065.tar/uf250-1065/ai/hoos/Shortcuts/UF250.1065.100/uf250-08.cnf"
    sat_prob = open(file_path, "r")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    encode = encode_sat_prob(sat_prob, device)
    print(f"num_vars: {encode['num_vars']}")
    print(f"num_clauses: {encode['num_clauses']}")

    print("end encoding...")
    print("="*80)
    print("start decoding...")

    # for smaller problem size
    # steps = 2000
    # start_temp = 1.0
    # end_temp = 0.001
    # group = encode["num_vars"] // 2
    # log_interval = 200

    # for larger problem size
    steps = 10000
    start_temp = 5.0
    end_temp = 0.0001
    group = encode["num_vars"] // 2
    log_interval = 500

    is_satisfied = False
    count = 0
    spins = None
    while not is_satisfied:
        spins, is_satisfied = subgroup_update_simulated_annealing(encode, device, steps, start_temp, end_temp, group, log_interval)

        if is_satisfied:
            break

        count += 1
    
    print("end decoding...")
    print("="*80)

    pu.verify_and_print_clauses(spins, encode["clauses"])
    print(f"re-run count: {count}")

    print("="*80)
    pu.generate_all_visualizations(encode)
    print("="*80)