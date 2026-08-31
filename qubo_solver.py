import torch
import utils

def qubo_encode_sat_prob(sat_prob, device, sigma, penalty):
    num_vars, num_clauses, clauses = utils.parse_sat_prob(sat_prob)
    total_num_vars = num_vars + num_clauses

    bias = 0.0
    linear = torch.zeros(total_num_vars, dtype=torch.float32, device=device)
    quadratic = torch.zeros((total_num_vars, total_num_vars), dtype=torch.float32, device=device)

    for c_idx, clause in enumerate(clauses):
        y_idx = num_vars + c_idx

        terms = []
        for x_idx in clause:
            idx = abs(x_idx) - 1
            if x_idx > 0:
                terms.append((idx, 1.0, -1.0))
            else:
                terms.append((idx, 0.0, 1.0))

        terms.sort(key=lambda item: item[0])

        (pivot_idx, pivot_c, pivot_x), (q_idx, q_c, q_x), (r_idx, r_c, r_x) = terms

        # pivot
        linear[y_idx] += pivot_c
        i_p, i_y = sorted([pivot_idx, y_idx])
        quadratic[i_p, i_y] += pivot_x

        # rosenberg penalty
        # p * (q * r - 2 * q * y - 2 * r * y + 3 * y)

        # p * q * r
        bias += penalty * (q_c * r_c)
        linear[q_idx] += penalty * (q_x * r_c)
        linear[r_idx] += penalty * (q_c * r_x)
        i_q, i_r = sorted([q_idx, r_idx])
        quadratic[i_q, i_r] += penalty * (q_x * r_x)

        # -2 * p * q * y
        linear[y_idx] += -2.0 * penalty * q_c
        i_q, i_y = sorted([q_idx, y_idx])
        quadratic[i_q, i_y] += -2.0 * penalty * q_x

        # -2 * p * r * y
        linear[y_idx] += -2.0 * penalty * r_c
        i_r, i_y = sorted([r_idx, y_idx])
        quadratic[i_r, i_y] += -2.0 * penalty * r_x

        # 3 * p * y
        linear[y_idx] += 3.0 * penalty

    # pre-symmetrize tensors to optimize GPU compute_gradient
    sym_quadratic = quadratic + quadratic.T
    
    # modeling non-symmetrical physical hardware
    # apply programming-error noise (additive gaussian noise ONLY on non-zero entries)
    if sigma:
        linear_mask = (linear != 0)
        linear += torch.randn_like(linear) * sigma * linear_mask
        sym_quadratic_mask = (sym_quadratic != 0)
        sym_quadratic += torch.randn_like(sym_quadratic) * sigma * sym_quadratic_mask
    
    return {
        "bias": bias,
        "linear": linear,
        "quadratic": quadratic,
        "sym_quadratic": sym_quadratic,
        "num_vars": num_vars,
        "num_clauses": num_clauses,
        "total_num_vars": total_num_vars,
        "clauses": clauses
    }

def qubo_subgroup_update_simulated_annealing(encode, device, steps, start_temp, end_temp, group, log_interval):
    print("-"*80)
    total_num_vars = encode["total_num_vars"]
    num_vars = encode["num_vars"]
    clauses = encode["clauses"]
        
    spins = torch.randint(0, 2, (total_num_vars,), dtype=torch.float32, device=device)
    
    for step in range(steps):
        # temperature cooling process
        cooling_ratio = end_temp / start_temp
        progress = step / steps
        decay_factor = cooling_ratio ** progress
        current_temp = start_temp * decay_factor

        # choose batch indices
        if step % 2 == 0:
            # permutation 2-step group(batch_indices) selection strategy
            all_indices = torch.randperm(total_num_vars, device=device)
            batch_indices = all_indices[:group]
        else:
            mask = torch.ones(total_num_vars, dtype=torch.bool, device=device)
            mask[batch_indices] = False
            batch_indices = torch.arange(total_num_vars, device=device)[mask]
            
        gradient = utils.qubo_compute_gradient(spins, encode)

        # add randomness for gradient descent
        scale_noise = torch.rand(len(batch_indices), device=device) * 2 * current_temp
        shift_left = current_temp
        random_noise = scale_noise - shift_left
        spins[batch_indices] = (gradient[batch_indices] < random_noise).float()

        if step % log_interval == 0 or step == steps - 1:
            current_energy = utils.qubo_compute_energy(spins, encode).item()
            unsatisfied = utils.count_unsatisfied_clauses(spins[:num_vars], clauses)
            utils.print_solver_metrics(step, current_temp, current_energy, unsatisfied)

        is_satisfied = utils.solution_found(spins[:num_vars], encode)
        if is_satisfied:
            current_energy = utils.pubo_compute_energy(spins, encode).item()
            unsatisfied = utils.count_unsatisfied_clauses(spins[:num_vars], clauses)
            utils.print_solver_metrics(step, current_temp, current_energy, unsatisfied)
            print(f"solution found at step {step}!")
            break

    print("-"*80)
    
    return spins, is_satisfied

if __name__ == "__main__":
    print("="*80)
    print("start encoding...")

    file_path = "/home/taehy/sat/sat_problem_dataset/uf100-430.tar/uf100-01.cnf"
    # file_path = "/home/taehy/sat/sat_problem_dataset/uuf100-430.tar/UUF100.430.1000/uuf100-01.cnf"
    # file_path = "/home/taehy/sat/sat_problem_dataset/uf150-645.tar/ai/hoos/Research/SAT/Formulae/UF150.645.100/uf150-01.cnf"
    # file_path = "/home/taehy/sat/sat_problem_dataset/uf200-860.tar/uf200-860/uf200-01.cnf"
    # file_path = "/home/taehy/sat/sat_problem_dataset/uf250-1065.tar/uf250-1065/ai/hoos/Shortcuts/UF250.1065.100/uf250-09.cnf"
    sat_prob = open(file_path, "r")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    #!!! higher sigma for higher randomness !!!
    sigma = 2 ** -8
    print(f"sigma: {sigma:.8f}")

    #!!! higher penalty for higher randomness !!!
    penalty = 2
    print(f"penalty: {penalty:.8f}")

    encode = qubo_encode_sat_prob(sat_prob, device, sigma, penalty)
    print(f"num_vars: {encode['num_vars']}")
    print(f"num_clauses: {encode['num_clauses']}")

    print("end encoding...")
    print("="*80)
    print("start decoding...")

    # for smaller problem size
    steps = 2000
    start_temp = 5.0
    end_temp = 0.001
    group = encode["total_num_vars"] // 2
    log_interval = 200

    # for larger problem size
    # steps = 10000
    # start_temp = 5.0
    # end_temp = 0.0001
    # group = encode["total_num_vars"] // 2
    # log_interval = 500

    print(f"steps: {steps}")
    print(f"start_temp: {start_temp}")
    print(f"end_temp: {end_temp}")
    print(f"group: {group}")
    print(f"log_interval: {log_interval}")

    is_satisfied = False
    count = 0
    spins = None
    while not is_satisfied:
        spins, is_satisfied = qubo_subgroup_update_simulated_annealing(encode, device, steps, start_temp, end_temp, group, log_interval)

        if is_satisfied:
            break

        count += 1
    
    print("end decoding...")
    print("="*80)

    utils.verify_and_print_clauses(spins, encode["clauses"])
    print(f"re-run count: {count}")

    print("="*80)
    # utils.pubo_generate_all_visualizations(encode)
    print("="*80)
