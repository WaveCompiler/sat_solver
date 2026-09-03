import torch
import utils
import config

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

def qubo_subgroup_update_simulated_annealing(encode, device, steps, start_temp, end_temp, group, log_interval, batch_size):
    # print("-"*80)
    total_num_vars = encode["total_num_vars"]
    num_vars = encode["num_vars"]
    clauses = encode["clauses"]
    len_clauses = len(clauses)

    spins = torch.randint(0, 2, (batch_size, total_num_vars), dtype=torch.float32, device=device)
    
    for step in range(steps):
        # temperature cooling process
        cooling_ratio = end_temp / start_temp
        progress = step / steps
        decay_factor = cooling_ratio ** progress
        current_temp = start_temp * decay_factor

        # choose batch indices
        if step % 2 == 0:
            all_indices = torch.randperm(total_num_vars, device=device)
            batch_indices = all_indices[:group]
            complement_indices = all_indices[group:]
        else:
            batch_indices = complement_indices

        gradient = utils.qubo_compute_gradient(spins, encode)

        scale_noise = torch.rand((batch_size, len(batch_indices)), device=device) * 2 * current_temp
        shift_left = current_temp
        random_noise = scale_noise - shift_left
        grad_sub = gradient[:, batch_indices]
        spins[:, batch_indices] = (grad_sub < random_noise).float()

        if step % log_interval == 0 or step == steps - 1:
            utils.print_solver_metrics(step, steps, current_temp, spins, encode, len_clauses)

    rates = utils.get_success_rates(spins[:, :num_vars], encode, len_clauses)
    # print("-"*80)
    
    return rates

if __name__ == "__main__":
    print("="*80)
    print("start encoding...")

    file_path = config.FILE_PATH 
    sat_prob = open(file_path, "r")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    #!!! higher sigma for higher randomness !!!
    sigma = config.SIGMA
    print(f"sigma: {sigma:.8f}")

    #!!! higher penalty for higher randomness !!!
    penalty = config.PENALTY
    print(f"penalty: {penalty:.8f}")

    encode = qubo_encode_sat_prob(sat_prob, device, sigma, penalty)
    print(f"num_vars: {encode['num_vars']}")
    print(f"num_clauses: {encode['num_clauses']}")

    print("end encoding...")
    print("="*80)
    print("start decoding...")

    # for smaller problem size
    steps = config.STEPS
    start_temp = config.START_TEMP
    end_temp = config.END_TEMP
    group = encode["total_num_vars"] // 2
    log_interval = config.LOG_INTERVAL
    batch_size = config.BATCH_SIZE
    print(f"steps: {steps}")
    print(f"start_temp: {start_temp}")
    print(f"end_temp: {end_temp}")
    print(f"group: {group}")
    print(f"log_interval: {log_interval}")
    print(f"batch_size: {batch_size}")

    rates = qubo_subgroup_update_simulated_annealing(encode, device, steps, start_temp, end_temp, group, log_interval, batch_size)
    successful_runs = rates >= config.TARGET_SUCCESS_RATE 
    success_count = successful_runs.sum().item()
    print(f"success_count: {success_count}")
    
    print("end decoding...")
    print("="*80)

    # utils.verify_and_print_clauses(spins, encode["clauses"])
    # print(f"re-run count: {count}")
