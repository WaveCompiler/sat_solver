import torch
import config
import utils

def pubo_encode_sat_prob(sat_prob, device, sigma):
    num_vars, num_clauses, clauses = utils.parse_sat_prob(sat_prob)

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

    # modeling non-symmetrical physical hardware
    # apply programming-error noise (additive gaussian noise ONLY on non-zero entries)
    if sigma:
        linear_mask = (linear != 0)
        linear += torch.randn_like(linear) * sigma * linear_mask
        sym_quadratic_mask = (sym_quadratic != 0)
        sym_quadratic += torch.randn_like(sym_quadratic) * sigma * sym_quadratic_mask
        sym_cubic_mask = (sym_cubic != 0)
        sym_cubic += torch.randn_like(sym_cubic) * sigma * sym_cubic_mask
    
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
        solved_steps[newly_solved] = step
        solved_mask[newly_solved] = True

        if step % log_interval == 0 or step == steps - 1:
            utils.print_solver_metrics(step, steps, current_temp, spins, clauses, unsatisfied, solved_mask)

    if solved_mask.any():
        mean_solved_steps = solved_steps[solved_mask].float().mean().item()
    else:
        mean_solved_steps = float(steps)

    num_satisfied = solved_mask.sum().item()
    return num_satisfied, mean_solved_steps

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

    encode = pubo_encode_sat_prob(sat_prob, device, sigma)
    print(f"num_vars: {encode['num_vars']}")
    print(f"num_clauses: {encode['num_clauses']}")

    print("end encoding...")
    print("="*80)
    print("start decoding...")

    # for smaller problem size
    steps = config.STEPS
    start_temp = config.START_TEMP
    end_temp = config.END_TEMP
    group = encode["num_vars"] // 2
    log_interval = config.LOG_INTERVAL
    batch_size = config.BATCH_SIZE
    print(f"steps: {steps}")
    print(f"start_temp: {start_temp}")
    print(f"end_temp: {end_temp}")
    print(f"group: {group}")
    print(f"log_interval: {log_interval}")
    print(f"batch_size: {batch_size}")

    rates = pubo_subgroup_update_simulated_annealing(encode, device, steps, start_temp, end_temp, group, log_interval, batch_size)
    successful_runs = rates >= config.TARGET_SUCCESS_RATE 
    success_count = successful_runs.sum().item()
    print(f"success_count: {success_count}")
    
    print("end decoding...")
    print("="*80)
