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

def pubo_subgroup_update_simulated_annealing(encode, device, steps, start_temp, end_temp, group, log_interval):
    print("-"*80)
    num_vars = encode["num_vars"]
    clauses = encode["clauses"]
    len_clauses = len(clauses)
        
    spins = torch.randint(0, 2, (num_vars,), dtype=torch.float32, device=device)
    
    for step in range(steps):
        # temperature cooling process
        cooling_ratio = end_temp / start_temp
        progress = step / steps
        decay_factor = cooling_ratio ** progress
        current_temp = start_temp * decay_factor

        # choose batch indices
        batch_indices = None
        if step % 2 == 0:
            # permutation 2-step group(batch_indices) selection strategy
            all_indices = torch.randperm(num_vars, device=device)
            batch_indices = all_indices[:group]
            complement_indices = all_indices[group:]
        else:
            batch_indices = complement_indices
            
        gradient = utils.pubo_compute_gradient(spins, encode)

        # add randomness for gradient descent
        # binary gradient theory & intuition (for noise construction & comparison with the actual gradient)
        # 1. why math works: E(s) is linear in s_i since s_i^2 = s_i (binary).
        #    E(s_i) = A * s_i + B  =>  dE/ds_i = A <= achieved by gradient descent. actual gradient descent.
        #    E(s_i=1) - E(s_i=0) = (A + B) - B = A <= achieved by mathmatical intuition. we add randomness here to make noise.
        #    thus, grad_i = dE/ds_i = E(s_i=1) - E(s_i=0) = A
        #
        # 2. decision logic:
        #    grad < 0  =>  E(s_i=1) < E(s_i=0)  =>  turning on LOWERS energy  => target s_i = 1
        #    grad > 0  =>  E(s_i=1) > E(s_i=0)  =>  turning ON RAISES energy  => target s_i = 0
        #
        # 3. update rule:
        #    (grad_i < noise) sets s_i = 1 when grad is negative, s_i = 0 when grad is positive.
        scale_noise = torch.rand(len(batch_indices), device=device) * 2 * current_temp
        shift_left = current_temp
        random_noise = scale_noise - shift_left
        spins[batch_indices] = (gradient[batch_indices] < random_noise).float()

        if step % log_interval == 0 or step == steps - 1:
            current_energy = utils.pubo_compute_energy(spins, encode).item()
            unsatisfied = utils.count_unsatisfied_clauses(spins, clauses)
            utils.print_solver_metrics(step, current_temp, current_energy, unsatisfied)

        rate = utils.get_success_rate(spins, encode, len_clauses)
        if rate >= config.TARGET_SUCESS_RATE:
            current_energy = utils.pubo_compute_energy(spins, encode).item()
            unsatisfied = utils.count_unsatisfied_clauses(spins, clauses)
            satisfied = len_clauses - unsatisfied
            utils.print_solver_metrics(step, current_temp, current_energy, unsatisfied)
            print(f"solution found at step: {step} | rate: {rate} | satisfied/total {satisfied}/{len_clauses}")
            break

    print("-"*80)
    
    return spins, rate

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
    print(f"steps: {steps}")
    print(f"start_temp: {start_temp}")
    print(f"end_temp: {end_temp}")
    print(f"group: {group}")
    print(f"log_interval: {log_interval}")

    count = 0
    spins = None
    success_rate = 0
    for i in range(100):
        spins, success_rate = pubo_subgroup_update_simulated_annealing(encode, device, steps, start_temp, end_temp, group, log_interval)
        if success_rate >= config.TARGET_SUCESS_RATE:
            count += 1
    print(f"success count: {count}")
    
    print("end decoding...")
    print("="*80)
