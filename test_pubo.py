import config
import torch
import utils
from config import PUBO_SIGMA, PUBO_STEPS, PUBO_START_TEMP, PUBO_END_TEMP, PUBO_LOG_INTERVAL, PUBO_BATCH_SIZE, PUBO_GROUP_SLICE

class PUBO:
    def __init__(self, num_vars, num_clauses, clauses, device):
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.clauses = clauses
        self.device = device

    def add_term(self, tensor_pos, tensor_neg, val, indices):
            if val > 0:
                tensor_pos[indices] += val
            elif val < 0:
                tensor_neg[indices] += abs(val)
    
    def encode_sat_prob(self, sigma):
        bias_pos = 0.0
        bias_neg = 0.0
        
        linear_pos = torch.zeros(self.num_vars, dtype=torch.float32, device=self.device)
        linear_neg = torch.zeros(self.num_vars, dtype=torch.float32, device=self.device)
        
        quadratic_pos = torch.zeros((self.num_vars, self.num_vars), dtype=torch.float32, device=self.device)
        quadratic_neg = torch.zeros((self.num_vars, self.num_vars), dtype=torch.float32, device=self.device)
        
        cubic_pos = torch.zeros((self.num_vars, self.num_vars, self.num_vars), dtype=torch.float32, device=self.device)
        cubic_neg = torch.zeros((self.num_vars, self.num_vars, self.num_vars), dtype=torch.float32, device=self.device)

        for clause in self.clauses:
            terms = []
            for var in clause:
                idx = abs(var) - 1
                if var > 0:
                    terms.append((idx, 1.0, -1.0))
                else:
                    terms.append((idx, 0.0, 1.0))

            terms.sort(key=lambda item: item[0])

            (i, c1, x1), (j, c2, x2), (k, c3, x3) = terms
            
            # bias
            b_val = c1 * c2 * c3
            if b_val > 0:
                bias_pos += b_val
            elif b_val < 0:
                bias_neg += abs(b_val)

            # linear terms
            self.add_term(linear_pos, linear_neg, x1 * c2 * c3, i)
            self.add_term(linear_pos, linear_neg, c1 * x2 * c3, j)
            self.add_term(linear_pos, linear_neg, c1 * c2 * x3, k)

            # quadratic terms
            self.add_term(quadratic_pos, quadratic_neg, x1 * x2 * c3, (i, j))
            self.add_term(quadratic_pos, quadratic_neg, x1 * c2 * x3, (i, k))
            self.add_term(quadratic_pos, quadratic_neg, c1 * x2 * x3, (j, k))

            # cubic terms
            self.add_term(cubic_pos, cubic_neg, x1 * x2 * x3, (i, j, k))

        # Pre-symmetrize positive and negative quadratic/quadratic & cubic tensors independently
        sym_quadratic_pos = quadratic_pos + quadratic_pos.T
        sym_quadratic_neg = quadratic_neg + quadratic_neg.T

        sym_cubic_pos = cubic_pos + cubic_pos.permute(1, 0, 2) + cubic_pos.permute(2, 1, 0) + cubic_pos.permute(0, 2, 1) + cubic_pos.permute(1, 2, 0) + cubic_pos.permute(2, 0, 1)
        sym_cubic_neg = cubic_neg + cubic_neg.permute(1, 0, 2) + cubic_neg.permute(2, 1, 0) + cubic_neg.permute(0, 2, 1) + cubic_neg.permute(1, 2, 0) + cubic_neg.permute(2, 0, 1)

        # modeling non-symmetrical physical hardware (apply noise independently if sigma > 0)
        if sigma:
            for t_pos, t_neg in [(linear_pos, linear_neg), (sym_quadratic_pos, sym_quadratic_neg), (sym_cubic_pos, sym_cubic_neg)]:
                mask_pos = (t_pos != 0)
                t_pos += torch.randn_like(t_pos) * sigma * mask_pos
                mask_neg = (t_neg != 0)
                t_neg += torch.randn_like(t_neg) * sigma * mask_neg
        
        return {
            "bias_pos": bias_pos,
            "bias_neg": bias_neg,
            "linear_pos": linear_pos,
            "linear_neg": linear_neg,
            "sym_quadratic_pos": sym_quadratic_pos,
            "sym_quadratic_neg": sym_quadratic_neg,
            "sym_cubic_pos": sym_cubic_pos,
            "sym_cubic_neg": sym_cubic_neg,
            "num_vars": self.num_vars,
            "num_clauses": self.num_clauses,
            "clauses": self.clauses,
            "device": self.device
        }
    
    def compute_gradient(self, spins, encode):
        # gradient = score(x) = make(x) - break(x)
        # grad_pos = make(x), score contribution of s_i = 1.
        # grad_neg = break(x), score withdawal of s_i = 1 => score contribution of s_i = 0.
        # !!! score, make, and break are not exact sat/unsat counts. they are contributions from positive and negative coefficients. !!!
        # just named in this way for simplicity
        
        # compute make gradient (from positive terms)
        grad_pos = encode["linear_pos"].clone()
        grad_pos += torch.matmul(spins, encode["sym_quadratic_pos"])
        cubic_gradient_term = "ijk,j,k->i"
        cubic_term_pos = torch.einsum(cubic_gradient_term, encode["sym_cubic_pos"], spins, spins)
        grad_pos += 0.5 * cubic_term_pos

        # compute break gradient (from negative terms)
        grad_neg = encode["linear_neg"].clone()
        grad_neg += torch.matmul(spins, encode["sym_quadratic_neg"])
        cubic_term_neg = torch.einsum(cubic_gradient_term, encode["sym_cubic_neg"], spins, spins)
        grad_neg += 0.5 * cubic_term_neg

        gradient = grad_pos - grad_neg
        return gradient, grad_pos, grad_neg

    def count_unsatisfied_clauses(self, spins):
        unsatisfied_count = 0
        for clause in self.clauses:
            clause_satisfied = False
            for var in clause:
                idx = abs(var) - 1
                if var > 0:
                    if spins[idx] == 1.0:
                        clause_satisfied = True
                        break
                else:
                    if spins[idx] == 0.0:
                        clause_satisfied = True
                        break
            
            if not clause_satisfied:
                unsatisfied_count += 1
        return unsatisfied_count

    def spins_to_vars(self, spins):
        spins_np = spins.cpu().numpy()
        solution_vars = {}
        for idx, val in enumerate(spins_np):
            var_idx = idx + 1
            bool_val = bool(val == 1.0)
            solution_vars[var_idx] = bool_val
        return solution_vars

    def print_vertical_table(self, gradient, grad_pos, grad_neg, random_noise, sub_indices, old_spins, spins):
        print(f"{'Idx':<5} | {'Gradient':<10} | {'Noise':<10} | {'Grad Pos':<10} | {'Grad Neg':<10} | {'Old Spin':<5} | {'Spin':<5} | Status")
        print("-" * 85)
        sub_list = sub_indices.detach().cpu().tolist()
        noise_map = {global_idx: local_i for local_i, global_idx in enumerate(sub_list)}
        
        for i in range(len(gradient)):
            if i in noise_map:
                local_i = noise_map[i]
                noise_val = random_noise[local_i].item()
            else:
                noise_val = 0.0
                
            old_val = int(old_spins[i].item())
            new_val = int(spins[i].item())
            status = "changed" if old_val != new_val else ""
            print(f"{i:<5} | {gradient[i].item():10.3f} | {noise_val:10.3f} | {grad_pos[i].item():10.3f} | {grad_neg[i].item():10.3f} | {old_val:8d} | {new_val:5d} | {status}")
        print("=" * 85)

    def compute_activity_and_balance(self, grad_pos, grad_neg):
        eps = 1e-8
        pos_activity = grad_pos.clamp_min(0.0)
        neg_activity = grad_neg.clamp_min(0.0)
        activity = pos_activity + neg_activity
        overlap = torch.minimum(pos_activity, neg_activity)
        balance = 2.0 * overlap / (activity + eps)
        return activity, balance

    def get_escape_idx(self, activity, balance):
        weights = activity * balance
        weight_sum = weights.sum().item()

        probs = 0
        if weight_sum > 0.0:
            probs = weights / weight_sum
        else:
            probs = torch.ones_like(weights) / self.num_vars

        escape_idx = torch.multinomial(probs, 1).item()
        return escape_idx

    def solve_simulated_annealing(self, max_steps, max_tries, start_temp, end_temp, group_slice, sigma):
        status = False
        encode = self.encode_sat_prob(sigma)
        cooling_ratio = end_temp / start_temp

        for try_idx in range(max_tries):
            spins = torch.randint(0, 2, (self.num_vars,), dtype=torch.float32, device=self.device)
            all_indices = None
            chunks = None

            for step in range(max_steps):
                unsatisfied = self.count_unsatisfied_clauses(spins)

                if unsatisfied == 0:
                    status = True
                    vars = self.spins_to_vars(spins)
                    return status, vars, step, try_idx

                # simulated annealing within [t, t)
                progress = step / max_steps
                decay_factor = cooling_ratio**progress
                current_temp = start_temp * decay_factor

                if step % group_slice == 0:
                    all_indices = torch.randperm(self.num_vars, device=self.device)
                    chunks = torch.tensor_split(all_indices, group_slice)

                group_idx = step % group_slice
                sub_indices = chunks[group_idx]
                scale_noise = (torch.rand(len(sub_indices), device=self.device) * 2 * current_temp)
                random_noise = scale_noise - current_temp

                gradient, grad_pos, grad_neg = self.compute_gradient(spins, encode)
                grad_sub = gradient[sub_indices]

                old_spins = spins.clone()
                spins[sub_indices] = (grad_sub < random_noise).float()

                print(f"sub_indices:", sub_indices)
                self.print_vertical_table(gradient, grad_pos, grad_neg, random_noise, sub_indices, old_spins, spins)
                print(f"Try {try_idx + 1:2d} | Step {step + 1:3d} | Unsatisfied: {unsatisfied}/{self.num_clauses} clauses")
                print()

        vars = self.spins_to_vars(spins)
        return status, vars, max_steps, max_tries

    def solve_escape_walk(self, max_steps, max_tries, start_temp, end_temp, group_slice, sigma, escape_prob):
        status = False
        encode = self.encode_sat_prob(sigma)
        cooling_ratio = end_temp / start_temp

        for try_idx in range(max_tries):
            spins = torch.randint(0, 2, (self.num_vars,), dtype=torch.float32, device=self.device)
            all_indices = None
            chunks = None

            for step in range(max_steps):
                unsatisfied = self.count_unsatisfied_clauses(spins)

                if unsatisfied == 0:
                    status = True
                    vars = self.spins_to_vars(spins)
                    return status, vars, step, try_idx

                # simulated annealing within [t, t)
                progress = step / max_steps
                decay_factor = cooling_ratio**progress
                current_temp = start_temp * decay_factor

                if step % group_slice == 0:
                    all_indices = torch.randperm(self.num_vars, device=self.device)
                    chunks = torch.tensor_split(all_indices, group_slice)

                group_idx = step % group_slice
                sub_indices = chunks[group_idx]
                scale_noise = (torch.rand(len(sub_indices), device=self.device) * 2 * current_temp)
                random_noise = scale_noise - current_temp

                # gradient = score(x) = make(x) - break(x)
                # grad_pos = make(x), score contribution of s_i = 1.
                # grad_neg = break(x), score withdawal of s_i = 1 => score contribution of s_i = 0.
                # !!! score, make, and break are not exact sat/unsat counts. they are contributions from positive and negative coefficients. !!!
                # just named in this way for simplicity
                gradient, grad_pos, grad_neg = self.compute_gradient(spins, encode)
                old_spins = spins.clone()

                activity, balance = self.compute_activity_and_balance(grad_pos, grad_neg,)
                use_escape_walk = torch.rand((), device=self.device).item() < escape_prob
                if use_escape_walk:
                    escape_idx = self.get_escape_idx(activity, balance)                 
                    spins[escape_idx] = 1.0 - spins[escape_idx]
                else:
                    grad_sub = gradient[sub_indices]
                    spins[sub_indices] = (grad_sub < random_noise).float()

                # self.print_vertical_table(gradient, grad_pos, grad_neg, random_noise, sub_indices, old_spins, spins)
                # print(f"Try {try_idx + 1:2d} | Step {step + 1:3d} | Unsatisfied: {unsatisfied}/{self.num_clauses} clauses")

        vars = self.spins_to_vars(spins)
        return status, vars, max_steps, max_tries

if __name__ == "__main__":
    file_path = config.FILE_PATH 
    sat_prob = open(file_path, "r")
    num_vars, num_clauses, clauses = utils.parse_sat_prob(sat_prob)
    sat_prob.close()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pubo_solver = PUBO(num_vars=num_vars, num_clauses=num_clauses, clauses=clauses, device=device)
    # status, vars_solution, step, try_idx = pubo_solver.solve_simulated_annealing(
    #     max_steps=PUBO_STEPS,
    #     max_tries=1,
    #     start_temp=PUBO_START_TEMP,
    #     end_temp=PUBO_END_TEMP,
    #     group_slice=PUBO_GROUP_SLICE,
    #     sigma=PUBO_SIGMA
    # )
    status, vars_solution, step, try_idx = pubo_solver.solve_escape_walk(
        max_steps=PUBO_STEPS,
        max_tries=1,
        start_temp=PUBO_START_TEMP,
        end_temp=PUBO_END_TEMP,
        group_slice=PUBO_GROUP_SLICE,
        sigma=PUBO_SIGMA,
        escape_prob=0.5
    )

    print(f"status:", status)
    print(f"vars:", vars_solution)
    print(f"step:", step)
    print(f"try_idx:", try_idx)