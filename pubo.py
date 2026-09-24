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

    def encode_sat_prob(self, sigma):
        bias = 0.0
        linear = torch.zeros(self.num_vars, dtype=torch.float32, device=self.device)
        quadratic = torch.zeros((self.num_vars, self.num_vars), dtype=torch.float32, device=self.device)
        cubic = torch.zeros((self.num_vars, self.num_vars, self.num_vars), dtype=torch.float32, device=self.device)

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
            "num_vars": self.num_vars,
            "num_clauses": self.num_clauses,
            "clauses": self.clauses,
            "device": self.device
        }
    
    def compute_gradient(self, spins, encode):
        linear = encode["linear"]
        sym_quadratic = encode["sym_quadratic"]
        sym_cubic = encode["sym_cubic"]

        gradient = linear.clone()
        gradient += torch.matmul(spins, sym_quadratic)
        cubic_gradient_term = "ijk,j,k->i"
        cubic_term = torch.einsum(cubic_gradient_term, sym_cubic, spins, spins)
        gradient += 0.5 * cubic_term
        return gradient

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

                gradient = self.compute_gradient(spins, encode)
                grad_sub = gradient[sub_indices]
                spins[sub_indices] = (grad_sub < random_noise).float()

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
    status, vars_solution, step, try_idx = pubo_solver.solve_simulated_annealing(
        max_steps=PUBO_STEPS,
        max_tries=1,
        start_temp=PUBO_START_TEMP,
        end_temp=PUBO_END_TEMP,
        group_slice=PUBO_GROUP_SLICE,
        sigma=PUBO_SIGMA)

    print(f"status:", status)
    print(f"vars:", vars_solution)
    print(f"step:", step)
    print(f"try_idx:", try_idx)