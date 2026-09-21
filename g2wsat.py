import torch
import utils
import config

# TODO still in progress

class G2WSAT:
    def __init__(self, num_vars, num_clauses, clauses, device):
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.clauses = clauses
        self.device = torch.device(device)
        
        # Clause matrix representation: Shape (num_clauses, num_vars + 1)
        # Values: +1 for positive literal, -1 for negative literal, 0 for absent
        self.clause_matrix = torch.zeros((self.num_clauses, self.num_vars + 1), dtype=torch.int8, device=self.device)
        self.clause_mask = torch.zeros((self.num_clauses, self.num_vars + 1), dtype=torch.bool, device=self.device)
        
        for c_idx, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit)
                sign = 1 if lit > 0 else -1
                self.clause_matrix[c_idx, var] = sign
                self.clause_mask[c_idx, var] = True

    def _compute_scores(self, assignment, clause_sat, clause_evals):
        """
        Computes score(x) = make(x) - break(x) for all variables in parallel.
        """
        unsat_mask = ~clause_sat  # Shape: (C,)
        
        # Make score: Unsatisfied clauses that become satisfied if variable is flipped
        make_scores = (unsat_mask.unsqueeze(1) & self.clause_mask).sum(dim=0, dtype=torch.float32)
        
        # Break score: Clauses satisfied by ONLY this variable (sat_count == 1)
        critical_clauses = (clause_evals == 1)  # Shape: (C,)
        is_sole_satisfier = (assignment.unsqueeze(0) * self.clause_matrix) > 0
        
        break_scores = (critical_clauses.unsqueeze(1) & is_sole_satisfier).sum(dim=0, dtype=torch.float32)
        
        return make_scores - break_scores

    def solve(self, max_steps=10000, prob_p=0.5, div_prob_dp=0.01):
        V = self.num_vars
        
        # Random initial assignment in {-1, +1}
        assignment = torch.where(
            torch.rand(V + 1, device=self.device) > 0.5,
            torch.tensor(1, dtype=torch.int8, device=self.device),
            torch.tensor(-1, dtype=torch.int8, device=self.device)
        )
        
        # Track last step each variable was flipped
        last_flip_time = torch.full((V + 1,), -1, dtype=torch.long, device=self.device)
        
        # Promising decreasing variable set
        promising_set = torch.zeros(V + 1, dtype=torch.bool, device=self.device)

        for step in range(max_steps):
            # Evaluate clauses in parallel
            clause_evals = torch.matmul(self.clause_matrix.float(), assignment.float())
            clause_sat = clause_evals > 0
            
            # Return early if solution is found
            if torch.all(clause_sat):
                solution = (assignment[1:] > 0).cpu().tolist()
                return True, solution, step, 0

            # Compute scores for all variables simultaneously
            scores = self._compute_scores(assignment, clause_sat, clause_evals)
            
            # Filter promising set: variables with positive score
            is_decreasing = scores > 0
            promising_set = promising_set & is_decreasing
            
            promising_indices = torch.where(promising_set[1:])[0] + 1

            # BRANCH 1: Greedy Step on Promising Decreasing Variable Set
            if len(promising_indices) > 0:
                p_scores = scores[promising_indices]
                max_score = torch.max(p_scores)
                candidates = promising_indices[p_scores == max_score]
                
                if len(candidates) == 1:
                    selected_var = candidates[0].item()
                else:
                    # Tie-breaking by least recently flipped
                    flip_times = last_flip_time[candidates]
                    min_idx = torch.argmin(flip_times)
                    selected_var = candidates[min_idx].item()

            # BRANCH 2: Fallback WalkSAT + Novelty++ Step
            else:
                unsat_clauses = torch.where(~clause_sat)[0]
                rand_c = unsat_clauses[torch.randint(0, len(unsat_clauses), (1,)).item()]
                c_vars = torch.where(self.clause_mask[rand_c])[0]
                
                # Novelty++ Diversification Step
                if torch.rand(1).item() < div_prob_dp:
                    flip_times = last_flip_time[c_vars]
                    selected_var = c_vars[torch.argmin(flip_times)].item()
                # Standard Novelty Step
                else:
                    c_scores = scores[c_vars]
                    sorted_order = torch.argsort(c_scores, descending=True)
                    sorted_vars = c_vars[sorted_order]
                    
                    best_var = sorted_vars[0]
                    second_best_var = sorted_vars[1] if len(sorted_vars) > 1 else best_var
                    
                    flip_times = last_flip_time[c_vars]
                    most_recent_var = c_vars[torch.argmax(flip_times)]
                    
                    if best_var != most_recent_var:
                        selected_var = best_var.item()
                    else:
                        if torch.rand(1).item() < prob_p:
                            selected_var = second_best_var.item()
                        else:
                            selected_var = best_var.item()

            # Execute flip
            assignment[selected_var] *= -1
            last_flip_time[selected_var] = step

            # Update promising set with new decreasing variables
            new_clause_evals = torch.matmul(self.clause_matrix.float(), assignment.float())
            new_clause_sat = new_clause_evals > 0
            new_scores = self._compute_scores(assignment, new_clause_sat, new_clause_evals)
            
            promising_set = promising_set | (new_scores > 0)
            promising_set[selected_var] = False  # Avoid immediately flipping back

        return False, [], max_steps, 0


if __name__ == "__main__":
    file_path = config.FILE_PATH 
    sat_prob = open(file_path, "r")
    num_vars, num_clauses, clauses = utils.parse_sat_prob(sat_prob)
    sat_prob.close()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g2wsat_solver = G2WSAT(num_vars=num_vars, num_clauses=num_clauses, clauses=clauses, device=device)
    status, vars_solution, step, try_idx = g2wsat_solver.solve(max_steps=1000, prob_p=0.5, div_prob_dp=0.01)
    
    print(f"status:", status)
    print(f"vars:", vars_solution)
    print(f"step:", step)
    print(f"try_idx:", try_idx)