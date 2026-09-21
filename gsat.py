import random

import config
import utils

class GSAT:
    def __init__(self, num_vars, num_clauses, clauses):
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.clauses = clauses
    
    def is_clause_satisfied(self, clause, vars):
        for var in clause:
            if var > 0:
                is_true = vars[var]
            else:
                var_idx = abs(var)
                is_true = not vars[var_idx]
            
            if is_true:
                return True

        return False

    def is_satisfied(self, vars):
        for clause in self.clauses:
            clause_satisfied = self.is_clause_satisfied(clause, vars)
            if not clause_satisfied:
                return False
        return True
    
    def count_satisfied_clauses(self, vars):
        num_satisfied_clauses = 0
        for clause in self.clauses:
            clause_satisfied = self.is_clause_satisfied(clause, vars)
            if clause_satisfied:
                num_satisfied_clauses += 1

        return num_satisfied_clauses

    def solve(self, max_steps, max_tries):
        choices = [True, False]
        status = False

        for try_idx in range(max_tries):
            # random vars generation
            vars = {}
            for var in range(1, self.num_vars + 1):
                vars[var] = random.choice(choices)

            for step in range(max_steps):
                if self.is_satisfied(vars):
                    status = True
                    return status, vars, step, try_idx

                max_satisfied_clauses = -1
                candidate_vars = []

                # evaluate net score change for each variable flip
                for var in range(1, self.num_vars + 1):
                    vars[var] = not vars[var]
                    num_satisfied_clauses = self.count_satisfied_clauses(vars)
                    
                    if num_satisfied_clauses > max_satisfied_clauses:
                        max_satisfied_clauses = num_satisfied_clauses
                        candidate_vars = [var]
                    elif num_satisfied_clauses == max_satisfied_clauses:
                        candidate_vars.append(var)

                    vars[var] = not vars[var]

                # select one var randomlmy from candidate vars to flip
                selected_var = random.choice(candidate_vars)
                vars[selected_var] = not vars[selected_var]

                # print(
                #     f"Try {try_idx + 1:2d} | Step {step + 1:3d} | "
                #     f"Flipped var x{selected_var:<2d} -> {vars[selected_var]!s:<5s} | "
                #     f"Satisfied: {max_satisfied_clauses}/{self.num_clauses} clauses "
                #     f"(Candidates: {candidate_vars})"
                # )

        return status, vars, max_steps, max_tries

if __name__ == "__main__":
    file_path = config.FILE_PATH 
    sat_prob = open(file_path, "r")
    num_vars, num_clauses, clauses = utils.parse_sat_prob(sat_prob)
    sat_prob.close()

    gsat_solver = GSAT(num_vars=num_vars, num_clauses=num_clauses, clauses=clauses)
    status, vars, step, try_idx = gsat_solver.solve(max_tries=10, max_steps=1000)
    print(f"status:", status)
    print(f"vars:", vars)
    print(f"step:", step)
    print(f"try_idx:", try_idx)