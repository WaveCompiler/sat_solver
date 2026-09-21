import random
import config
import utils

class WalkSAT:
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

    def get_unsatisfied_clauses(self, vars):
        unsatisfied_clauses = []
        for clause in self.clauses:
            clause_satisfied = self.is_clause_satisfied(clause, vars)
            if not clause_satisfied:
                unsatisfied_clauses.append(clause)
        return unsatisfied_clauses

    def count_unsatisfied_clauses(self, vars):
        unsatisfied_clauses = self.get_unsatisfied_clauses(vars)
        num_unsatisfied_clauses = len(unsatisfied_clauses)
        return num_unsatisfied_clauses
    
    def flip_and_count_unsatisfied_clauses(self, vars, var):
        vars[var] = not vars[var]
        num_unsatisfied_clauses = self.count_unsatisfied_clauses(vars)
        vars[var] = not vars[var]
        return num_unsatisfied_clauses

    def sort_clause_vars(self, clause_vars, vars, last_flip_time):
        # favor
        # 1. most satisfying clauses variable 
        # 2. least recently flipped variable

        decorated_vars = []
        for var in clause_vars:
            unsat_count = self.flip_and_count_unsatisfied_clauses(vars, var)
            flip_time = last_flip_time.get(var, -1) # if not flipped, returns default -1
            decorated_vars.append((unsat_count, flip_time, var))
        decorated_vars.sort()
        # print(f"decorated_vars:", decorated_vars)

        sorted_vars = []
        for item in decorated_vars:
            var = item[2]
            sorted_vars.append(var)

        return sorted_vars

    def get_most_recent_flipped_var(self, clause_vars, last_flip_time):
        most_recent_var = clause_vars[0]
        max_time = last_flip_time.get(clause_vars[0], -1) # if not flipped, returns default -1

        for var in clause_vars[1:]:
            var_time = last_flip_time.get(var, -1)
            if var_time > max_time:
                max_time = var_time
                most_recent_var = var

        return most_recent_var

    def get_least_recent_flipped_var(self, clause_vars, last_flip_time):
        least_recent_var = clause_vars[0]
        min_time = last_flip_time.get(clause_vars[0], -1)

        for var in clause_vars[1:]:
            var_time = last_flip_time.get(var, -1)
            if var_time < min_time:
                min_time = var_time
                least_recent_var = var

        return least_recent_var
    
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
                
                # select one random unsatisfied clause
                unsatisfied_clauses = self.get_unsatisfied_clauses(vars)
                selected_clause = random.choice(unsatisfied_clauses)

                # select one random var to flip from the selected clause
                selected_var = abs(random.choice(selected_clause))
                vars[selected_var] = not vars[selected_var]

                # print(
                #     f"Try {try_idx + 1:2d} | Step {step + 1:3d} | "
                #     f"Flipped var x{selected_var:<2d} -> {vars[selected_var]!s:<5s} | "
                #     f"Satisfied: {self.count_satisfied_clauses(vars)}/{self.num_clauses} clauses "
                # )

        return status, vars, max_steps, max_tries

    def solve_novelty(self, max_steps, max_tries, prob_p):
        choices = [True, False]
        status = False

        for try_idx in range(max_tries):
            vars = {}
            for var in range(1, self.num_vars + 1):
                vars[var] = random.choice(choices)

            last_flip_time = {}

            for step in range(max_steps):
                if self.is_satisfied(vars):
                    status = True
                    return status, vars, step, try_idx

                unsatisfied_clauses = self.get_unsatisfied_clauses(vars)
                selected_clause = random.choice(unsatisfied_clauses)

                # novelty heuristic
                clause_vars = []
                for var in selected_clause:
                    var_idx = abs(var)
                    clause_vars.append(var_idx)

                sorted_vars = self.sort_clause_vars(clause_vars, vars, last_flip_time)
                best_var = sorted_vars[0]
                second_best_var = sorted_vars[1]
                most_recent_flipped_var = self.get_most_recent_flipped_var(clause_vars, last_flip_time)
                # print(f"most_recent_flipped_var:", most_recent_flipped_var)

                if best_var != most_recent_flipped_var:
                    selected_var = best_var
                else:
                    if random.random() < prob_p:
                        selected_var = second_best_var
                    else:
                        selected_var = best_var

                vars[selected_var] = not vars[selected_var]
                last_flip_time[selected_var] = step

                # print(
                #     f"Try {try_idx + 1:2d} | Step {step + 1:3d} | "
                #     f"Flipped var x{selected_var:<2d} -> {vars[selected_var]!s:<5s} | "
                #     f"Satisfied: {self.count_satisfied_clauses(vars)}/{self.num_clauses} clauses "
                # )

        return status, vars, max_steps, max_tries

    def solve_r_novelty(self, max_steps, max_tries, prob_p):
        choices = [True, False]
        status = False

        for try_idx in range(max_tries):
            vars = {}
            for var in range(1, self.num_vars + 1):
                vars[var] = random.choice(choices)

            last_flip_time = {}

            for step in range(max_steps):
                if self.is_satisfied(vars):
                    status = True
                    return status, vars, step, try_idx

                unsatisfied_clauses = self.get_unsatisfied_clauses(vars)
                selected_clause = random.choice(unsatisfied_clauses)

                # r-novelty heuristic
                clause_vars = []
                for var in selected_clause:
                    var_idx = abs(var)
                    clause_vars.append(var_idx)

                if step > 0 and step % 100 == 0:
                    selected_var = random.choice(clause_vars) # break deterministic loops
                else:
                    sorted_vars = self.sort_clause_vars(clause_vars, vars, last_flip_time)
                    best_var = sorted_vars[0]
                    second_best_var = sorted_vars[1]
                    most_recent_flipped_var = self.get_most_recent_flipped_var(clause_vars, last_flip_time)
                    # print(f"most_recent_flipped_var:", most_recent_flipped_var)

                    if best_var != most_recent_flipped_var:
                        selected_var = best_var
                    else:
                        best_unsat = self.flip_and_count_unsatisfied_clauses(vars, best_var)
                        second_best_unsat = self.flip_and_count_unsatisfied_clauses(vars, second_best_var)
                        n = second_best_unsat - best_unsat

                        if prob_p < 0.5:
                            if n > 1:
                                selected_var = best_var
                            else: # n == 1
                                if random.random() < (2 * prob_p):
                                    selected_var = second_best_var
                                else:
                                    selected_var = best_var
                        else: # prob_p >= 0.5
                            if n == 1:
                                selected_var = second_best_var
                            else:
                                if random.random() < (2 * (prob_p - 0.5)):
                                    selected_var = second_best_var
                                else:
                                    selected_var = best_var

                vars[selected_var] = not vars[selected_var]
                last_flip_time[selected_var] = step

                # print(
                #     f"Try {try_idx + 1:2d} | Step {step + 1:3d} | "
                #     f"Flipped var x{selected_var:<2d} -> {vars[selected_var]!s:<5s} | "
                #     f"Satisfied: {self.count_satisfied_clauses(vars)}/{self.num_clauses} clauses "
                # )

        return status, vars, max_steps, max_tries

    def solve_novelty_plus(self, max_steps, max_tries, prob_p, walk_prob_wp):
        choices = [True, False]
        status = False

        for try_idx in range(max_tries):
            vars = {}
            for var in range(1, self.num_vars + 1):
                vars[var] = random.choice(choices)
            
            last_flip_time = {}

            for step in range(max_steps):
                if self.is_satisfied(vars):
                    status = True
                    return status, vars, step, try_idx
                
                unsatisfied_clauses = self.get_unsatisfied_clauses(vars)
                selected_clause = random.choice(unsatisfied_clauses)

                # novelty+ heuristic
                clause_vars = []
                for var in selected_clause:
                    var_idx = abs(var)
                    clause_vars.append(var_idx)

                if random.random() < walk_prob_wp:
                    selected_var = random.choice(clause_vars)
                else:
                    sorted_vars = self.sort_clause_vars(clause_vars, vars, last_flip_time)
                    best_var = sorted_vars[0]
                    second_best_var = sorted_vars[1]
                    most_recent_flipped_var = self.get_most_recent_flipped_var(clause_vars, last_flip_time)

                    if best_var != most_recent_flipped_var:
                        selected_var = best_var
                    else:
                        if random.random() < prob_p:
                            selected_var = second_best_var
                        else:
                            selected_var = best_var

                vars[selected_var] = not vars[selected_var]
                last_flip_time[selected_var] = step

                # print(
                #     f"Try {try_idx + 1:2d} | Step {step + 1:3d} | "
                #     f"Flipped var x{selected_var:<2d} -> {vars[selected_var]!s:<5s} | "
                #     f"Satisfied: {self.count_satisfied_clauses(vars)}/{self.num_clauses} clauses "
                # )

        return status, vars, max_steps, max_tries

    def solve_r_novelty_plus(self, max_steps, max_tries, prob_p, walk_prob_wp):
        choices = [True, False]
        status = False

        for try_idx in range(max_tries):
            vars = {}
            for var in range(1, self.num_vars + 1):
                vars[var] = random.choice(choices)
            
            last_flip_time = {}

            for step in range(max_steps):
                if self.is_satisfied(vars):
                    status = True
                    return status, vars, step, try_idx

                unsatisfied_clauses = self.get_unsatisfied_clauses(vars)
                selected_clause = random.choice(unsatisfied_clauses)

                # r-novelty+ heuristic
                clause_vars = []
                for var in selected_clause:
                    var_idx = abs(var)
                    clause_vars.append(var_idx)

                if random.random() < walk_prob_wp:
                    selected_var = random.choice(clause_vars)
                else:
                    sorted_vars = self.sort_clause_vars(clause_vars, vars, last_flip_time)
                    best_var = sorted_vars[0]
                    second_best_var = sorted_vars[1]
                    most_recent_flipped_var = self.get_most_recent_flipped_var(clause_vars, last_flip_time)

                    if best_var != most_recent_flipped_var:
                        selected_var = best_var
                    else:
                        best_unsat = self.flip_and_count_unsatisfied_clauses(vars, best_var)
                        second_best_unsat = self.flip_and_count_unsatisfied_clauses(vars, second_best_var)
                        n = second_best_unsat - best_unsat

                        if prob_p < 0.5:
                            if n > 1:
                                selected_var = best_var
                            else:  # n == 1
                                if random.random() < (2 * prob_p):
                                    selected_var = second_best_var
                                else:
                                    selected_var = best_var
                        else: # prob_p >= 0.5
                            if n == 1:
                                selected_var = second_best_var
                            else:  # n > 1
                                if random.random() < (2 * (prob_p - 0.5)):
                                    selected_var = second_best_var
                                else:
                                    selected_var = best_var

                vars[selected_var] = not vars[selected_var]
                last_flip_time[selected_var] = step

                # print(
                #     f"Try {try_idx + 1:2d} | Step {step + 1:3d} | "
                #     f"Flipped var x{selected_var:<2d} -> {vars[selected_var]!s:<5s} | "
                #     f"Satisfied: {self.count_satisfied_clauses(vars)}/{self.num_clauses} clauses "
                # )

        return status, vars, max_steps, max_tries

    def solve_novelty_plus_plus(self, max_steps, max_tries, prob_p, div_prob_dp):
        choices = [True, False]
        status = False

        for try_idx in range(max_tries):
            vars = {}
            for var in range(1, self.num_vars + 1):
                vars[var] = random.choice(choices)
            
            last_flip_time = {}

            for step in range(max_steps):
                if self.is_satisfied(vars):
                    status = True
                    return status, vars, step, try_idx
                
                unsatisfied_clauses = self.get_unsatisfied_clauses(vars)
                selected_clause = random.choice(unsatisfied_clauses)

                clause_vars = []
                for var in selected_clause:
                    var_idx = abs(var)
                    clause_vars.append(var_idx)

                # novelty++ heuristic
                if random.random() < div_prob_dp:
                    selected_var = self.get_least_recent_flipped_var(clause_vars, last_flip_time)
                else:
                    sorted_vars = self.sort_clause_vars(clause_vars, vars, last_flip_time)
                    best_var = sorted_vars[0]
                    second_best_var = sorted_vars[1]
                    most_recent_flipped_var = self.get_most_recent_flipped_var(clause_vars, last_flip_time)

                    if best_var != most_recent_flipped_var:
                        selected_var = best_var
                    else:
                        if random.random() < prob_p:
                            selected_var = second_best_var
                        else:
                            selected_var = best_var

                vars[selected_var] = not vars[selected_var]
                last_flip_time[selected_var] = step

                # print(
                #     f"Try {try_idx + 1:2d} | Step {step + 1:3d} | "
                #     f"Flipped var x{selected_var:<2d} -> {vars[selected_var]!s:<5s} | "
                #     f"Satisfied: {self.count_satisfied_clauses(vars)}/{self.num_clauses} clauses "
                # )

        return status, vars, max_steps, max_tries

if __name__ == "__main__":
    file_path = config.FILE_PATH 
    sat_prob = open(file_path, "r")
    num_vars, num_clauses, clauses = utils.parse_sat_prob(sat_prob)
    sat_prob.close()

    walk_solver = WalkSAT(num_vars=num_vars, num_clauses=num_clauses, clauses=clauses)
    # status, vars, step, try_idx = walk_solver.solve(max_tries=10, max_steps=1000)
    # status, vars, step, try_idx = walk_solver.solve_novelty(max_tries=10, max_steps=1000, prob_p=0.5)
    # status, vars, step, try_idx = walk_solver.solve_r_novelty(max_tries=10, max_steps=1000, prob_p=0.5)
    # status, vars, step, try_idx = walk_solver.solve_novelty_plus(max_tries=10, max_steps=1000, prob_p=0.5, walk_prob_wp=0.01)
    # status, vars, step, try_idx = walk_solver.solve_r_novelty_plus(max_tries=10, max_steps=1000, prob_p=0.5, walk_prob_wp=0.01)
    status, vars, step, try_idx = walk_solver.solve_novelty_plus_plus(max_tries=10, max_steps=1000, prob_p=0.5, div_prob_dp=0.01)
    print(f"status:", status)
    print(f"vars:", vars)
    print(f"step:", step)
    print(f"try_idx:", try_idx)