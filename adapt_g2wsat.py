import random
import config
import utils


class AdaptG2WSATP:
    def __init__(self, num_vars, num_clauses, clauses):
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.clauses = clauses

    def is_clause_satisfied(self, clause, vars_assignment):
        for var in clause:
            if var > 0:
                is_true = vars_assignment[var]
            else:
                var_idx = abs(var)
                is_true = not vars_assignment[var_idx]

            if is_true:
                return True
        return False

    def is_satisfied(self, vars_assignment):
        for clause in self.clauses:
            if not self.is_clause_satisfied(clause, vars_assignment):
                return False
        return True

    def get_unsatisfied_clauses(self, vars_assignment):
        unsatisfied_clauses = []
        for clause in self.clauses:
            if not self.is_clause_satisfied(clause, vars_assignment):
                unsatisfied_clauses.append(clause)
        return unsatisfied_clauses

    def count_unsatisfied_clauses(self, vars_assignment):
        return len(self.get_unsatisfied_clauses(vars_assignment))

    def flip_and_count_unsatisfied_clauses(self, vars_assignment, var):
        vars_assignment[var] = not vars_assignment[var]
        num_unsatisfied = self.count_unsatisfied_clauses(vars_assignment)
        vars_assignment[var] = not vars_assignment[var]
        return num_unsatisfied

    def calculate_variable_scores(self, vars_assignment):
        scores = {}
        num_unsatisfied_before = self.count_unsatisfied_clauses(vars_assignment)
        for var in range(1, self.num_vars + 1):
            num_unsatisfied_after = self.flip_and_count_unsatisfied_clauses(
                vars_assignment, var
            )
            scores[var] = num_unsatisfied_before - num_unsatisfied_after
        return scores

    def compute_promising_score(
        self, var, current_vars, base_score, old_scores, promising_vars
    ):
        """Calculates pscore(x) = score_A(x) + score_B(x') via look-ahead simulation."""
        # 1. Simulate flipping var
        current_vars[var] = not current_vars[var]
        new_scores = self.calculate_variable_scores(current_vars)

        # 2. Identify promising decreasing variables after flipping var
        temp_promising = set(promising_vars)
        for v in range(1, self.num_vars + 1):
            if old_scores[v] <= 0 and new_scores[v] > 0:
                temp_promising.add(v)
            elif new_scores[v] <= 0:
                temp_promising.discard(v)
        temp_promising.discard(var)

        # 3. Find x' (best promising decreasing variable after flip)
        best_promising_score = -float("inf")
        if len(temp_promising) > 0:
            for p_var in temp_promising:
                if new_scores[p_var] > best_promising_score:
                    best_promising_score = new_scores[p_var]

        # 4. Revert state
        current_vars[var] = not current_vars[var]

        if best_promising_score > -float("inf"):
            return base_score + best_promising_score
        else:
            return base_score

    def get_least_recently_flipped(self, var_list, last_flip_time):
        least_recent_var = var_list[0]
        min_time = last_flip_time.get(var_list[0], -1)

        for var in var_list[1:]:
            var_time = last_flip_time.get(var, -1)
            if var_time < min_time:
                min_time = var_time
                least_recent_var = var

        return least_recent_var

    def novelty_plus_p(
        self,
        p,
        wp,
        clause,
        vars_assignment,
        last_flip_time,
        scores,
        old_scores,
        promising_vars,
    ):
        """Implements Novelty+P heuristic with limited look-ahead."""
        clause_vars = [abs(var) for var in clause]

        # Random walk step
        if random.random() < wp:
            return random.choice(clause_vars)

        # Sort variables by score, breaking ties by least recently flipped
        decorated_vars = []
        for var in clause_vars:
            score = scores[var]
            flip_time = last_flip_time.get(var, -1)
            decorated_vars.append((-score, flip_time, var))
        decorated_vars.sort()

        best = decorated_vars[0][2]
        second = decorated_vars[1][2] if len(decorated_vars) > 1 else best

        # Determine recency
        best_time = last_flip_time.get(best, -1)
        second_time = last_flip_time.get(second, -1)
        most_recent_var = clause_vars[0]
        max_time = last_flip_time.get(clause_vars[0], -1)

        for v in clause_vars[1:]:
            v_time = last_flip_time.get(v, -1)
            if v_time > max_time:
                max_time = v_time
                most_recent_var = v

        if best == most_recent_var:
            if random.random() < p:
                return second
            else:
                if best_time > second_time:
                    pscore_best = self.compute_promising_score(
                        best,
                        vars_assignment,
                        scores[best],
                        old_scores,
                        promising_vars,
                    )
                    pscore_second = self.compute_promising_score(
                        second,
                        vars_assignment,
                        scores[second],
                        old_scores,
                        promising_vars,
                    )
                    if pscore_second >= pscore_best:
                        return second
                    else:
                        return best
                else:
                    return best
        else:
            if best_time > second_time:
                pscore_best = self.compute_promising_score(
                    best,
                    vars_assignment,
                    scores[best],
                    old_scores,
                    promising_vars,
                )
                pscore_second = self.compute_promising_score(
                    second,
                    vars_assignment,
                    scores[second],
                    old_scores,
                    promising_vars,
                )
                if pscore_second >= pscore_best:
                    return second
                else:
                    return best
            else:
                return best

    def solve(self, max_steps, max_tries):
        choices = [True, False]
        status = False

        # Adaptive noise hyper-parameters (Section 3.2: theta=1/5, phi=0.1)
        theta = 1.0 / 5.0
        phi = 0.1
        check_interval = int(theta * self.num_clauses)
        if check_interval < 1:
            check_interval = 1

        for try_idx in range(max_tries):
            vars_assignment = {}
            for var in range(1, self.num_vars + 1):
                vars_assignment[var] = random.choice(choices)

            p = 0.0
            wp = 0.0
            last_flip_time = {}
            scores = self.calculate_variable_scores(vars_assignment)

            # Initialize DecVar stack with all initial decreasing variables
            promising_vars = set()
            for var in range(1, self.num_vars + 1):
                if scores[var] > 0:
                    promising_vars.add(var)

            last_check_step = 0
            best_unsat_count = self.count_unsatisfied_clauses(vars_assignment)
            last_improved_unsat = best_unsat_count

            for step in range(max_steps):
                if self.is_satisfied(vars_assignment):
                    status = True
                    return status, vars_assignment, step, try_idx

                current_unsat = self.count_unsatisfied_clauses(vars_assignment)

                # Adaptive noise adjustment logic
                if current_unsat < last_improved_unsat:
                    p = p - (p * phi / 2.0)
                    wp = p / 10.0
                    last_improved_unsat = current_unsat
                    last_check_step = step

                if (step - last_check_step) >= check_interval:
                    p = p + (1.0 - p) * phi
                    wp = p / 10.0
                    last_check_step = step

                selected_var = None

                # 1. Deterministic exploitation of promising decreasing variables
                if len(promising_vars) > 0:
                    promising_list = list(promising_vars)
                    selected_var = self.get_least_recently_flipped(promising_list, last_flip_time)
                else:
                    # 2. Look-ahead fallback: Novelty+P
                    unsatisfied_clauses = self.get_unsatisfied_clauses(vars_assignment)
                    selected_clause = random.choice(unsatisfied_clauses)
                    selected_var = self.novelty_plus_p(
                        p,
                        wp,
                        selected_clause,
                        vars_assignment,
                        last_flip_time,
                        scores,
                        scores.copy(),
                        promising_vars,
                    )

                # Perform the flip
                vars_assignment[selected_var] = not vars_assignment[selected_var]
                last_flip_time[selected_var] = step

                # Update scores and promising_vars stack
                old_scores = scores.copy()
                scores = self.calculate_variable_scores(vars_assignment)

                for var in range(1, self.num_vars + 1):
                    if old_scores[var] <= 0 and scores[var] > 0:
                        promising_vars.add(var)
                    elif scores[var] <= 0:
                        promising_vars.discard(var)
                promising_vars.discard(selected_var)

        return status, vars_assignment, max_steps, max_tries

if __name__ == "__main__":
    file_path = config.FILE_PATH
    with open(file_path, "r") as sat_prob:
        num_vars, num_clauses, clauses = utils.parse_sat_prob(sat_prob)

    solver = AdaptG2WSATP(num_vars=num_vars, num_clauses=num_clauses, clauses=clauses)
    status, vars_solution, step, try_idx = solver.solve(max_steps=10000, max_tries=1)

    print("status:", status)
    print("vars:", vars_solution)
    print("step:", step)
    print("try_idx:", try_idx)