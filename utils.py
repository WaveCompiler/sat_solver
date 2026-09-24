# compute
def parse_sat_prob(sat_prob):
    clauses = []
    num_vars = 0
    num_clauses = 0
    for line in sat_prob:
        line = line.strip()
        if not line or line.startswith('c') or line.startswith('%'):
            continue
        if line.startswith('0'):
            break

        if line.startswith('p cnf'):
            parts = line.split()
            num_vars = int(parts[2])
            num_clauses = int(parts[3])
            continue

        clause = line.split()
        x_idxs = []
        for x_idx in clause:
            if x_idx == '0':
                break
            x_idxs.append(int(x_idx))

        if x_idxs:
            clauses.append(x_idxs)

    return num_vars, num_clauses, clauses
