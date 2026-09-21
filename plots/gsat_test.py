import os
import sys

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import config
import utils
from gsat import GSAT

if __name__ == "__main__":
    file_path = config.FILE_PATH

    with open(file_path, "r") as sat_prob:
        num_vars, num_clauses, clauses = utils.parse_sat_prob(sat_prob)

    solver = GSAT(num_vars=num_vars, num_clauses=num_clauses)
    status, vars, flip_idx, try_idx = solver.solve(clauses=clauses, max_flips=50, max_tries=10)

    print("\n--- Test Complete ---")
    print(f"Status: {status}")
    print(f"Flips : {flip_idx}")
    print(f"Tries : {try_idx + 1}")