FILE_PATH = "/home/taehy/sat/sat_problem_dataset/uf100-430.tar/uf100-01.cnf"
# FILE_PATH = "/home/taehy/sat/sat_problem_dataset/uuf100-430.tar/UUF100.430.1000/uuf100-01.cnf"
# FILE_PATH = "/home/taehy/sat/sat_problem_dataset/uf150-645.tar/ai/hoos/Research/SAT/Formulae/UF150.645.100/uf150-01.cnf"
# FILE_PATH = "/home/taehy/sat/sat_problem_dataset/uf200-860.tar/uf200-860/uf200-01.cnf"
# FILE_PATH = "/home/taehy/sat/sat_problem_dataset/uf250-1065.tar/uf250-1065/ai/hoos/Shortcuts/UF250.1065.100/uf250-09.cnf"

# both pubo & qubo
SIGMA = 2 ** -8
STEPS = 100
START_TEMP = 3.0
END_TEMP = 0.001
LOG_INTERVAL = 1

# qubo
PENALTY = 1

# plot
# MULTIPLE_FILES_PATH = "/home/taehy/sat/sat_problem_dataset/uf100-430.tar/*.cnf"
# MULTIPLE_FILES_PATH = "/home/taehy/sat/sat_problem_dataset/uf150-645.tar/ai/hoos/Research/SAT/Formulae/UF150.645.100/*.cnf"
# MULTIPLE_FILES_PATH = "/home/taehy/sat/sat_problem_dataset/uf200-860.tar/uf200-860/*.cnf"
MULTIPLE_FILES_PATH = "/home/taehy/sat/sat_problem_dataset/uf250-1065.tar/uf250-1065/ai/hoos/Shortcuts/UF250.1065.100/*.cnf"
INSTANCE = 10
RUNS_PER_INSTANCE = 100