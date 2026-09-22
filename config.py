# FILE_PATH = "/DATA/FCD_LAB/user1/TH/dataset/uf20-91/uf20-01.cnf"
# FILE_PATH = "/DATA/FCD_LAB/user1/TH/dataset/uf50-218/uf50-01.cnf"
FILE_PATH = "/DATA/FCD_LAB/user1/TH/dataset/uf100-430/uf100-01.cnf"
# FILE_PATH = "/DATA/FCD_LAB/user1/TH/dataset/uf150-645/ai/hoos/Research/SAT/Formulae/UF150.645.100/uf150-01.cnf"
# FILE_PATH = "/DATA/FCD_LAB/user1/TH/dataset/uf200-860/uf200-01.cnf"
# FILE_PATH = "/DATA/FCD_LAB/user1/TH/dataset/uf250-1065/ai/hoos/Shortcuts/UF250.1065.100/uf250-01.cnf"

# pubo
SIGMA = 2 ** -7
STEPS = 500
START_TEMP = 1.0
END_TEMP = 0.01
LOG_INTERVAL = 100
BATCH_SIZE = 100

# qubo
# SIGMA = 2 ** -7
# STEPS = 5000
# START_TEMP = 1.0
# END_TEMP = 0.1
# LOG_INTERVAL = 250
# BATCH_SIZE = 100
# PENALTY = 0.5

INSTANCE = 3
RUNS_PER_INSTANCE = 100

# experiment
TARGET_SUCCESS_RATE = 0.99