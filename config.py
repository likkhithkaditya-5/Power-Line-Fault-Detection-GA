# ----- File paths -----
# The real Kaggle dataset ships as two separate files, used for two
# separate tasks:
#   detect_dataset.csv -- binary fault / no-fault detection
#   classData.csv       -- fault-type classification (G, C, B, A bits)
BINARY_DATA_PATH = "data/detect_dataset.csv"
MULTICLASS_DATA_PATH = "data/classData.csv"

# detect_dataset.csv's binary label column is usually called "Output", but
# this tries a short list of likely names (case-insensitive) so a slightly
# different header on the actual download doesn't just crash.
BINARY_TARGET_CANDIDATES = ["Output (S)", "Output", "output", "Fault", "fault", "Target", "target"]

RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"

# ----- Feature / column names -----
FEATURE_COLUMNS = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
FAULT_BIT_COLUMNS = ["G", "C", "B", "A"]

# ----- Train / validation / test split -----
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
RANDOM_SEED = 0

# ----- K-fold CV used INSIDE the GA fitness evaluation (on training data only) -----
GA_CV_FOLDS = 3

# ----- GA hyperparameters (DEAP) -----
GA_POPULATION_SIZE = 20
GA_NUM_GENERATIONS = 15
GA_CROSSOVER_PROB = 0.7
GA_MUTATION_PROB = 0.3
GA_TOURNAMENT_SIZE = 3
GA_FEATURE_PENALTY_LAMBDA = 0.05  # lambda in: fitness = macro_F1 - lambda * (features_used / 6)

# The GA's internal fitness evaluation (many hundreds of model fits per run)
# uses a random stratified SUBSAMPLE of the training set to keep search time
# reasonable. The final chosen individual is then re-fit on the FULL training
# set for the number reported in the comparison table, so accuracy is never
# measured on the subsample -- only the search itself is sped up by it.
GA_SEARCH_SAMPLE_SIZE = 2000

# ----- Multiclass fault-type labels (order matters for plots/confusion matrices) -----
FAULT_TYPE_LABELS = ["LG", "LL", "LLG", "LLL", "LLLG"]
