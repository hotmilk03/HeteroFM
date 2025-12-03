# =============================================================================
# FEDERATED LEARNING & MODEL PARAMETERS
# =============================================================================

CLIENT_SIZES = [32, 32, 48, 48, 64, 64, 96, 96, 128, 128]
NUM_CLIENTS = len(CLIENT_SIZES)
MAX_HIDDEN_SIZE = max(CLIENT_SIZES)

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

COMMUNICATION_ROUNDS = 100
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# =============================================================================
# DATA SPLIT PARAMETERS
# =============================================================================

# Data split mode. Can be 'iid' or 'non-iid'.
DATA_SPLIT_MODE = 'iid'

# Number of classes assigned to each client in the non-iid setting.
# For MNIST, there are 10 classes in total.
NON_IID_N_CLASSES_PER_CLIENT = 2


# =============================================================================
# DATA & ENVIRONMENT PARAMETERS
# =============================================================================

# Directory to store the MNIST dataset, inside the project folder.
DATA_DIR = './data'
