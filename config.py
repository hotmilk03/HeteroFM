"""
Configuration file for the HeteroFM experiment.
All tunable parameters are centralized here.
"""

# =============================================================================
# FEDERATED LEARNING PARAMETERS
# =============================================================================

COMMUNICATION_ROUNDS = 4
LOCAL_EPOCHS = 5

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

CLIENT_SIZES = [6, 6, 8, 8, 12, 12, 16, 16, 32, 32] # [32, 32, 48, 48, 64, 64, 96, 96, 128, 128]
NUM_CLIENTS = len(CLIENT_SIZES)
MAX_HIDDEN_SIZE = max(CLIENT_SIZES)

# =============================================================================
# DATA PARAMETERS
# =============================================================================

BATCH_SIZE = 32
DATA_DIR = './data'
DATA_SPLIT_MODE = 'non-iid' # Data split mode. 'iid' or 'non-iid'.

# Number of classes assigned to each client in the non-iid setting.
# For MNIST, there are 10 classes in total.
NON_IID_N_CLASSES_PER_CLIENT = 2

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

# Learning rate for the SGD optimizer.
LEARNING_RATE = 0.01

USE_MASKED_LOSS = True
NUM_WORKERS = 4