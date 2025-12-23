"""
Configuration file for the HeteroFM experiment.
All tunable parameters are centralized here.
"""

# =============================================================================
# FEDERATED LEARNING PARAMETERS
# =============================================================================

COMMUNICATION_ROUNDS = 100
LOCAL_EPOCHS = 5

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

CLIENT_SIZES = [128, 128, 256, 256, 512, 512, 512, 1024, 1024, 1024]  # [32, 32, 48, 48, 64, 64, 96, 96, 128, 128]
NUM_CLIENTS = len(CLIENT_SIZES)
MAX_HIDDEN_SIZE = max(CLIENT_SIZES)
MIN_HIDDEN_SIZE = min(CLIENT_SIZES)

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
LEARNING_RATE = 0.01 # 0.001
LR_SCHEDULER = 'MultiStepLR'
LR_DECAY_MILESTONES = [50, 75]  # [150, 250]
LR_DECAY_GAMMA = 0.1
GRAD_CLIP_NORM = 1.0

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

USE_MASKED_LOSS = True
NUM_WORKERS = 0 # 변수 없애고 사용하는 부분도 옵션 그냥 없애기?

# =============================================================================
# REARRANGEMENT PARAMETERS
# =============================================================================

REARRANGE = True
PERMUTE = 'M' # 'Z' for zeroing, 'M' for maximizing
MATCH = 'E' # 'C' for contraction, 'E' for extension

# Z + C : loss = 긴 것을 짧은 것의 길이에 맞추어 내적한 값의 절대값
# M + C : loss = 긴 것을 짧은 것의 길이에 맞추어 내적한 값의 (-1) 배
# Z + E : loss = 짧은 것을 긴 것의 길이에 맞추어 0으로 패딩 후 내적한 값의 절대값
# M + E : loss = 짧은 것을 긴 것의 길이에 맞추어 copy 패딩 (뒤쪽 dim에 해당하는 원소의 값은 긴 것의 값과 일치시킴) 후 내적한 값의 (-1) 배