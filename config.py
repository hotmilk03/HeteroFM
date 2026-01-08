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

MODEL = 'vgg' # 'mlp2', 'mlp3', 'vgg', 'resnet'

VGG_WIDTH_MULTIPLIER = 1 # 8
VGG_CFG_ORIGIN = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}
VGG_CFG = {
    key: [item if item == 'M' else item * VGG_WIDTH_MULTIPLIER for item in VGG_CFG_ORIGIN[key]]
    for key in VGG_CFG_ORIGIN
}

MLP_BASE_WIDTH = 1024 # * 16
VGG_BASE_WIDTH = 512 * VGG_WIDTH_MULTIPLIER  # Defines the width of the VGG model that w=1.0 corresponds to
RESNET_BASE_WIDTH = 64

W_CLIENT = [1/8, 1/8, 1/4, 1/4, 1/2, 1/2, 1/2, 1, 1, 1] # [1,1,1,1,1,1,1,1,1,1]
NUM_CLIENTS = len(W_CLIENT)
MAX_W = max(W_CLIENT)
MIN_W = min(W_CLIENT)

# =============================================================================
# DATA PARAMETERS
# =============================================================================

model_to_dataset = {
    'mlp2': 'mnist',
    'mlp3': 'mnist',
    'vgg': 'cifar10',
    'resnet': 'imagenet'
}
DATA_SET = model_to_dataset[MODEL] # 'mnist', 'cifar10', 'imagenet'

BATCH_SIZE = 32
DATA_DIR = './data'
DATA_SPLIT_MODE = 'non-iid' # Data split mode. 'iid' or 'non-iid'.

# Number of classes assigned to each client in the non-iid setting.
# This value is a ratio of the total number of classes.
NON_IID_CLASSES_RATIO_PER_CLIENT = 1

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

# Learning rate for the SGD optimizer.
LEARNING_RATE = 0.01 # 0.001
LR_SCHEDULER = 'MultiStepLR'
LR_DECAY_MILESTONES = [50, 75]  # [150, 250]
LR_DECAY_GAMMA = 0.1 # 0.5 ~ 0.8
GRAD_CLIP_NORM = 1.0

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

USE_MASKED_LOSS = True
NUM_WORKERS = 0 # 변수 없애고 사용하는 부분도 옵션 그냥 없애기?

# =============================================================================
# REARRANGEMENT PARAMETERS
# =============================================================================

REARRANGE = False
PERMUTE = 'M' # 'Z' for zeroing, 'M' for maximizing
MATCH = 'E' # 'C' for contraction, 'E' for extension

# =============================================================================
# PRINTING OPTIONS
# =============================================================================

SILENT = True
PERM_WARNING = True
CLIENT_EVAL = True
DRAW_ALL = True # TODO : draw clients' acc/loss range in graph