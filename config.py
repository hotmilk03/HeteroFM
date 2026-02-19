"""
Configuration file for the HeteroFM experiment.
All tunable parameters are centralized here.
"""

# =============================================================================
# FEDERATED LEARNING PARAMETERS
# =============================================================================

COMMUNICATION_ROUNDS = 10
LOCAL_EPOCHS = 5 # 40

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

MODEL = 'resnet50' # 'mlp2', 'mlp3', 'vgg11', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'

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

# Normalization layer option for Conv models
# Options: 'bn', 'in', 'ln', 'gn', 'none'
NORM = 'gn'

W_CLIENT = [1,1,1,1,1,1,1,1,1,1] # [1/8, 1/8, 1/4, 1/4, 1/2, 1/2, 1/2, 1, 1, 1]
NUM_CLIENTS = len(W_CLIENT)
MAX_W = max(W_CLIENT)
MIN_W = min(W_CLIENT)

# =============================================================================
# DATA PARAMETERS
# =============================================================================

model_to_dataset = {
    'mlp2': 'mnist',
    'mlp3': 'mnist',
    'vgg11': 'cifar10',
    'resnet18': 'cifar10',
    'resnet34': 'cifar10',
    'resnet50': 'imagenet',
    'resnet101': 'imagenet',
    'resnet152': 'imagenet',
}
DATA_SET = model_to_dataset[MODEL] # 'mnist', 'cifar10', 'imagenet'

IMAGENET_SUBSET_RATIO = 0.01
BATCH_SIZE = 32 # only 32 OK
DATA_DIR = './data'
DATA_SPLIT_MODE = 'iid' # Data split mode. 'iid' or 'non-iid'.
DYNAMIC_ON_NON_IID_SPLIT = False

# Number of classes assigned to each client in the non-iid setting.
# This value is a ratio of the total number of classes.
NON_IID_CLASSES_RATIO_PER_CLIENT = 1

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

# Learning rate for the SGD optimizer.
LEARNING_RATE = 0.1 # 0.1 / 0.01
LR_SCHEDULER = 'MultiStepLR'
LR_DECAY_MILESTONES = [50, 75]  # [150, 250]
LR_DECAY_GAMMA = 0.1 # 0.5 ~ 0.8
GRAD_CLIP_NORM = 1.0

# Mixed precision helps reduce GPU memory pressure on CUDA devices.
USE_AMP = True

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

USE_MASKED_LOSS = False
NUM_WORKERS = 1 # only 1 OK

# =============================================================================
# REARRANGEMENT PARAMETERS
# =============================================================================

REARRANGE = True
PERMUTE = 'M' # 'Z' for zeroing, 'M' for maximizing
MATCH = 'C' # 'C' for contraction, 'E' for extension

# Interpolation between two mode combinations (e.g., MC + ZE)
REARRANGE_INTERPOLATE = False  # Enable interpolation between two mode pairs
PERMUTE_INTERPOLATE = 'Z'  # Second permute mode to interpolate with
MATCH_INTERPOLATE = 'E'    # Second match mode to interpolate with
INTERPOLATE_WEIGHT = 0.5   # Weight for interpolation (0.5 = equal mix)

SINKHORN = False
SINKHORN_MAX_ITER = 50  # Maximum GD iterations (50-100 needed for Hungarian parity)
SINKHORN_NUM_ITER = 20  # Number of Sinkhorn-Knopp iterations per GD step
SINKHORN_LAMBDA = 1.0
SINKHORN_LR = 1.0  # Learning rate for gradient descent optimization # 0.1 or 1.0
SINKHORN_SCALING = 1.0  # Cost scaling factor

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

# Global random seed for reproducibility
# Set to None to disable reproducibility
SEED = 42

# =============================================================================
# PRINTING OPTIONS
# =============================================================================

SILENT = True
PERM_WARNING = True
CLIENT_EVAL = True
DRAW = 'all' # 'acc' or 'loss' or 'all'

# =============================================================================
# INTERPOLATION
# =============================================================================

INTERPOLATION = True
INTERPOLATION_ROUND = 100
INTERPOLATION_LAMBDA_START = -0.2
INTERPOLATION_LAMBDA_END = 1.2
INTERPOLATION_LAMBDA_STEP = 0.1  # 0.1 or 0.02
INTERPOLATION_ANIMATION = True  # True for animated plot, False for static plot
INTERPOLATION_3D = False  # True for 3D plot, False for 2D plot