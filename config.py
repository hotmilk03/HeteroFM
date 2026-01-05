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

MODEL = 'mlp2' # 'mlp2', 'mlp3', 'vgg', 'resnet'

VGG_CFG = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}

MLP_BASE_WIDTH = 1024
VGG_BASE_WIDTH = 512  # Defines the width of the VGG model that w=1.0 corresponds to
RESNET_BASE_WIDTH = 64

W_CLIENT = [1/8, 1/8, 1/4, 1/4, 1/2, 1/2, 1/2, 1, 1, 1]
NUM_CLIENTS = len(W_CLIENT)
MAX_W = max(W_CLIENT)
MIN_W = min(W_CLIENT)

# CLIENT_SIZES = [128, 128, 256, 256, 512, 512, 512, 1024, 1024, 1024]

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
NON_IID_CLASSES_RATIO_PER_CLIENT = 0.2

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

REARRANGE = True
PERMUTE = 'M' # 'Z' for zeroing, 'M' for maximizing
MATCH = 'E' # 'C' for contraction, 'E' for extension

# Z + C : loss = 긴 것을 짧은 것의 길이에 맞추어 내적한 값의 절대값
# M + C : loss = 긴 것을 짧은 것의 길이에 맞추어 내적한 값의 (-1) 배
# Z + E : loss = 짧은 것을 긴 것의 길이에 맞추어 0으로 패딩 후 내적한 값의 절대값
# M + E : loss = 짧은 것을 긴 것의 길이에 맞추어 copy 패딩 (뒤쪽 dim에 해당하는 원소의 값은 긴 것의 값과 일치시킴) 후 내적한 값의 (-1) 배