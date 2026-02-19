import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Scaler
import config

# =============================================================================
# Model Initialization
# =============================================================================

def init_param(m):
    pass
    # if isinstance(m, (nn.Conv2d, nn.Linear)):
    #     nn.init.xavier_normal_(m.weight)
    #     if m.bias is not None:
    #         m.bias.data.zero_()
    # elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
    #     if m.weight is not None:
    #         m.weight.data.fill_(1)
    #     if m.bias is not None:
    #         m.bias.data.zero_()

def _get_gn_groups(num_channels, desired_groups=4):
    if num_channels % desired_groups == 0:
        return desired_groups
    return 1

def _get_norm2d(num_channels, track):
    norm = getattr(config, 'NORM', 'bn')
    if norm == 'bn':
        return nn.BatchNorm2d(num_channels, momentum=None, track_running_stats=track)
    if norm == 'in':
        return nn.GroupNorm(num_channels, num_channels)
    if norm == 'ln':
        return nn.GroupNorm(1, num_channels)
    if norm == 'gn':
        return nn.GroupNorm(_get_gn_groups(num_channels), num_channels)
    if norm == 'none':
        return nn.Identity()
    raise ValueError(f"Unknown norm: {norm}")

# =============================================================================
# MLP Models (for MNIST)
# =============================================================================

class MLP2(nn.Module):
    def __init__(self, w, scaler_rate=1.0, track=False, in_channels=1, num_classes=10): # w : hidden_size_ratio
        super(MLP2, self).__init__()
        hidden_size = int(w * config.MLP_BASE_WIDTH)
        hidden_size = max(1, hidden_size)
        
        self.layers = nn.Sequential(
            nn.Flatten(),

            nn.Linear(in_channels * 28 * 28, hidden_size, bias=False),
            Scaler(scaler_rate),
            nn.BatchNorm1d(hidden_size, momentum=None, track_running_stats=track),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout_p),
            
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.layers(x)

class MLP3(nn.Module):
    def __init__(self, w, scaler_rate=1.0, track=False, in_channels=1, num_classes=10):
        super(MLP3, self).__init__()
        hidden_size = int(w * config.MLP_BASE_WIDTH)
        hidden_size = max(1, hidden_size)

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 28 * 28, hidden_size, bias=False),
            Scaler(scaler_rate),
            nn.BatchNorm1d(hidden_size, momentum=None, track_running_stats=track),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout_p),

            nn.Linear(hidden_size, hidden_size, bias=False),
            Scaler(scaler_rate),
            nn.BatchNorm1d(hidden_size, momentum=None, track_running_stats=track),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout_p),
            
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.layers(x)

# =============================================================================
# VGG Model (for CIFAR-10)
# =============================================================================

class VGG(nn.Module):
    def __init__(self, vgg_name, w, scaler_rate=1.0, track=False, in_channels=3, num_classes=10):
        super(VGG, self).__init__()
        self.w = w
        self.scaler_rate = scaler_rate
        self.track = track
        self.features = self._make_layers(config.VGG_CFG[vgg_name], in_channels)
        classifier_input = int(self.w * config.VGG_BASE_WIDTH)
        self.classifier = nn.Linear(max(1, classifier_input), num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, in_channels):
        layers = []
        # Keep track of the actual input channels for each conv layer
        current_in_channels = in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                out_channels = max(1, int(self.w * x))
                layers += [
                    nn.Conv2d(current_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    Scaler(self.scaler_rate),
                    _get_norm2d(out_channels, self.track),
                    nn.ReLU(inplace=True)
                ]
                current_in_channels = out_channels
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)

# =============================================================================
# ResNet Model (for CIFAR-10 or ImageNet)
# =============================================================================

class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, scaler_rate, track):
        super(Block, self).__init__()
        self.n1 = _get_norm2d(in_planes, track)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n2 = _get_norm2d(planes, track)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.scaler = Scaler(scaler_rate)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.n1(self.scaler(x)))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.n2(self.scaler(out))))
        out += shortcut
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, scaler_rate, track):
        super(Bottleneck, self).__init__()
        self.track = track
        self.scaler_rate = scaler_rate

        self.n1 = _get_norm2d(in_planes, self.track)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.n2 = _get_norm2d(planes, self.track)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n3 = _get_norm2d(planes, self.track)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        self.scaler = Scaler(self.scaler_rate)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.n1(self.scaler(x)))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.n2(self.scaler(out))))
        out = self.conv3(F.relu(self.n3(self.scaler(out))))
        out += shortcut
        return out

# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, w, scaler_rate=1.0, track=False, in_channels=3, num_classes=1000):
#         super(ResNet, self).__init__()
        
#         base_width = int(w * config.RESNET_BASE_WIDTH)
#         base_width = max(1, base_width)

#         self.in_planes = base_width
#         self.scaler_rate = scaler_rate
#         self.track = track

#         self.conv1 = nn.Conv2d(in_channels, base_width, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(base_width, momentum=None, track_running_stats=self.track)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.layer1 = self._make_layer(block, base_width, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, base_width*2, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, base_width*4, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, base_width*8, num_blocks[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.linear = nn.Linear(base_width*8*block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride, self.scaler_rate, self.track))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.relu(self.bn1(Scaler(self.scaler_rate)(self.conv1(x))))
#         out = self.maxpool(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.avgpool(out)
#         out = torch.flatten(out, 1)
#         out = self.linear(out)
#         return out

class ResNet(nn.Module):
    def __init__(self, data_shape, hidden_size, block, num_blocks, num_classes, rate, track, use_imagenet_stem=False):
        super(ResNet, self).__init__()
        self.in_planes = hidden_size[0]
        self.use_imagenet_stem = use_imagenet_stem
        
        if use_imagenet_stem:
            # ImageNet-style: 7x7 conv, stride=2, maxpool
            self.conv1 = nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=7, stride=2, padding=3, bias=False)
            self.n1 = _get_norm2d(hidden_size[0], track)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            # CIFAR-10-style: 3x3 conv, stride=1, no maxpool
            self.conv1 = nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1, rate=rate, track=track)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2, rate=rate, track=track)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2, rate=rate, track=track)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2, rate=rate, track=track)
        self.n4 = _get_norm2d(hidden_size[3] * block.expansion, track)
        
        self.scaler = Scaler(rate)
        self.linear = nn.Linear(hidden_size[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, rate, track):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, rate, track))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.use_imagenet_stem:
            out = F.relu(self.n1(out))
            out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.n4(self.scaler(out)))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def _get_resnet_dataset_defaults():
    if config.DATA_SET == 'mnist':
        return 10, (1, 28, 28)
    if config.DATA_SET == 'cifar10':
        return 10, (3, 32, 32)
    if config.DATA_SET == 'imagenet':
        return int(1000 * config.IMAGENET_SUBSET_RATIO), (3, 224, 224)
    raise ValueError(f"Unknown dataset: {config.DATA_SET}")

def _build_resnet_inputs(w):
    num_classes, data_shape = _get_resnet_dataset_defaults()

    base_width = max(1, int(w * config.RESNET_BASE_WIDTH))
    hidden_size = [base_width, base_width * 2, base_width * 4, base_width * 8]
    return data_shape, hidden_size, num_classes

def ResNet18(w, scaler_rate=1.0, track=False, in_channels=None, num_classes=None):
    data_shape, hidden_size, num_classes = _build_resnet_inputs(w)
    return ResNet(data_shape, hidden_size, Block, [2, 2, 2, 2], num_classes, scaler_rate, track, use_imagenet_stem=False)

def ResNet34(w, scaler_rate=1.0, track=False, in_channels=None, num_classes=None):
    data_shape, hidden_size, num_classes = _build_resnet_inputs(w)
    return ResNet(data_shape, hidden_size, Block, [3, 4, 6, 3], num_classes, scaler_rate, track, use_imagenet_stem=False)

def ResNet50(w, scaler_rate=1.0, track=False, in_channels=None, num_classes=None):
    data_shape, hidden_size, num_classes = _build_resnet_inputs(w)
    return ResNet(data_shape, hidden_size, Bottleneck, [3, 4, 6, 3], num_classes, scaler_rate, track, use_imagenet_stem=True)

def ResNet101(w, scaler_rate=1.0, track=False, in_channels=None, num_classes=None):
    data_shape, hidden_size, num_classes = _build_resnet_inputs(w)
    return ResNet(data_shape, hidden_size, Bottleneck, [3, 4, 23, 3], num_classes, scaler_rate, track, use_imagenet_stem=True)

def ResNet152(w, scaler_rate=1.0, track=False, in_channels=None, num_classes=None):
    data_shape, hidden_size, num_classes = _build_resnet_inputs(w)
    return ResNet(data_shape, hidden_size, Bottleneck, [3, 8, 36, 3], num_classes, scaler_rate, track, use_imagenet_stem=True)

# =============================================================================
# Model Factory
# =============================================================================

def init_model(w, scaler_rate=1.0, seed=None):
    """
    Initialize a model with optional seed for reproducibility.
    
    Args:
        w: Width multiplier ratio
        scaler_rate: Scaler rate for the model
        seed: Random seed for model initialization. If None, uses config.SEED if available.
    
    Returns:
        Initialized model
    """
    # Set seed if provided or use config.SEED
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    elif hasattr(config, 'SEED') and config.SEED is not None:
        torch.manual_seed(config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.SEED)
    
    # Determine model parameters based on dataset
    if config.DATA_SET == 'mnist':
        in_channels, num_classes = 1, 10
    elif config.DATA_SET == 'cifar10':
        in_channels, num_classes = 3, 10
    elif config.DATA_SET == 'imagenet':
        in_channels = 3
        num_classes = int(1000 * config.IMAGENET_SUBSET_RATIO)
    else:
        raise ValueError(f"Unknown dataset: {config.DATA_SET}")

    # Instantiate model
    if config.MODEL == 'mlp2':
        model = MLP2(w, scaler_rate, False, in_channels, num_classes)
    elif config.MODEL == 'mlp3':
        model = MLP3(w, scaler_rate, False, in_channels, num_classes)
    elif config.MODEL == 'vgg11':
        model = VGG('VGG11', w, scaler_rate, False, in_channels, num_classes)
    elif config.MODEL == 'resnet18':
        model = ResNet18(w, scaler_rate, False, in_channels, num_classes)
    elif config.MODEL == 'resnet34':
        model = ResNet34(w, scaler_rate, False, in_channels, num_classes)
    elif config.MODEL == 'resnet50':
        model = ResNet50(w, scaler_rate, False, in_channels, num_classes)
    elif config.MODEL == 'resnet101':
        model = ResNet101(w, scaler_rate, False, in_channels, num_classes)
    elif config.MODEL == 'resnet152':
        model = ResNet152(w, scaler_rate, False, in_channels, num_classes)
    else:
        raise ValueError(f"Unknown model: {config.MODEL}")
        
    model.apply(init_param)
    return model