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
                    nn.BatchNorm2d(out_channels, momentum=None, track_running_stats=self.track),
                    nn.ReLU(inplace=True)
                ]
                current_in_channels = out_channels
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)

# =============================================================================
# ResNet Model (for ImageNet)
# =============================================================================

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, scaler_rate, track):
        super(Bottleneck, self).__init__()
        self.track = track
        self.scaler_rate = scaler_rate

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=None, track_running_stats=self.track)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=None, track_running_stats=self.track)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes, momentum=None, track_running_stats=self.track)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, momentum=None, track_running_stats=self.track)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(Scaler(self.scaler_rate)(out)))
        out = self.conv2(out)
        out = F.relu(self.bn2(Scaler(self.scaler_rate)(out)))
        out = self.conv3(out)
        out = self.bn3(Scaler(self.scaler_rate)(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, w, scaler_rate=1.0, track=False, in_channels=3, num_classes=1000):
        super(ResNet, self).__init__()
        
        base_width = int(w * config.RESNET_BASE_WIDTH)
        base_width = max(1, base_width)

        self.in_planes = base_width
        self.scaler_rate = scaler_rate
        self.track = track

        self.conv1 = nn.Conv2d(in_channels, base_width, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width, momentum=None, track_running_stats=self.track)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, base_width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_width*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_width*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_width*8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(base_width*8*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for strd in strides:
            layers.append(block(self.in_planes, planes, strd, self.scaler_rate, self.track))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(Scaler(self.scaler_rate)(self.conv1(x))))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

def ResNet50(w, scaler_rate, track, in_channels, num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], w, scaler_rate, track, in_channels, num_classes)

# =============================================================================
# Model Factory
# =============================================================================

def init_model(w, scaler_rate=1.0):
    # Determine model parameters based on dataset
    if config.DATA_SET == 'mnist':
        in_channels, num_classes = 1, 10
    elif config.DATA_SET == 'cifar10':
        in_channels, num_classes = 3, 10
    elif config.DATA_SET == 'imagenet':
        in_channels, num_classes = 3, 1000
    else:
        raise ValueError(f"Unknown dataset: {config.DATA_SET}")

    # Instantiate model
    if config.MODEL == 'mlp2':
        model = MLP2(w, scaler_rate, False, in_channels, num_classes)
    elif config.MODEL == 'mlp3':
        model = MLP3(w, scaler_rate, False, in_channels, num_classes)
    elif config.MODEL == 'vgg':
        model = VGG('VGG11', w, scaler_rate, False, in_channels, num_classes)
    elif config.MODEL == 'resnet':
        model = ResNet50(w, scaler_rate, False, in_channels, num_classes)
    else:
        raise ValueError(f"Unknown model: {config.MODEL}")
        
    model.apply(init_param)
    return model