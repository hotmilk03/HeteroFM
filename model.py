import torch.nn as nn
from modules import Scaler

def init_param(m):
    if isinstance(m, nn.Linear):
        m.bias.data.zero_()

class MLP(nn.Module):
    def __init__(self, hidden_size, scaler_rate=1.0):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, hidden_size),
            Scaler(scaler_rate),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x):
        return self.layers(x)

def init_model(hidden_size, scaler_rate=1.0):
    model = MLP(hidden_size, scaler_rate)
    model.apply(init_param)
    return model