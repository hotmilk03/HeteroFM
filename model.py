import torch.nn as nn
from modules import Scaler

def init_param(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

class MLP2(nn.Module):
    def __init__(self, hidden_size, scaler_rate=1.0, track=False): # dropout_p=0.5
        super(MLP2, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),

            nn.Linear(28*28, hidden_size, bias=False),
            Scaler(scaler_rate),
            nn.BatchNorm1d(hidden_size, momentum=None, track_running_stats=track),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout_p),
            
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x):
        return self.layers(x)

class MLP3(nn.Module):
    def __init__(self, hidden_size, scaler_rate=1.0, track=False): # dropout_p=0.5
        super(MLP3, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),

            nn.Linear(28*28, hidden_size, bias=False),
            Scaler(scaler_rate),
            nn.BatchNorm1d(hidden_size, momentum=None, track_running_stats=track),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout_p),

            nn.Linear(hidden_size, hidden_size, bias=False),
            Scaler(scaler_rate),
            nn.BatchNorm1d(hidden_size, momentum=None, track_running_stats=track),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout_p),
            
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x):
        return self.layers(x)

def init_model(hidden_size, scaler_rate=1.0):
    model = MLP2(hidden_size, scaler_rate)
    # model = MLP3(hidden_size, scaler_rate)
    model.apply(init_param)
    return model