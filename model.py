import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x):
        return self.layers(x)

def init_model(hidden_size):
    return MLP(hidden_size)