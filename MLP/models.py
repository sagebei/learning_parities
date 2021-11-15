import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_layers):
        super(MLP, self).__init__()

        self.mlp = nn.Linear(input_size, hidden_size)

        self.hidden_layers = nn.ModuleList()
        for n in range(n_layers - 1):
            layer = nn.Linear(hidden_size, hidden_size)
            self.hidden_layers.append(layer)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.mlp(x)
        out = F.relu(out)

        for layer in self.hidden_layers:
            out = layer(out)
            out = F.relu(out)

        out = self.fc(out)
        return out
