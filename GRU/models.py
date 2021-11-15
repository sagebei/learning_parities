import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = x.new_zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)

        out = out[:, -1, :]
        out = self.fc(out)
        return out




