import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride,
                 n_layers):
        super(CNN, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=0 if n_layers == 1 else 1)
        self.norm_layer = nn.BatchNorm1d(out_channel)

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for n in range(n_layers - 1):
            if n == n_layers - 2:
                conv = nn.Conv1d(in_channels=out_channel,
                                 out_channels=out_channel,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=0)
                norm_layer = nn.BatchNorm1d(out_channel)
            else:
                conv = nn.Conv1d(in_channels=out_channel,
                                 out_channels=out_channel,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=1)
                norm_layer = nn.BatchNorm1d(out_channel)

            self.conv_layers.append(conv)
            self.norm_layers.append(norm_layer)

        self.fc = nn.Linear(out_channel, 1)

    def forward(self, x):

        out = self.conv(x)
        out = self.norm_layer(out)
        out = F.relu(out)
        print(out.size())

        for conv, norm in zip(self.conv_layers, self.norm_layers):
            out = conv(out)
            out = norm(out)
            out = F.relu(out)

        out = out.view(x.size(0), -1)
        print(out.size())
        out = self.fc(out)
        return out

