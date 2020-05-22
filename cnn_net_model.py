import torch
import torch.nn as nn


class CnnNet(nn.Module):

    def __init__(self):
        super(CnnNet, self).__init__()
        self.input_conv = nn.Conv1d(1, 32, 5, padding=2)
        self.rec_conv = nn.Conv1d(32, 32, 5, padding=2)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool1d(5, stride=2)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        prev_x = self.input_conv(x)

        for i in range(5):
            x = self.rec_conv(prev_x)
            x = self.activation(x)
            x = self.rec_conv(x) + prev_x
            x = self.activation(x)
            prev_x = self.pool(x)

        x = torch.flatten(prev_x, start_dim=1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x
