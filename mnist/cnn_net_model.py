import torch
import torch.nn as nn


class CnnNet(nn.Module):

    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 5, stride=2, padding=2)

        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 5, stride=2, padding=2)

        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout2d(p=0.4)
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm1d(128)

        self.activation = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.norm1(x)
        x = self.dropout(x)

        x = self.activation(self.conv4(x))
        x = self.activation(self.conv5(x))
        x = self.activation(self.conv6(x))
        x = self.norm2(x)
        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.fc1(x))
        x = self.norm3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        x = self.softmax(x)
        return x
