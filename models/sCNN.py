import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, inDim=3, outDim=10, flattenLayerValue=8 * 8):
        super(SimpleCNN, self).__init__()
        self.flattenLayerValue = flattenLayerValue
        self.conv1 = nn.Conv2d(inDim, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * flattenLayerValue, 256)  # 64 * 8 * 8
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 32, 32, 32
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 64, 8, 8
        x = x.view(-1, 64 * self.flattenLayerValue)  # -> n, 400
        x = F.relu(self.fc1(x))  # -> n, 120
        x = F.relu(self.fc2(x))  # -> n, 84
        x = self.fc3(x)  # -> n, 10
        return x
