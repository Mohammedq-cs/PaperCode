import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = self.fc1(x)
        return out