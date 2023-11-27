import torch.nn as nn


class FC(nn.Module):
    def __init__(self, input_size, output_size=10, hidden_size=256):
        super(FC, self).__init__()
        self.input_size = input_size
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = self.hidden(x)
        out = self.relu(out)
        out = self.output(out)
        return out
