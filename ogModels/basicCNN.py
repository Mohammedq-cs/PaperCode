import torch.nn as nn


class BasicCNN(nn.Sequential):
    def __init__(self, in_channels, conv_channels, fc_nodes, fc_nodes_in, maxpool_param, num_classes):
        super(BasicCNN, self).__init__(
            nn.Conv2d(in_channels, conv_channels, 3),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, 3),
            nn.ReLU(),
            nn.MaxPool2d(maxpool_param),
            nn.Conv2d(conv_channels, conv_channels * 2, 3),
            nn.ReLU(),
            nn.Conv2d(conv_channels * 2, conv_channels * 2, 3),
            nn.ReLU(),
            nn.MaxPool2d(maxpool_param),
            nn.Flatten(),
            nn.Linear(fc_nodes_in, fc_nodes),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_nodes, fc_nodes),
            nn.ReLU(),
            nn.Linear(fc_nodes, num_classes)
        )
