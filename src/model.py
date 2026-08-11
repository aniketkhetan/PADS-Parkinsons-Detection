"""MLP classifier over MiniRocket features."""

import torch.nn as nn


class ROCKETMLP(nn.Module):
    def __init__(self, input_dim, num_classes=6, dropout=0.3):
        super(ROCKETMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # no activation; CrossEntropyLoss applies softmax
        return x
