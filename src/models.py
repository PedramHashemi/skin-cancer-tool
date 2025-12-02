"""Models for image classification."""

import torch.nn as nn

def TailModel():
    def __init__(self, in_features: int, num_classes: int, dropout: float):
        super(TailModel, self).__init__()
        self.linear1 = nn.Linear(in_features, 64)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(64, num_classes)
                           
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x