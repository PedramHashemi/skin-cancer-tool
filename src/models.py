"""Models for image classification."""

import torch.nn as nn
from torchvision import models

class TailModel(nn.Module):
    def __init__(self, num_classes: int, dropout: float):
        super(TailModel, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Freeze backbone (optional)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.linear1 = nn.Linear(self.backbone.fc.in_features, 64)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(64, num_classes)
                           
    def forward(self, x):
        x = self.backbone(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x