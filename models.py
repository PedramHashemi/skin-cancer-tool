"""Models for image classification."""

from torchvision import models
import torch.nn as nn


def get_model(num_classes: int):
    """Generate the model.

    Args:
        num_classes (int): number of classes for the classification task.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(64, num_classes),
    )
    return model