"""Models for image classification."""

from torchvision import models
from torch.nn import nn


def get_model(num_classes: int):
    """Generate the model.

    Args:
        num_classes (int): number of classes for the classification task.
    """
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(16, num_classes),
    )
    return model