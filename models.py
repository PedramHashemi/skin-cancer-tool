"""Models for image classification."""

import logging
from torchvision import models
import torch.nn as nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_model(num_classes: int, dropout: float):
    """Generate the model.

    Args:
        num_classes (int): number of classes for the classification task.
    """
    logger.info("---> Starting model. Restnet50")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    logger.info("Freezing the parameters.")
    for param in model.parameters():
        param.requires_grad = False

    logger.info("Adding layers to the end of the model.")
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(64, num_classes),
    )

    logger.info("<--- Exiting the get Model. successfully created the model.")
    return model