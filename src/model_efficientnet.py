# src/model.py

import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def get_model(num_classes, pretrained=True):
    """
    Loads EfficientNet-B0 and modifies the final classifier layer.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use pretrained ImageNet weights.

    Returns:
        model (torch.nn.Module): The modified EfficientNet-B0 model.
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    # Replace the final classification layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
