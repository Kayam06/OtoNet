# model_mobilenet.py

import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def get_model(num_classes, pretrained=True, freeze_features=True, dropout_rate=0.3):
    """
    Loads a MobileNetV2 model and customizes the classifier.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Use ImageNet pretrained weights if True.
        freeze_features (bool): If True, freezes base model features to prevent overfitting.
        dropout_rate (float): Dropout rate for classifier.

    Returns:
        model (torch.nn.Module): Modified MobileNetV2 model.
    """
    # Load pretrained MobileNetV2
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)

    # freeze the base features
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(model.last_channel, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(512, num_classes),
    )

    return model
