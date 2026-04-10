"""
Structural Crack Detection - Model Definition
Uses ResNet50 pretrained on ImageNet with transfer learning
for binary classification: Cracked / Not Cracked
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes=2, freeze_backbone=True):
    """
    Build ResNet50 transfer learning model for crack detection.
    
    Args:
        num_classes: 2 (Cracked, Not Cracked)
        freeze_backbone: Freeze pretrained layers, only train classifier
    
    Returns:
        model: PyTorch model
    """
    # Load pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze backbone layers
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable


if __name__ == "__main__":
    model = build_model()
    count_parameters(model)
    print(f"\nModel architecture:\n{model.fc}")
