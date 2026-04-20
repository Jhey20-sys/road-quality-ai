import torch.nn as nn
from torchvision import models

def get_resnet18(num_classes=4):
    model = models.resnet18(pretrained=True)

    # Freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
