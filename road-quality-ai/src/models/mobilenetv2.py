# models/mobilenetv2.py
import torch.nn as nn
from torchvision import models


def get_mobilenetv2(num_classes, pretrained=True):
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # Replace the final classifier layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model