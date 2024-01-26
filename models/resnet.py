import torch
from torchvision import models


class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self, resnet_model: torch.nn.Module):
        super().__init__()
        self.features = torch.nn.Sequential(*list(resnet_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ResNet18FeatureExtractor(ResNetFeatureExtractor):
    def __init__(self):
        super().__init__(models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1))
