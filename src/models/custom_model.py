import torch.nn as nn
from torchvision.models import resnet18

from .base_model import BaseModel


class CustomNet(BaseModel):
    """A light-weight CNN based on ResNet18 for CIFAR-10 size images."""

    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        # Use torchvision's ResNet18 but tweak the first conv and fc
        self.backbone = resnet18(pretrained=pretrained)
        # Adjust for 32x32 images
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):  # type: ignore[override]
        return self.backbone(x) 