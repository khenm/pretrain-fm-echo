import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from src.registry import register_model

@register_model("resnet18")
class ResNet18Model(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    @classmethod
    def from_config(cls, cfg):
        model_cfg = cfg.get('model', {})
        num_classes = cfg.get('data', {}).get('num_classes', 10)
        pretrained = model_cfg.get('pretrained', True)
        return cls(num_classes=num_classes, pretrained=pretrained)
