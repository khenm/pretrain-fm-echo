import torch
import torch.nn as nn
from src.registry import register_loss

@register_loss("CrossEntropy")
class CrossEntropyLossWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(**kwargs)

    def forward(self, preds, targets):
        return self.criterion(preds, targets)

@register_loss("MSE")
class MSELossWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.criterion = nn.MSELoss(**kwargs)

    def forward(self, preds, targets):
        return self.criterion(preds, targets)
