import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class MAE:
    """Computes Mean Absolute Error via buffered predictions."""

    def __init__(self, reduction="mean"):
        self.reduction = reduction
        self.reset()

    def reset(self):
        self.preds = []
        self.targets = []

    def __call__(self, preds, targets):
        if isinstance(preds, torch.Tensor):
            preds = preds.float().detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.float().detach().cpu().numpy()

        self.preds.extend(preds.reshape(-1).tolist())
        self.targets.extend(targets.reshape(-1).tolist())

    def aggregate(self):
        if not self.preds:
            return 0.0
        return mean_absolute_error(self.targets, self.preds)


class RMSE:
    """Computes Root Mean Squared Error via buffered predictions."""

    def __init__(self, reduction="mean"):
        self.reduction = reduction
        self.reset()

    def reset(self):
        self.preds = []
        self.targets = []

    def __call__(self, preds, targets):
        if isinstance(preds, torch.Tensor):
            preds = preds.float().detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.float().detach().cpu().numpy()

        self.preds.extend(preds.reshape(-1).tolist())
        self.targets.extend(targets.reshape(-1).tolist())

    def aggregate(self):
        if not self.preds:
            return 0.0
        return np.sqrt(mean_squared_error(self.targets, self.preds))


class R2Score:
    """Computes the Coefficient of Determination (R² Score)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.preds = []
        self.targets = []

    def __call__(self, preds, targets):
        if isinstance(preds, torch.Tensor):
            preds = preds.float().detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.float().detach().cpu().numpy()

        self.preds.extend(preds.reshape(-1).tolist())
        self.targets.extend(targets.reshape(-1).tolist())

    def aggregate(self):
        if not self.preds:
            return 0.0
        return r2_score(self.targets, self.preds)
