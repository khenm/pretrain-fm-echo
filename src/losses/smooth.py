import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class TemporalSmoothnessLoss(nn.Module):
    """
    Computes an Area-Weighted Smoothness Loss:
    L_smooth = 1/(T-1) \sum_{t=1}^{T-1} A_{t-1}(A_t - A_{t-1})^2
    where A_t is the mask area, normalized by the image shape.
    """
    def __init__(self):
        super().__init__()
        logger.info("Initialized TemporalSmoothnessLoss")

    def forward(self, mask_logits):
        """
        Args:
            mask_logits: Tensor of shape (B, T, C, H, W)
                Logits corresponding to the temporal masks.
        Returns:
            Computed scalar smoothness loss.
        """
        probs = torch.sigmoid(mask_logits)
        A = probs.mean(dim=(-2, -1))
        if A.shape[1] <= 1:
            return torch.tensor(0.0, device=A.device, dtype=A.dtype)
            
        A_t = A[:, 1:, :]
        A_t_minus_1 = A[:, :-1, :]
        loss = A_t_minus_1 * (A_t - A_t_minus_1) ** 2
        
        return loss.mean()
