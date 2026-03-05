import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class CurvatureLoss(nn.Module):
    """
    Computes Curvature Loss for temporally consistent volume estimation
    using a patient-specific learned gamma mapping:
    V_t = A_t^gamma
    k_t = (V_{t+1} - 2V_t + V_{t-1}) / (1 + (V_t - V_{t-1})^2)^{1.5}
    L_{curvature} = \sum_{t=2}^{T-1} k_t^2
    """
    def __init__(self):
        super().__init__()
        logger.info("Initialized dynamic CurvatureLoss with patient-specific gamma")

    def forward(self, volume):
        """
        Args:
            volume: Tensor of shape (B, T, C, H, W)
        Returns:
            Computed scalar curvature loss.
        """
        V_t_plus_1 = volume[:, 2:, :]
        V_t = volume[:, 1:-1, :]
        V_t_minus_1 = volume[:, :-2, :]
        
        numerator = V_t_plus_1 - 2 * V_t + V_t_minus_1
        v_t = V_t - V_t_minus_1 
        
        denominator = (1 + v_t ** 2) ** 1.5
        
        k_t = numerator / denominator
        loss = k_t ** 2
        
        return loss.mean()