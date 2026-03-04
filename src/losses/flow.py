import torch
from torch import nn
from torch.nn import functional as F

class FlowConsistencyLoss(nn.Module):
    """
    Penalizes inconsistencies between predicted masks at consecutive frames 
    after accounting for motion via optical flow.
    """
    def __init__(self, loss_type='l1'):
        super().__init__()
        self.loss_type = loss_type

    def _warp(self, x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp an image or mask `x` using the optical `flow`.
        Args:
            x: Tensor of shape (B, C, H, W)
            flow: optical flow of shape (B, 2, H, W) - (dx, dy) in pixel coordinates
        """
        B, C, H, W = x.size()
        
        # Create a meshgrid of pixel coordinates
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H, device=x.device), 
                                        torch.arange(0, W, device=x.device), 
                                        indexing='ij')
        
        # Normalize flow from pixel coordinates to [-1, 1] range for grid_sample
        norm_flow_x = 2.0 * flow[:, 0] / max(W - 1, 1)
        norm_flow_y = 2.0 * flow[:, 1] / max(H - 1, 1)
        
        # Base grid + flow mappings normalized to [-1, 1]
        grid_x = (grid_x.expand(B, -1, -1).float() / (W - 1)) * 2.0 - 1.0 + norm_flow_x
        grid_y = (grid_y.expand(B, -1, -1).float() / (H - 1)) * 2.0 - 1.0 + norm_flow_y
        
        # Stack to (B, H, W, 2)
        grid = torch.stack((grid_x, grid_y), dim=3)
        
        # Interpolate the pixels
        warped_x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return warped_x

    def forward(self, mask_logits, flow, frame_mask=None):
        """
        Args:
            mask_logits: (B, T, C, H, W)
            flow: (B, T-1, 2, H, W) representing pixel-wise flow from t to t+1
            frame_mask: (B, T) Optional validity mask to ignore padded frames
        """
        B, T, C, H, W = mask_logits.shape
        if T < 2:
            return torch.tensor(0.0, device=mask_logits.device)

        # Convert logits to probabilities
        soft_masks = torch.sigmoid(mask_logits)
        
        mask_t = soft_masks[:, :-1].reshape(-1, C, H, W)         # Frames 0 to T-2
        mask_t_plus_1 = soft_masks[:, 1:].reshape(-1, C, H, W)   # Frames 1 to T-1
        
        flow_flat = flow.reshape(-1, 2, H, W)
        
        # Warp mask_t forwards to match the position at t+1
        warped_mask_t = self._warp(mask_t, flow_flat)
        
        if self.loss_type == 'l1':
            loss = F.l1_loss(warped_mask_t, mask_t_plus_1, reduction='none')
        elif self.loss_type == 'l2':
            loss = F.mse_loss(warped_mask_t, mask_t_plus_1, reduction='none')
        else:
            raise ValueError(f"Unknown loss type {self.loss_type}")
            
        # Mask out padded frames if a frame mask is provided
        if frame_mask is not None:
            # We care about transitions where BOTH t and t+1 are valid
            valid_transitions = (frame_mask[:, :-1] > 0.5) & (frame_mask[:, 1:] > 0.5)
            valid_flat = valid_transitions.reshape(-1, 1, 1, 1)
            loss = loss * valid_flat
            if valid_flat.sum() > 0:
                return loss.sum() / valid_flat.sum()
            return torch.tensor(0.0, device=loss.device)
            
        return loss.mean()
