import torch
import torch.nn as nn
import torch.nn.functional as F
from src.registry import register_loss

@register_loss("JointVolumeSegLoss")
class JointVolumeSegLoss(nn.Module):
    """
    Combines voxel-level segmentation loss with structural volume curve loss.
    Provides temporally consistent supervision to the 3D Segmentation Head.
    """
    def __init__(self, seg_weight=1.0, vol_weight=1.0):
        super().__init__()
        self.seg_weight = seg_weight
        self.vol_weight = vol_weight
        
        # Core losses
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, outputs, target_mask, target_vol):
        """
        outputs: dictionary containing "mask_logits" (B, 1, T, H, W) and "ln_vol_curve" (B, T)
        target_mask: true binary masks, shape (B, 1, T, H, W)
        target_vol: ground truth volume curves, shape (B, T). Missing/invalid frames should be 0 or NaN.
        """
        mask_logits = outputs["mask_logits"]
        pred_ln_vol = outputs["ln_vol_curve"]
        
        # 1) Spatiotemporal Mask Loss
        loss_seg = self.bce(mask_logits, target_mask.float())
        
        # 2) Geometric Volume Loss (MSE on log-volume at ED/ES frames)
        valid_mask = (target_vol > 0) & ~torch.isnan(target_vol)
        if valid_mask.sum() > 0:
            target_ln_vol = torch.log(torch.clamp(target_vol[valid_mask], min=1e-3))
            loss_vol = self.mse(pred_ln_vol[valid_mask], target_ln_vol.float())
        else:
            loss_vol = torch.tensor(0.0, device=pred_ln_vol.device)
        
        # Joint Optimization
        total_loss = (self.seg_weight * loss_seg) + (self.vol_weight * loss_vol)
        
        return {
            "loss": total_loss,
            "loss_seg": loss_seg.detach(),
            "loss_vol": loss_vol.detach()
        }

    @classmethod
    def from_config(cls, cfg):
        loss_cfg = cfg.get("loss", {})
        return cls(
            seg_weight=loss_cfg.get("seg_weight", 1.0),
            vol_weight=loss_cfg.get("vol_weight", 1.0)
        )
