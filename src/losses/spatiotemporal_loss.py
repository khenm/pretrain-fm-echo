import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss

from src.registry import register_loss

@register_loss("SpatiotemporalLoss")
class SpatiotemporalLoss(nn.Module):
    """
    Computes spatial segmentation (DiceCE) and L1 regression losses 
    for Spatiotemporal echocardiography.
    """
    def __init__(
        self,
        dice_weight: float = 1.0,
        volume_weight: float = 1.0,
        edv_weight: float = 1.0,
        esv_weight: float = 1.0,
        ef_weight: float = 100.0,
        ef_weight_target: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            import logging
            logging.getLogger().warning(f"SpatiotemporalLoss ignoring unexpected kwargs: {list(kwargs.keys())}")
        self.dice_weight = dice_weight
        self.volume_weight = volume_weight
        
        self.edv_weight = edv_weight
        self.esv_weight = esv_weight
        self.ef_weight = ef_weight

        self.res_mag_weight = kwargs.get('res_mag_weight', 0.1)
        self.res_smooth_weight = kwargs.get('res_smooth_weight', 1.0)

        self.dice_func = DiceCELoss(sigmoid=True, reduction='mean')
        self.l1_loss = nn.L1Loss(reduction='none')

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Contains 'mask_logits', 'vol_curve', 'pred_edv', 'pred_esv', 'pred_ef'
            targets (dict): Raw dataloader batch containing 'label', 'target_edv', 'target_esv', 'target_ef', 'frame_mask'
        """
        mask_logits = outputs['mask_logits']

        target_masks = targets['label']
        target_edv = targets['target_edv']
        target_esv = targets['target_esv']
        target_ef = targets.get('target_ef')
        frame_mask = targets['frame_mask']

        loss_dice = self._compute_dice_loss(mask_logits, target_masks, frame_mask)

        pred_edv = outputs.get('pred_edv', torch.zeros_like(target_edv))
        pred_esv = outputs.get('pred_esv', torch.zeros_like(target_esv))
        pred_ef = outputs.get('pred_ef', torch.zeros_like(pred_edv))

        valid_edv_esv = (target_edv >= 0) & (target_esv >= 0)
        
        loss_edv = torch.tensor(0.0, device=pred_edv.device)
        loss_esv = torch.tensor(0.0, device=pred_esv.device)
        loss_ef = torch.tensor(0.0, device=pred_edv.device)

        if valid_edv_esv.any():
            loss_edv = self.l1_loss(pred_edv[valid_edv_esv], target_edv[valid_edv_esv]).mean()
            loss_esv = self.l1_loss(pred_esv[valid_edv_esv], target_esv[valid_edv_esv]).mean()
            
        if target_ef is not None:
            valid_ef = (target_ef >= 0)
            if valid_ef.any():
                loss_ef = self.l1_loss(pred_ef[valid_ef], target_ef[valid_ef]).mean()

        loss_vol = (self.edv_weight * loss_edv) + (self.esv_weight * loss_esv) + (self.ef_weight * loss_ef)
        
        vol_residual = outputs.get('vol_residual')
        loss_res_mag = torch.tensor(0.0, device=mask_logits.device)
        loss_res_smooth = torch.tensor(0.0, device=mask_logits.device)

        if vol_residual is not None:
            # 1. Keep the residual small (prevent it from overriding the mask)
            loss_res_mag = (vol_residual ** 2).mean()
            
            # 2. Keep the residual smooth (Second derivative penalty)
            if vol_residual.shape[1] >= 3:
                diff1 = vol_residual[:, 1:] - vol_residual[:, :-1]
                diff2 = diff1[:, 1:] - diff1[:, :-1]
                loss_res_smooth = diff2.abs().mean()

        total_loss = (self.dice_weight * loss_dice) + (self.volume_weight * loss_vol) + \
                     (self.res_mag_weight * loss_res_mag) + (self.res_smooth_weight * loss_res_smooth)

        loss_dict = {
            "loss": total_loss,
            "dice_loss": loss_dice.detach(),
            "volume_loss": loss_vol.detach(),
            "loss_edv": loss_edv.detach(),
            "loss_esv": loss_esv.detach(),
            "loss_ef": loss_ef.detach(),
            "loss_res_mag": loss_res_mag.detach(),
            "loss_res_smooth": loss_res_smooth.detach(),
        }

        return total_loss, loss_dict

    def _compute_dice_loss(self, pred_logits, target_masks, frame_mask):
        """Vectorized Dice+CE on valid labeled frames."""
        if pred_logits.shape[-2:] != target_masks.shape[-2:]:
            target_size = target_masks.shape[-2:]
            pred_logits = F.interpolate(
                pred_logits, size=(pred_logits.shape[2], *target_size),
                mode='trilinear', align_corners=False
            )

        if pred_logits.shape[1] == 1 and pred_logits.shape[2] > 1:
            pred_logits = pred_logits.permute(0, 2, 1, 3, 4)

        if target_masks.shape[1] == 1 and target_masks.shape[2] > 1:
            target_masks = target_masks.permute(0, 2, 1, 3, 4)

        batch_size, seq_len, channels, height, width = pred_logits.shape

        pred_flat = pred_logits.reshape(-1, channels, height, width)
        target_flat = target_masks.reshape(-1, channels, height, width)
        mask_flat = frame_mask.reshape(-1)

        valid_indices = mask_flat > 0.5

        if valid_indices.sum() == 0:
            return 0.0 * pred_logits.sum()

        return self.dice_func(
            pred_flat[valid_indices],
            target_flat[valid_indices]
        )

    @classmethod
    def from_config(cls, cfg):
        loss_cfg = cfg.get("loss", {})
        return cls(
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            volume_weight=loss_cfg.get("volume_weight", 1.0),
            edv_weight=loss_cfg.get("edv_weight", 1.0),
            esv_weight=loss_cfg.get("esv_weight", 1.0),
            ef_weight=loss_cfg.get("ef_weight", 100.0)
        )
