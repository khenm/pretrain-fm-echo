import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss
from src.losses.flow import FlowConsistencyLoss

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
        flow_weight: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            import logging
            logging.getLogger().warning(f"SpatiotemporalLoss ignoring unexpected kwargs: {list(kwargs.keys())}")
        self.dice_weight = dice_weight
        self.flow_weight = flow_weight
        self.dice_func = DiceCELoss(sigmoid=True, reduction='mean')
        self.flow_func = FlowConsistencyLoss(loss_type='l2')

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Contains 'mask_logits'
            targets (dict): Raw dataloader batch containing 'label', 'frame_mask'
        """
        mask_logits = outputs['mask_logits']

        target_masks = targets['label']
        frame_mask = targets['frame_mask']

        loss_dice = self._compute_dice_loss(mask_logits, target_masks, frame_mask)

        total_loss = self.dice_weight * loss_dice

        loss_dict = {
            "loss": total_loss,
            "dice_loss": loss_dice.detach(),
        }

        if flow in targets and self.flow_weight > 0:
            loss_flow = self.flow_func(mask_logits, targets['flow'], frame_mask)
            total_loss += self.flow_weight * loss_flow
            loss_dict['flow_loss'] = loss_flow.detach()
            loss_dict['loss'] = total_loss

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
            dice_weight=loss_cfg.get("dice_weight", 1.0)
        )
