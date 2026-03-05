import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss
from src.losses.flow import FlowConsistencyLoss
from src.losses.smooth import TemporalSmoothnessLoss

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
        smooth_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            import logging
            logging.getLogger().warning(f"SpatiotemporalLoss ignoring unexpected kwargs: {list(kwargs.keys())}")
        self.dice_weight = dice_weight
        self.flow_weight = flow_weight
        self.smooth_weight = smooth_weight
        self.dice_func = DiceCELoss(sigmoid=True, reduction='mean')
        self.flow_func = FlowConsistencyLoss(loss_type='l2')
        self.smooth_func = TemporalSmoothnessLoss()

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Contains 'mask_logits'
            targets (dict): Raw dataloader batch containing 'label', 'frame_mask'
        """
        mask_logits = outputs['mask_logits']

        target_masks = targets['label']
        label_mask = targets['frame_mask']

        if mask_logits.shape[-2:] != target_masks.shape[-2:]:
            target_size = target_masks.shape[-2:]
            mask_logits = F.interpolate(
                mask_logits, size=(mask_logits.shape[2], *target_size),
                mode='trilinear', align_corners=False
            )

        if mask_logits.shape[1] == 1 and mask_logits.shape[2] > 1:
            mask_logits = mask_logits.permute(0, 2, 1, 3, 4)

        loss_dice = self._compute_dice_loss(mask_logits, target_masks, label_mask)

        total_loss = self.dice_weight * loss_dice

        loss_dict = {
            "loss": total_loss,
            "dice_loss": loss_dice.detach(),
        }

        if self.smooth_weight > 0:
            loss_smooth = self.smooth_func(mask_logits)
            total_loss += self.smooth_weight * loss_smooth
            loss_dict['smooth_loss'] = loss_smooth.detach()
            loss_dict['loss'] = total_loss

        if 'flow' in targets and self.flow_weight > 0:
            flow_target = targets['flow']
            if flow_target.shape[-2:] != mask_logits.shape[-2:]:
                B, T_minus_1, C_flow, H_f, W_f = flow_target.shape
                flow_target = flow_target.view(B, T_minus_1 * C_flow, H_f, W_f)
                flow_target = F.interpolate(flow_target, size=mask_logits.shape[-2:], mode='bilinear', align_corners=False)
                scale_h = mask_logits.shape[-2] / H_f
                scale_w = mask_logits.shape[-1] / W_f
                flow_target = flow_target.view(B, T_minus_1, C_flow, *mask_logits.shape[-2:])
                flow_target[:, :, 0, :, :] *= scale_w
                flow_target[:, :, 1, :, :] *= scale_h

            loss_flow = self.flow_func(mask_logits, flow_target, None)
            total_loss += self.flow_weight * loss_flow
            loss_dict['flow_loss'] = loss_flow.detach()
            loss_dict['loss'] = total_loss

        return total_loss, loss_dict

    def _compute_dice_loss(self, pred_logits, target_masks, frame_mask):
        """Vectorized Dice+CE on valid labeled frames."""
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
            flow_weight=loss_cfg.get("flow_weight", 0.5),
            smooth_weight=loss_cfg.get("smooth_weight", 1.0)
        )
