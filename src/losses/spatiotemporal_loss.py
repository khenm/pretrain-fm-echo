import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss
from src.losses.flow import FlowConsistencyLoss
from src.losses.curvature import CurvatureLoss

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
        curvature_weight: float = 1.0,
        volume_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            import logging
            logging.getLogger().warning(f"SpatiotemporalLoss ignoring unexpected kwargs: {list(kwargs.keys())}")
        self.dice_weight = dice_weight
        self.flow_weight = flow_weight
        self.curvature_weight = curvature_weight
        self.volume_weight = volume_weight
        self.dice_func = DiceCELoss(sigmoid=True, reduction='mean')
        self.flow_func = FlowConsistencyLoss(loss_type='l2')
        self.curvature_func = CurvatureLoss()

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Contains 'mask_logits'
            targets (dict): Raw dataloader batch containing 'label', 'frame_mask'
        """
        mask_logits = outputs['mask_logits']
        target_masks = targets['label']
        label_mask = targets['frame_mask']
        volume = outputs['volume']

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

        if self.curvature_weight > 0:
            loss_curvature = self.curvature_func(volume)
            total_loss += self.curvature_weight * loss_curvature
            loss_dict['curvature_loss'] = loss_curvature.detach()

        if self.volume_weight > 0:
            target_edv = targets.get('target_edv')
            target_esv = targets.get('target_esv')
            
            if target_edv is not None or target_esv is not None:
                loss_vol = 0.0
                valid_vol_samples = 0
                
                for b in range(volume.shape[0]):
                    b_frame_mask = label_mask[b]
                    b_vol = volume[b, :, 0] if volume.shape[-1] == 1 else volume[b].view(-1)
                    
                    if target_edv is not None:
                        ed_idx = torch.where(b_frame_mask == 2.0)[0]
                        if len(ed_idx) > 0 and target_edv[b] >= 0:
                            loss_vol += F.l1_loss(b_vol[ed_idx].mean(), target_edv[b])
                            valid_vol_samples += 1
                            
                    if target_esv is not None:
                        es_idx = torch.where(b_frame_mask == 1.0)[0]
                        if len(es_idx) > 0 and target_esv[b] >= 0:
                            loss_vol += F.l1_loss(b_vol[es_idx].mean(), target_esv[b])
                            valid_vol_samples += 1
                            
                if valid_vol_samples > 0:
                    loss_vol = loss_vol / valid_vol_samples
                    total_loss += self.volume_weight * loss_vol
                    loss_dict['volume_loss'] = loss_vol.detach()

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
            curvature_weight=loss_cfg.get("curvature_weight", 1.0),
            volume_weight=loss_cfg.get("volume_weight", 1.0)
        )
