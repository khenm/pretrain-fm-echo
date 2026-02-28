import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss

from src.registry import register_loss

class PolarFocalVolumeLoss(nn.Module):
    """
    A difficulty-aware orthogonal loss function. 
    Projects EDV/ESV into polar space (Magnitude/Angle) and applies independent 
    focal difficulty weighting to both the scale and the physiological ratio.
    """
    def __init__(
        self, 
        gamma: float = 2.0,
        scale_weight: float = 1.0, 
        ratio_weight: float = 10.0, 
        sv_weight: float = 1.0,
        clip_threshold: float = 0.5,
        eps: float = 1e-7
    ):
        super().__init__()
        self.gamma = gamma
        self.scale_weight = scale_weight
        self.ratio_weight = ratio_weight
        self.sv_weight = sv_weight
        self.clip_threshold = clip_threshold
        self.eps = eps

    def forward(
        self, 
        pred_edv: torch.Tensor, 
        pred_esv: torch.Tensor, 
        target_edv: torch.Tensor, 
        target_esv: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        
        p_edv, p_esv = pred_edv.view(-1), pred_esv.view(-1)
        t_edv, t_esv = target_edv.view(-1), target_esv.view(-1)
        
        valid = (t_edv >= 0) & (t_esv >= 0)
        
        if not valid.any():
            dummy_loss = 0.0 * p_edv.sum()
            return dummy_loss, {"scale_loss": dummy_loss.detach(), "ratio_loss": dummy_loss.detach(), "sv_loss": dummy_loss.detach()}

        pred_vec = torch.stack([p_edv[valid], p_esv[valid]], dim=-1)
        target_vec = torch.stack([t_edv[valid], t_esv[valid]], dim=-1)

        pred_mag = torch.norm(pred_vec, p=2, dim=-1)
        target_mag = torch.norm(target_vec, p=2, dim=-1)
        
        base_loss_scale = F.huber_loss(pred_mag, target_mag, reduction='none', delta=self.clip_threshold)
        
        error_scale = torch.abs(pred_mag - target_mag)
        relative_error_scale = error_scale / (target_mag + self.eps)
        p_scale = torch.clamp(relative_error_scale, min=0.0, max=1.0)
        weight_scale = (1.0 + torch.pow(p_scale, self.gamma)).detach()
        
        focal_loss_scale = (weight_scale * base_loss_scale).mean()

        cos_sim = F.cosine_similarity(pred_vec, target_vec, dim=-1, eps=self.eps) 
        base_loss_ratio = 1.0 - cos_sim
        
        p_ratio = torch.clamp(base_loss_ratio, min=0.0, max=1.0)
        weight_ratio = (1.0 + torch.pow(p_ratio, self.gamma)).detach()
        
        focal_loss_ratio = (weight_ratio * base_loss_ratio).mean()

        pred_sv = p_edv[valid] - p_esv[valid]
        target_sv = t_edv[valid] - t_esv[valid]
        
        base_loss_sv = F.huber_loss(pred_sv, target_sv, reduction='none', delta=self.clip_threshold)
        
        error_sv = torch.abs(pred_sv - target_sv)
        relative_error_sv = error_sv / (torch.abs(target_sv) + self.eps)
        p_sv = torch.clamp(relative_error_sv, min=0.0, max=1.0)
        weight_sv = (1.0 + torch.pow(p_sv, self.gamma)).detach()
        
        focal_loss_sv = (weight_sv * base_loss_sv).mean()

        total_loss = (self.scale_weight * focal_loss_scale) + \
                     (self.ratio_weight * focal_loss_ratio) + \
                     (self.sv_weight * focal_loss_sv)

        return total_loss, {
            "scale_loss": focal_loss_scale.detach(),
            "ratio_loss": focal_loss_ratio.detach(),
            "sv_loss": focal_loss_sv.detach()
        }

@register_loss("SpatiotemporalLoss")
class SpatiotemporalLoss(nn.Module):
    """
    Computes spatial segmentation (DiceCE) and volumetric regression losses (PolarFocalVolumeLoss) 
    for spatiotemporal echocardiography.
    """
    def __init__(
        self,
        dice_weight: float = 1.0,
        volume_weight: float = 1.0,
        phase_weight: float = 0.5,
        gamma: float = 2.0,
        focal_clip_threshold: float = 0.5,
        focal_scale_weight: float = 1.0,
        focal_ratio_weight: float = 10.0,
        focal_sv_weight: float = 1.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.volume_weight = volume_weight
        self.phase_weight = phase_weight

        self.dice_func = DiceCELoss(sigmoid=True, reduction='mean')
        self.phase_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        self.vol_loss_func = PolarFocalVolumeLoss(
            gamma=gamma,
            scale_weight=focal_scale_weight,
            ratio_weight=focal_ratio_weight,
            sv_weight=focal_sv_weight,
            clip_threshold=focal_clip_threshold
        )

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Contains 'mask_logits', 'vol_curve', 'phase_logits', 'pred_edv', 'pred_esv'
            targets (dict): Raw dataloader batch containing 'label', 'target_edv', 'target_esv', 'frame_mask'
        """
        mask_logits = outputs['mask_logits']
        vol_curve = outputs['vol_curve']

        target_masks = targets['label']
        target_edv = targets['target_edv']
        target_esv = targets['target_esv']
        frame_mask = targets['frame_mask']

        B, T = vol_curve.shape

        loss_dice = self._compute_dice_loss(mask_logits, target_masks, frame_mask)

        pred_edv = outputs.get('pred_edv', torch.zeros_like(target_edv))
        pred_esv = outputs.get('pred_esv', torch.zeros_like(target_esv))

        loss_vol, vol_loss_dict = self.vol_loss_func(
            pred_edv, pred_esv, target_edv, target_esv
        )

        total_loss = (self.dice_weight * loss_dice) + (self.volume_weight * loss_vol)

        # --- Phase Classification Loss ---
        loss_phase = torch.tensor(0.0, device=vol_curve.device)
        phase_logits = outputs.get('phase_logits')
        if phase_logits is not None and self.phase_weight > 0:
            phase_targets = frame_mask.long()
            _, num_phases, T_phase = phase_logits.shape

            if T_phase != phase_targets.shape[1]:
                phase_logits = F.interpolate(
                    phase_logits, 
                    size=phase_targets.shape[1], 
                    mode='linear', 
                    align_corners=False
                )

            loss_phase = self.phase_loss_fn(phase_logits, phase_targets)
            total_loss = total_loss + (self.phase_weight * loss_phase)

        loss_dict = {
            "loss": total_loss,
            "dice_loss": loss_dice.detach(),
            "volume_loss": loss_vol.detach(),
            "phase_loss": loss_phase.detach(),
        }

        if vol_loss_dict:
            loss_dict.update({k: v.detach() for k, v in vol_loss_dict.items()})

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
            phase_weight=loss_cfg.get("phase_weight", 0.5),
            gamma=loss_cfg.get("gamma", 2.0),
            focal_clip_threshold=loss_cfg.get("clip_threshold", 0.5),
            focal_scale_weight=loss_cfg.get("scale_weight", 1.0),
            focal_ratio_weight=loss_cfg.get("ratio_weight", 10.0),
            focal_sv_weight=loss_cfg.get("sv_weight", 1.0)
        )
