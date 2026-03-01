import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.logging import get_logger
from src.registry import register_model

logger = get_logger()

class SpatiotemporalDecoder(nn.Module):
    """
    3D U-Net style decoder to upsample spatiotemporal features back to the required mask resolution.
    
    Args:
        in_channels (int): Input feature channels.
        out_channels (int): Output feature channels (e.g., number of semantic classes).
        hidden_dims (list[int]): Feature channels for each transposed convolution layer.
    """
    def __init__(self, in_channels, out_channels=1, hidden_dims=[256, 128, 64, 32, 16]):
        super().__init__()
        layers = []
        current_channels = in_channels
        
        for i, h_dim in enumerate(hidden_dims):
            if i == 0:
                k_size = (4, 4, 4)
                stride = (2, 2, 2)
            else:
                k_size = (3, 4, 4)
                stride = (1, 2, 2)
                
            layers.append(
                nn.ConvTranspose3d(
                    current_channels, h_dim, 
                    kernel_size=k_size, stride=stride, padding=(1, 1, 1)
                )
            )
            layers.append(nn.BatchNorm3d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            current_channels = h_dim
            
        layers.append(
            nn.Conv3d(current_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, features):
        """
        Upsamples capabilities for the spatiotemporal mask.
        
        Args:
            features (Tensor): (batch_size, channels, time, height, width)
            
        Returns:
            Tensor: Upsampled spatiotemporal mask logits.
        """
        return self.decoder(features)

import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectVolumeRegressor(nn.Module):
    """
    A minimal, high-capacity 1D sequence engine to regress 
    continuous volume directly from spatiotemporal features.
    """
    def __init__(self, in_channels=256, hidden_dim=512):
        super().__init__()
        
        # Squeeze channel projection (in_channels * 2 because of Avg + Max pool concat)
        self.channel_proj = nn.Conv1d(in_channels * 2, hidden_dim, kernel_size=1)
        
        # 1D ConvNeXt-style Temporal Block
        self.temporal_engine = nn.Sequential(
            # Depthwise large-kernel convolution over time
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim),
            nn.GroupNorm(1, hidden_dim), 
            
            # Pointwise inverted bottleneck
            nn.Conv1d(hidden_dim, hidden_dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim * 4, hidden_dim, kernel_size=1)
        )
        
        # Final projection to a single scalar volume per frame
        self.regressor = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, fused_features, mask_logits):
        """
        Args:
            fused_features (Tensor): (B, C, T, H, W)
            mask_logits (Tensor): (B, 1, T, H_out, W_out)
            
        Returns:
            Tensor: Continuous volume curve (B, T)
        """
        mask_prob = torch.sigmoid(mask_logits)
        _, _, time_f, h_f, w_f = fused_features.shape

        if mask_prob.shape[-2:] != (h_f, w_f):
            mask_prob_s = F.adaptive_avg_pool3d(mask_prob, output_size=(time_f, h_f, w_f))
        else:
            mask_prob_s = mask_prob

        # Multiply to ensure gradients flow EF -> Volume -> Segmentation Masks
        masked_features = fused_features * mask_prob_s
        
        # 1. Dual-Pooling Spatial Squeeze
        avg_pool = masked_features.mean(dim=(3, 4)) # (B, C, T)
        max_pool = masked_features.amax(dim=(3, 4)) # (B, C, T)
        
        # 2. Sequence Projection
        x_seq = torch.cat([avg_pool, max_pool], dim=1) # (B, 2C, T)
        x_seq = self.channel_proj(x_seq)               # (B, hidden_dim, T)
        
        # 3. High-Capacity Temporal Modeling (with residual connection)
        x_seq = x_seq + self.temporal_engine(x_seq)
        
        # 4. Direct Frame-by-Frame Regression
        vol_curve = self.regressor(x_seq).squeeze(1)   # (B, T)
        
        return vol_curve

class ProjectionAdapter(nn.Module):
    """
    Dynamically projects heterogeneous Foundation Model features into a
    Universal Geometric Latent Space using spatial-temporal interpolation,
    temporal smoothing, and channel projection.
    """
    def __init__(self, in_channels, out_channels, out_time, out_spatial=(14, 14)):
        super().__init__()
        self.out_time = out_time
        self.out_spatial = out_spatial
        
        # NEW: Depthwise 1D temporal convolution to smooth interpolated frames
        self.temporal_smooth = nn.Conv3d(
            in_channels, in_channels, 
            kernel_size=(3, 1, 1), 
            padding=(1, 0, 0), 
            groups=in_channels
        )
        self.channel_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 1. Spatial-Temporal Interpolation
        x_interp = F.interpolate(
            x, 
            size=(self.out_time, self.out_spatial[0], self.out_spatial[1]),
            mode='trilinear', 
            align_corners=False
        )
        # 2. Temporal Smoothing & Channel Projection
        x_smoothed = self.temporal_smooth(x_interp)
        return self.channel_proj(x_smoothed)


class MoFMRouter(nn.Module):
    """
    Generalized Context-Aware Collaborative Router for N Foundation Models.
    Uses a 3x3x3 receptive field to ensure spatially coherent model assignment.
    """
    def __init__(self, num_models, shared_channels, hidden_dim=64):
        super().__init__()
        self.num_models = num_models
        concat_channels = num_models * shared_channels
        
        self.net = nn.Sequential(
            # Channel reduction
            nn.Conv3d(concat_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # NEW: 3x3x3 Depthwise convolution for anatomical context
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # Final routing weights
            nn.Conv3d(hidden_dim, num_models, kernel_size=1)
        )

    def forward(self, projected_features):
        concat_features = torch.cat(projected_features, dim=1)
        logits = self.net(concat_features) # (B, N, T, H, W)
        return F.softmax(logits, dim=1)


@register_model("SpatiotemporalEchoModel")
class SpatiotemporalEchoModel(nn.Module):
    """
    Unified 3D model utilizing a Mixture of Foundation Models (MoFM) architecture.
    Projects arbitrary backbones into a universal space and fuses them dynamically.
    """
    def __init__(self, fm_configs, fusion_space_cfg, num_classes=1, num_phases=3):
        super().__init__()
        self.fm_names = [name for name, cfg in fm_configs.items() if cfg.get('enabled', False)]
        self.num_models = len(self.fm_names)
        
        self.shared_channels = fusion_space_cfg.get('channels', 256)
        self.out_time = fusion_space_cfg.get('time_steps', 16)
        self.out_spatial = tuple(fusion_space_cfg.get('spatial_size', (14, 14)))
        
        # 1. Dynamic Adapter Factory
        self.adapters = nn.ModuleDict()
        for name in self.fm_names:
            fm_cfg = fm_configs[name]
            in_channels = fm_cfg.get('out_channels')
            if in_channels is None:
                raise ValueError(f"Config for foundation model {name} must specify out_channels.")
            
            self.adapters[name] = ProjectionAdapter(
                in_channels=in_channels,
                out_channels=self.shared_channels,
                out_time=self.out_time,
                out_spatial=self.out_spatial
            )
            
        # 2. Voxel-Wise Router
        self.router = MoFMRouter(
            num_models=self.num_models,
            shared_channels=self.shared_channels,
            hidden_dim=64
        )
        
        # 3. Task Heads
        self.decoder = SpatiotemporalDecoder(in_channels=self.shared_channels, out_channels=num_classes)
        self.volume_head = DirectVolumeRegressor(in_channels=self.shared_channels, hidden_dim=512)

    def forward(self, fm_features: dict):
        """
        Maps multi-modal outputs into a unified space, routes them voxel-wise, and decodes.

        Args:
            fm_features (dict[str, Tensor]): Dict mapping FM names to their specific tensors.
        """
        projected_fms = []
        for name in self.fm_names:
            if name not in fm_features:
                raise KeyError(f"Expected foundation model {name} missing from inputs.")
            raw_feat = fm_features[name]
            proj_feat = self.adapters[name](raw_feat)
            projected_fms.append(proj_feat)
            
        # Voxel-wise Router predicts distribution over models
        router_weights = self.router(projected_fms) # (B, N, T, H, W)
        
        # Compute fused representation F_final = \sum_i W_i * X_hat_i
        fused_features = torch.zeros_like(projected_fms[0])
        for i, proj_feat in enumerate(projected_fms):
            w_i = router_weights[:, i:i+1, ...] # Extract weight for i-th model, keep dim=1 empty
            fused_features = fused_features + (proj_feat * w_i)

        # Standard processing on uniform space
        mask_logits = self.decoder(fused_features)
        vol_curve = self.volume_head(fused_features, mask_logits)

        pred_edv = vol_curve.max(dim=1)[0]
        pred_esv = vol_curve.min(dim=1)[0]
        
        pred_ef = (pred_edv - pred_esv) / pred_edv.clamp(min=1e-3)

        return {
            "mask_logits": mask_logits,
            "vol_curve": vol_curve,
            "pred_edv": pred_edv,
            "pred_esv": pred_esv,
            "pred_ef": pred_ef,
            "router_weights": router_weights
        }

    @classmethod
    def from_config(cls, cfg):
        model_cfg = cfg.get('model', {})
        fm_configs = model_cfg.get('foundation_models', {})
        fusion_space_cfg = model_cfg.get('fusion_space', {})
        
        if not fm_configs:
            fm_configs = {
                "panecho": {"enabled": True, "out_channels": model_cfg.get('pan_echo_dim', 768)},
                "echoprime": {"enabled": True, "out_channels": model_cfg.get('echo_prime_dim', 768)}
            }
        
        if not fusion_space_cfg:
            fusion_space_cfg = {
                "channels": model_cfg.get('fused_dim', 256),
                "time_steps": model_cfg.get('max_clip_len', 16),
                "spatial_size": (14, 14)
            }
            
        return cls(
            fm_configs=fm_configs,
            fusion_space_cfg=fusion_space_cfg,
            num_classes=cfg.get('data', {}).get('num_classes', 1),
            num_phases=model_cfg.get('num_phases', 3)
        )


@register_model("SpatiotemporalEchoModel")
class SpatiotemporalEchoModel(nn.Module):
    """
    Unified 3D model utilizing a Mixture of Foundation Models (MoFM) architecture.
    Projects arbitrary backbones into a universal space and fuses them dynamically.
    """
    def __init__(self, fm_configs, fusion_space_cfg, num_classes=1, num_phases=3, temperature=10.0):
        super().__init__()
        self.fm_names = [name for name, cfg in fm_configs.items() if cfg.get('enabled', False)]
        self.num_models = len(self.fm_names)
        self.temperature = temperature # Alpha parameter for Soft-Extrema
        
        self.shared_channels = fusion_space_cfg.get('channels', 256)
        self.out_time = fusion_space_cfg.get('time_steps', 16)
        self.out_spatial = tuple(fusion_space_cfg.get('spatial_size', (14, 14)))
        
        # 1. Dynamic Adapter Factory
        self.adapters = nn.ModuleDict()
        for name in self.fm_names:
            fm_cfg = fm_configs[name]
            in_channels = fm_cfg.get('out_channels')
            if in_channels is None:
                raise ValueError(f"Config for foundation model {name} must specify out_channels.")
            
            self.adapters[name] = ProjectionAdapter(
                in_channels=in_channels,
                out_channels=self.shared_channels,
                out_time=self.out_time,
                out_spatial=self.out_spatial
            )
            
        # 2. Voxel-Wise Router
        self.router = MoFMRouter(
            num_models=self.num_models,
            shared_channels=self.shared_channels,
            hidden_dim=64
        )
        
        # 3. Task Heads
        self.decoder = SpatiotemporalDecoder(in_channels=self.shared_channels, out_channels=num_classes)
        self.volume_head = DirectVolumeRegressor(in_channels=self.shared_channels, hidden_dim=512)

    def forward(self, fm_features: dict):
        projected_fms = []
        for name in self.fm_names:
            if name not in fm_features:
                raise KeyError(f"Expected foundation model {name} missing from inputs.")
            raw_feat = fm_features[name]
            proj_feat = self.adapters[name](raw_feat)
            projected_fms.append(proj_feat)
            
        # Voxel-wise Router predicts distribution over models
        router_weights = self.router(projected_fms) # (B, N, T, H, W)
        
        # Compute fused representation
        fused_features = torch.zeros_like(projected_fms[0])
        for i, proj_feat in enumerate(projected_fms):
            w_i = router_weights[:, i:i+1, ...] 
            fused_features = fused_features + (proj_feat * w_i)

        # Standard processing on uniform space
        mask_logits = self.decoder(fused_features)
        vol_curve = self.volume_head(fused_features, mask_logits)

        weight_edv = F.softmax(vol_curve * self.temperature, dim=1)
        pred_edv = torch.sum(vol_curve * weight_edv, dim=1)
        
        # Soft-Min for ESV
        weight_esv = F.softmax(-vol_curve * self.temperature, dim=1)
        pred_esv = torch.sum(vol_curve * weight_esv, dim=1)
        
        # Ejection Fraction
        pred_ef = (pred_edv - pred_esv) / pred_edv.clamp(min=1e-3)

        return {
            "mask_logits": mask_logits,
            "vol_curve": vol_curve,
            "pred_edv": pred_edv,
            "pred_esv": pred_esv,
            "pred_ef": pred_ef,
            "router_weights": router_weights
        }

    @classmethod
    def from_config(cls, cfg):
        # ... (keep existing from_config logic) ...
        model_cfg = cfg.get('model', {})
        fm_configs = model_cfg.get('foundation_models', {})
        fusion_space_cfg = model_cfg.get('fusion_space', {})
        
        if not fm_configs:
            fm_configs = {
                "panecho": {"enabled": True, "out_channels": model_cfg.get('pan_echo_dim', 768)},
                "echoprime": {"enabled": True, "out_channels": model_cfg.get('echo_prime_dim', 768)}
            }
        
        if not fusion_space_cfg:
            fusion_space_cfg = {
                "channels": model_cfg.get('fused_dim', 256),
                "time_steps": model_cfg.get('max_clip_len', 16),
                "spatial_size": (14, 14)
            }
            
        return cls(
            fm_configs=fm_configs,
            fusion_space_cfg=fusion_space_cfg,
            num_classes=cfg.get('data', {}).get('num_classes', 1),
            num_phases=model_cfg.get('num_phases', 3),
            temperature=model_cfg.get('extrema_temperature', 10.0) # Added config hook
        )