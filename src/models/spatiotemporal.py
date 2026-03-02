import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.logging import get_logger
from src.registry import register_model, get_model_class

logger = get_logger()

class SpatiotemporalDecoder(nn.Module):
    """
    3D U-Net style decoder to upsample spatiotemporal features back to the required mask resolution.
    
    Args:
        in_channels (int): Input feature channels.
        out_channels (int): Output feature channels (e.g., number of semantic classes).
        hidden_dims (list[int]): Feature channels for each transposed convolution layer.
    """
    def __init__(self, in_channels, out_channels=1, hidden_dims=[128, 64, 32, 16]):
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

class SpatialProjectionAdapter(nn.Module):
    """
    For spatial feature maps (e.g. EchoPrime): (B,C,T,H,W) -> (B,C_shared,T,H_out,W_out)
    """
    def __init__(self, in_channels, out_channels, out_time, out_spatial=(14, 14)):
        super().__init__()
        self.out_time = out_time
        self.out_spatial = out_spatial

        self.temporal_smooth = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=in_channels
        )
        self.channel_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # interpolate time+space to target grid
        x = F.interpolate(
            x,
            size=(self.out_time, self.out_spatial[0], self.out_spatial[1]),
            mode="trilinear",
            align_corners=False,
        )
        x = self.temporal_smooth(x)
        x = self.channel_proj(x)
        return x

class GlobalContextAdapter(nn.Module):
    """
    For global streams (e.g. PanEcho): (B,C,T,1,1) -> context (B,C_shared,T)
    No fake spatial upsampling.
    """
    def __init__(self, in_channels, out_channels, out_time):
        super().__init__()
        self.out_time = out_time

        # time alignment + smoothing in (B,C,T)
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.smooth = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, groups=out_channels)

    def forward(self, x):
        # x: (B,C,T,1,1) OR (B,C,T)
        if x.dim() == 5:
            x = x.squeeze(-1).squeeze(-1)  # (B,C,T)
        # align T
        if x.shape[-1] != self.out_time:
            x = F.interpolate(x, size=self.out_time, mode="linear", align_corners=False)

        x = self.proj(x)                  # (B,C_shared,T)
        x = x + self.smooth(x)            # depthwise temporal smoothing (residual)
        return x


class FiLMFromGlobal(nn.Module):
    """
    Uses global context (B,Cg,T) to modulate spatial tensor (B,Cs,T,H,W).
    """
    def __init__(self, global_channels, spatial_channels):
        super().__init__()
        self.to_gamma = nn.Conv1d(global_channels, spatial_channels, kernel_size=1)
        self.to_beta  = nn.Conv1d(global_channels, spatial_channels, kernel_size=1)

    def forward(self, spatial_x, global_ctx):
        # spatial_x: (B,C,T,H,W)
        # global_ctx: (B,Cg,T)
        gamma = self.to_gamma(global_ctx).unsqueeze(-1).unsqueeze(-1)  # (B,C,T,1,1)
        beta  = self.to_beta(global_ctx).unsqueeze(-1).unsqueeze(-1)   # (B,C,T,1,1)
        return spatial_x * (1.0 + torch.tanh(gamma)) + beta


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
    def __init__(self, fm_configs, fusion_space_cfg, peft_cfg=None, num_classes=1, num_phases=3):
        super().__init__()
        self.fm_names = [name for name, cfg in fm_configs.items() if cfg.get('enabled', False)]
        self.num_models = len(self.fm_names)
        
        self.shared_channels = fusion_space_cfg.get('channels', 256)
        self.out_time = fusion_space_cfg.get('time_steps', 16)
        self.out_spatial = tuple(fusion_space_cfg.get('spatial_size', (14, 14)))
        
        # 0. FM Backbone Wrappers
        self.backbones = nn.ModuleDict()
        backbone_registry = {
            'panecho': 'PanEchoWrapper',
            'echoprime': 'EchoPrimeWrapper',
        }
        for name in self.fm_names:
            registry_name = backbone_registry.get(name)
            if registry_name is None:
                logger.warning(f"No backbone wrapper registered for FM '{name}', skipping.")
                continue
            backbone_cls = get_model_class(registry_name)
            backbone = backbone_cls.from_config({'model': {'peft': peft_cfg, 'max_clip_len': self.out_time}})
            self.backbones[name] = backbone
            logger.info(f"Loaded FM backbone: {registry_name}")
        
        # 1. Dynamic Adapter Factory
        self.adapters = nn.ModuleDict()
        self.fm_out_spatial = {}

        for name in self.fm_names:
            fm_cfg = fm_configs[name]
            in_channels = fm_cfg.get('out_channels')
            if in_channels is None:
                raise ValueError(f"Config for foundation model {name} must specify out_channels.")

            out_spatial = tuple(fm_cfg.get("out_spatial", (7, 7)))
            self.fm_out_spatial[name] = out_spatial

            # PanEcho config has out_spatial [1,1] in your YAML :contentReference[oaicite:2]{index=2}
            if out_spatial == (1, 1):
                self.adapters[name] = GlobalContextAdapter(
                    in_channels=in_channels,
                    out_channels=self.shared_channels,
                    out_time=self.out_time
                )
            else:
                self.adapters[name] = SpatialProjectionAdapter(
                    in_channels=in_channels,
                    out_channels=self.shared_channels,
                    out_time=self.out_time,
                    out_spatial=self.out_spatial
                )

        # global->spatial conditioning
        self.film = FiLMFromGlobal(global_channels=self.shared_channels, spatial_channels=self.shared_channels)
            
        # 2. Voxel-Wise Router
        self.router = MoFMRouter(
            num_models=self.num_models,
            shared_channels=self.shared_channels,
            hidden_dim=64
        )
        
        # 3. Task Heads
        self.decoder = SpatiotemporalDecoder(in_channels=self.shared_channels, out_channels=num_classes)
        self.volume_head = DirectVolumeRegressor(in_channels=self.shared_channels, hidden_dim=512)

    def _extract_fm_features(self, video):
        fm_features = {}
        for name in self.fm_names:
            if name in self.backbones:
                fm_features[name] = self.backbones[name](video)
        return fm_features

    def forward(self, video):
        """
        End-to-end forward: raw video → FM extraction → fusion → task heads.

        Args:
            video (Tensor): Raw video tensor (B, C, T, H, W).
        """
        fm_features = self._extract_fm_features(video)
        spatial_feats = []
        global_ctxs = []

        spatial_names = []
        global_names = []

        for name in self.fm_names:
            raw_feat = fm_features[name]
            out_spatial = self.fm_out_spatial.get(name, (7, 7))

            if out_spatial == (1, 1):
                ctx = self.adapters[name](raw_feat)
                global_ctxs.append(ctx)
                global_names.append(name)
            else:
                s = self.adapters[name](raw_feat)
                spatial_feats.append(s)
                spatial_names.append(name)

        if len(spatial_feats) == 0:
            raise RuntimeError("No spatial foundation model features available (need at least one like EchoPrime).")

        if len(spatial_feats) == 1:
            fused_spatial = spatial_feats[0]
            router_weights = None
        else:
            router_weights = self.router(spatial_feats)  # (B, N_spatial, T, H, W)
            fused_spatial = torch.zeros_like(spatial_feats[0])
            for i, s in enumerate(spatial_feats):
                w = router_weights[:, i:i+1]
                fused_spatial = fused_spatial + s * w

        if len(global_ctxs) > 0:
            global_ctx = torch.stack(global_ctxs, dim=0).mean(dim=0)  # (B,C_shared,T)
            fused_features = self.film(fused_spatial, global_ctx)
        else:
            fused_features = fused_spatial

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
            "router_weights": router_weights,
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
        
        peft_cfg = model_cfg.get('peft')
            
        return cls(
            fm_configs=fm_configs,
            fusion_space_cfg=fusion_space_cfg,
            peft_cfg=peft_cfg,
            num_classes=cfg.get('data', {}).get('num_classes', 1),
            num_phases=model_cfg.get('num_phases', 3)
        )