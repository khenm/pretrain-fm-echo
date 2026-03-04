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

class FiLMFromGlobal(nn.Module):
    """
    Uses global context (B,Cg,T) to modulate spatial tensor (B,Cs,T,H,W).
    Upgraded with Zero-Initialization to guarantee safe early-epoch training.
    """
    def __init__(self, global_channels, spatial_channels):
        super().__init__()
        self.to_gamma = nn.Conv1d(global_channels, spatial_channels, kernel_size=1)
        self.to_beta  = nn.Conv1d(global_channels, spatial_channels, kernel_size=1)
        
        # Zero-initialize the modulation weights
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, spatial_x, global_ctx):
        gamma = self.to_gamma(global_ctx).unsqueeze(-1).unsqueeze(-1)  # (B,C,T,1,1)
        beta  = self.to_beta(global_ctx).unsqueeze(-1).unsqueeze(-1)   # (B,C,T,1,1)
        return spatial_x * (1.0 + torch.tanh(gamma)) + beta

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
        
        # 1. Inline Normalized Projections
        panecho_dim = fm_configs.get('panecho', {}).get('out_channels', 768)
        echoprime_dim = fm_configs.get('echoprime', {}).get('out_channels', 768)

        self.panecho_proj = nn.Sequential(
            nn.Conv1d(panecho_dim, self.shared_channels, kernel_size=1),
            nn.GroupNorm(num_groups=min(32, self.shared_channels), num_channels=self.shared_channels),
            nn.ReLU(inplace=True)
        )
        
        self.echoprime_proj = nn.Sequential(
            nn.Conv3d(echoprime_dim, self.shared_channels, kernel_size=1),
            nn.GroupNorm(num_groups=min(32, self.shared_channels), num_channels=self.shared_channels),
            nn.ReLU(inplace=True)
        )

        # global->spatial conditioning
        self.film = FiLMFromGlobal(global_channels=self.shared_channels, spatial_channels=self.shared_channels)
        
        self.decoder = SpatiotemporalDecoder(in_channels=self.shared_channels, out_channels=num_classes)

    def _extract_fm_features(self, video):
        fm_features = {}
        for name in self.fm_names:
            if name in self.backbones:
                fm_features[name] = self.backbones[name](video)
        return fm_features

    def forward(self, video):
        """
        End-to-end forward: raw video → FM extraction → fusion → task heads.
        """
        fm_features = self._extract_fm_features(video)

        # 1. PanEcho minimal processing (Global Context)
        if 'panecho' in fm_features:
            panecho_feat = fm_features['panecho']  # (B, C, T) directly from updated wrapper
            if panecho_feat.shape[-1] != self.out_time:
                panecho_feat = F.interpolate(panecho_feat, size=self.out_time, mode="linear", align_corners=False)
            global_ctx = self.panecho_proj(panecho_feat) # (B, 256, 16)
        else:
            global_ctx = None

        # 2. EchoPrime minimal processing (Spatial Features)
        if 'echoprime' in fm_features:
            spatial_feat = fm_features['echoprime'] # (B, C, T, H, W)
            spatial_feat = F.interpolate(
                spatial_feat, 
                size=(self.out_time, self.out_spatial[0], self.out_spatial[1]), 
                mode="trilinear", 
                align_corners=False
            )
            fused_spatial = self.echoprime_proj(spatial_feat) # (B, 256, 16, 14, 14)
        else:
            raise RuntimeError("EchoPrime spatial features are required.")

        # 3. Direct Fusion via FiLM
        if global_ctx is not None:
            fused_features = self.film(fused_spatial, global_ctx)
        else:
            fused_features = fused_spatial

        # Task Heads
        mask_logits = self.decoder(fused_features)

        return {
            "mask_logits": mask_logits,
            "router_weights": None
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