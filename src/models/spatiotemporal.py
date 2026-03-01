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

class GeometricVolumeHead(nn.Module):
    """
    Calculates the 1D Volume sequence from the 3D target mask volume using a lightweight temporal model.
    """
    def __init__(self, feature_dim=256, hidden_dim=16):
        super().__init__()
        self.temporal_model = nn.Sequential(
            nn.Conv1d(feature_dim + 1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)
        )

    def forward(self, mask_logits, fused_features):
        """
        Converts probability masks into a continuous volume curve.
        
        Args:
            mask_logits (Tensor): (batch_size, channels, time, height, width)
            fused_features (Tensor): (batch_size, channels, time, height, width)
            
        Returns:
            Tensor: Volume curve dimensions (batch_size, time)
        """
        mask_prob = torch.sigmoid(mask_logits)
        
        # True high-res area
        area = mask_prob.sum(dim=(3, 4)) + 1e-6
        area_feat = torch.log(area)
        
        # Match temporal and spatial dimensions efficiently for feature weighting
        time_m = mask_prob.shape[2]
        _, _, time_f, h_f, w_f = fused_features.shape
        
        if time_m != time_f:
            fused_features_t = F.interpolate(
                fused_features, size=(time_m, h_f, w_f), 
                mode='trilinear', align_corners=False
            )
        else:
            fused_features_t = fused_features
            
        mask_prob_s = F.adaptive_avg_pool3d(mask_prob, output_size=(time_m, h_f, w_f))
        pooled_area = mask_prob_s.sum(dim=(3, 4)) + 1e-6
        
        numerator = (fused_features_t * mask_prob_s).sum(dim=(3, 4))
        avg_feat = numerator / pooled_area
        
        feat_vec = torch.cat([avg_feat, area_feat], dim=1)
        
        vol_curve = self.temporal_model(feat_vec).squeeze(1)
        
        return vol_curve

class FeatureRouter(nn.Module):
    """
    Lightweight, trainable gating network that calculates a probability distribution over experts for spatial patches.
    """
    def __init__(self, in_channels, num_experts=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, num_experts, kernel_size=1)
        )

    def forward(self, features):
        """
        Calculates expert selection probability.
        
        Args:
            features (Tensor): (batch_size, channels, time, height, width)
            
        Returns:
            Tensor: Probability weights vector scaling over experts.
        """
        logits = self.net(features)
        return F.softmax(logits, dim=1)

class PhaseClassifier(nn.Module):
    """
    Lightweight 1D temporal classifier for per-frame cardiac phase prediction.
    Operates on GAP-pooled fused features to produce per-frame logits
    for 3 classes: {0: background, 1: ES, 2: ED}.
    """
    def __init__(self, in_channels, num_phases=3, hidden_dim=64):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.ConvTranspose1d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, num_phases, kernel_size=3, padding=1)
        )

    def forward(self, fused_features):
        """
        Args:
            fused_features (Tensor): (B, C, T, H, W)

        Returns:
            Tensor: Per-frame phase logits (B, num_phases, T)
        """
        pooled = fused_features.mean(dim=(3, 4))
        return self.classifier(pooled)


@register_model("SpatiotemporalEchoModel")
class SpatiotemporalEchoModel(nn.Module):
    """
    Unified 3D model combining spatial features from multiple representations.
    Contains dynamic routing, 3D upsampling, temporal geometric regressions,
    and per-frame cardiac phase classification.
    """
    def __init__(self, pan_echo_dim=768, echo_prime_dim=768, fused_dim=256, num_classes=1, num_phases=3):
        super().__init__()
        concat_dim = pan_echo_dim + echo_prime_dim

        self.router = FeatureRouter(
            in_channels=concat_dim,
            num_experts=2,
            hidden_dim=64
        )
        self.fusion_conv = nn.Conv3d(concat_dim, fused_dim, kernel_size=1)
        self.decoder = SpatiotemporalDecoder(in_channels=fused_dim, out_channels=num_classes)
        self.volume_head = GeometricVolumeHead(feature_dim=fused_dim)
        self.phase_head = PhaseClassifier(in_channels=fused_dim, num_phases=num_phases)

    def forward(self, pan_features, prime_features):
        """
        Routes and fuses multidimensional backbone features into a continuous volume sequence.

        Args:
            pan_features (Tensor): Target spatial sequence elements
            prime_features (Tensor): Companion target elements mapped over sequence.
        """
        if pan_features.shape[2:] != prime_features.shape[2:]:
            # pan_features is (B, C, T, 1, 1), prime is (B, C, T, H, W)
            # Expand the 1x1 spatial dims of PanEcho to match EchoPrime's HxW
            _, _, T_prime, H, W = prime_features.shape
            T_pan = pan_features.shape[2]
            if T_pan != T_prime:
                pan_features = F.interpolate(
                    pan_features, size=(T_prime, 1, 1),
                    mode='trilinear', align_corners=False
                )
            pan_features = pan_features.expand(-1, -1, -1, H, W)

        concat_features = torch.cat([pan_features, prime_features], dim=1)
        router_weights = self.router(concat_features)

        w_pan = router_weights[:, 0:1, ...]
        w_prime = router_weights[:, 1:2, ...]

        weighted_pan = pan_features * w_pan
        weighted_prime = prime_features * w_prime
        weighted_concat = torch.cat([weighted_pan, weighted_prime], dim=1)

        fused_features = self.fusion_conv(weighted_concat)
        mask_logits = self.decoder(fused_features)
        vol_curve = self.volume_head(mask_logits, fused_features)
        phase_logits = self.phase_head(fused_features)

        # Probabilistic Volume Extraction using Softmax Temperature
        phase_probs = torch.softmax(phase_logits, dim=1) # (B, num_phases, T)
        
        # 0: background, 1: ES, 2: ED
        p_es = phase_probs[:, 1, :]
        p_ed = phase_probs[:, 2, :]
        
        pred_edv = torch.sum(p_ed * vol_curve, dim=1) / torch.sum(p_ed, dim=1).clamp(min=1e-3)
        pred_esv = torch.sum(p_es * vol_curve, dim=1) / torch.sum(p_es, dim=1).clamp(min=1e-3)

        return {
            "mask_logits": mask_logits,
            "vol_curve": vol_curve,
            "phase_logits": phase_logits,
            "pred_edv": pred_edv,
            "pred_esv": pred_esv
        }

    @classmethod
    def from_config(cls, cfg):
        model_cfg = cfg.get('model', {})
        return cls(
            pan_echo_dim=model_cfg.get('pan_echo_dim', 768),
            echo_prime_dim=model_cfg.get('echo_prime_dim', 768),
            fused_dim=model_cfg.get('fused_dim', 256),
            num_classes=cfg.get('data', {}).get('num_classes', 1),
            num_phases=model_cfg.get('num_phases', 3)
        )

@register_model("SpatiotemporalPipeline")
class SpatiotemporalPipeline(nn.Module):
    """
    End-to-End Pipeline wrapping PanEcho, EchoPrime, and SpatiotemporalEchoModel.
    Consumes raw video frames and outputs continuous 3D mask logits and volume curves.
    """
    def __init__(self, cfg):
        super().__init__()
        from src.models.panecho_wrapper import PanEchoWrapper
        from src.models.echoprime_wrapper import EchoPrimeWrapper
        
        original_clip_len = cfg.get('model', {}).get('max_clip_len', 16)
        if 'model' in cfg:
            cfg['model']['max_clip_len'] = 16
        
        self.panecho = PanEchoWrapper.from_config(cfg)
        self.echoprime = EchoPrimeWrapper.from_config(cfg)
        self.chunk_size = 16
        
        # Restore the original clip length for downstream pipeline components like positional embeddings if needed
        if 'model' in cfg:
            cfg['model']['max_clip_len'] = original_clip_len
        
        pan_dim = self.panecho.feature_dim
        prime_dim = self.echoprime.feature_dim
        
        model_cfg = cfg.get('model', {})
        model_cfg['pan_echo_dim'] = pan_dim
        model_cfg['echo_prime_dim'] = prime_dim
        cfg['model'] = model_cfg
        
        self.fusion_model = SpatiotemporalEchoModel.from_config(cfg)

    def forward(self, frames):
        """
        Args:
            frames (Tensor): Raw video (B, C, T, H, W)
            
        Returns:
            Dict: Output from SpatiotemporalEchoModel
        """
        B, C, T, H, W = frames.shape
        
        pan_feature_chunks = []
        prime_feature_chunks = []
        
        # Chunking along the temporal dimension (T)
        for i in range(0, T, self.chunk_size):
            end_idx = min(i + self.chunk_size, T)
            chunk = frames[:, :, i:end_idx, :, :]
            
            # Pad the last chunk if it's smaller than chunk_size
            curr_chunk_size = chunk.shape[2]
            if curr_chunk_size < self.chunk_size:
                pad_size = self.chunk_size - curr_chunk_size
                # Pad along temporal dimension (dim 2)
                # pad format is (W_pad_left, W_pad_right, H_pad_top, H_pad_bottom, T_pad_front, T_pad_back)
                chunk = F.pad(chunk, (0, 0, 0, 0, 0, pad_size))
                
            pan_feat = self.panecho(chunk)
            prime_feat = self.echoprime(chunk)
            
            # If we padded, slice off the padded features from the output
            if curr_chunk_size < self.chunk_size:
                pan_feat = pan_feat[:, :, :curr_chunk_size, ...]
                prime_feat = prime_feat[:, :, :curr_chunk_size, ...]
                
            pan_feature_chunks.append(pan_feat)
            prime_feature_chunks.append(prime_feat)
            
        # Concatenate features along the temporal dimension
        pan_features = torch.cat(pan_feature_chunks, dim=2)
        prime_features = torch.cat(prime_feature_chunks, dim=2)
        
        return self.fusion_model(pan_features, prime_features)

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg)

