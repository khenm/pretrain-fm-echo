import logging

import torch
import torch.nn as nn
import torchvision.models.video

import torch.nn.functional as F

from src.registry import register_model

logger = logging.getLogger(__name__)

MVIT_NATIVE_CLIP_LEN = 16


@register_model("EchoPrimeWrapper")
class EchoPrimeWrapper(nn.Module):
    """Integrates EchoPrime spatiotemporal feature extraction into the main pipeline."""

    def __init__(self, device='cpu', clip_len=32, peft_cfg=None):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.feature_dim = 768
        self._head_dim = 512
        self._norm_output = None
        self.native_clip_len = MVIT_NATIVE_CLIP_LEN
        self.clip_len = clip_len

        self._init_encoder()

        if peft_cfg and peft_cfg.get('enabled', True):
            self._apply_peft(peft_cfg)

        self._register_norm_hook()

    def _get_norm_layer(self):
        """Resolves the norm layer from the encoder, handling PEFT wrapping."""
        model = self.encoder
        if hasattr(model, 'base_model'):
            model = model.base_model
        if hasattr(model, 'model'):
            model = model.model
        return model.norm

    def _register_norm_hook(self):
        """Registers a forward hook on the norm layer to capture its output."""
        norm_layer = self._get_norm_layer()
        norm_layer.register_forward_hook(self._norm_hook)
        logger.info("Registered forward hook on encoder norm layer")

    def _norm_hook(self, module, input, output):
        """Forward hook that stores the norm layer output."""
        self._norm_output = output

    def _init_encoder(self):
        """Initializes the MViT encoder and loads pretrained weights if available."""
        self.encoder = torchvision.models.video.mvit_v2_s()
        self.encoder.head[-1] = nn.Linear(self.encoder.head[-1].in_features, self._head_dim)

        try:
            checkpoint = torch.load("model_data/weights/echo_prime_encoder.pt", map_location=self.device)
            self.encoder.load_state_dict(checkpoint)
        except Exception as e:
            logger.warning(f"Could not load EchoPrime encoder weights: {e}")

        self.encoder.eval()
        self.encoder.to(self.device)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def _apply_peft(self, peft_cfg):
        """Applies LoRA adapters to MViT attention layers."""
        try:
            from peft import get_peft_model, LoraConfig

            target_modules = peft_cfg.get('target_modules', ['qkv', 'proj'])
            lora_config = LoraConfig(
                r=peft_cfg.get('rank', 8),
                lora_alpha=peft_cfg.get('alpha', 16),
                lora_dropout=peft_cfg.get('dropout', 0.05),
                target_modules=target_modules,
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, lora_config)

            trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.encoder.parameters())
            logger.info(
                f"EchoPrime PEFT applied: {trainable:,} trainable / {total:,} total "
                f"({100 * trainable / total:.2f}%)"
            )
        except ImportError:
            logger.warning("peft library not installed. Skipping LoRA for EchoPrime.")

    def _encode_native(self, frames):
        """Runs a single 16-frame clip through MViT and returns norm-layer features.

        Args:
            frames (Tensor): (B, C, 16, H, W)

        Returns:
            Tensor: (B, feature_dim, T_out, H_out, W_out)
        """
        self._norm_output = None
        self.encoder(frames)
        features = self._norm_output

        if features.dim() == 3 and features.shape[1] == 393:
            batch_size, _, channels = features.shape
            features = features[:, 1:, :]
            features = features.view(batch_size, 8, 7, 7, channels)
            features = features.permute(0, 4, 1, 2, 3).contiguous()

        return features

    def forward(self, frames):
        """Extracts spatiotemporal features, handling arbitrary temporal lengths.

        Uniformly subsamples input to native 16 frames, encodes through MViT,
        then interpolates output features back to the target temporal resolution.

        Args:
            frames (Tensor): Video frames (batch_size, channels, time, height, width)

        Returns:
            Tensor: Spatiotemporal features (batch_size, feature_dim, time, height, width)
        """
        total_t = frames.shape[2]
        native = self.native_clip_len

        if total_t <= native:
            if total_t < native:
                frames = F.pad(frames, (0, 0, 0, 0, 0, native - total_t), mode='replicate')
            features = self._encode_native(frames)
            if total_t < native and features.shape[2] > 1:
                target_t = max(1, total_t // 2)
                features = F.interpolate(features, size=(target_t, features.shape[3], features.shape[4]),
                                         mode='trilinear', align_corners=False)
            return features

        # Uniformly subsample to native_clip_len frames
        indices = torch.linspace(0, total_t - 1, native).long().to(frames.device)
        sampled_frames = frames[:, :, indices]
        features = self._encode_native(sampled_frames)

        return features

    @classmethod
    def from_config(cls, cfg):
        device = cfg.get('model', {}).get('device', 'cpu')
        clip_len = cfg.get('model', {}).get('max_clip_len', 32)
        peft_cfg = cfg.get('model', {}).get('peft')
        return cls(device=device, clip_len=clip_len, peft_cfg=peft_cfg)
