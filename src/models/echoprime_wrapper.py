import logging

import torch
import torch.nn as nn
import torchvision.models.video
from torchvision.models.feature_extraction import create_feature_extractor

from src.registry import register_model

logger = logging.getLogger(__name__)


@register_model("EchoPrimeWrapper")
class EchoPrimeWrapper(nn.Module):
    """Integrates EchoPrime spatiotemporal feature extraction into the main pipeline."""

    def __init__(self, device='cpu', peft_cfg=None):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.feature_dim = 768
        self._head_dim = 512

        self._init_encoder()

        if peft_cfg and peft_cfg.get('enabled', True):
            self._apply_peft(peft_cfg)

        self.feature_extractor = create_feature_extractor(
            self.encoder,
            return_nodes={'norm': 'norm'}
        )

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

    def forward(self, frames):
        """
        Extracts spatiotemporal features from video frames.

        Args:
            frames (Tensor): Video frames (batch_size, channels, time, height, width)

        Returns:
            Tensor: Spatiotemporal features (batch_size, feature_dim, time, height, width)
        """
        features_dict = self.feature_extractor(frames)
        features = features_dict['norm']

        if features.dim() == 3 and features.shape[1] == 393:
            batch_size, seq_len, channels = features.shape
            features = features[:, 1:, :]
            features = features.view(batch_size, 8, 7, 7, channels)
            features = features.permute(0, 4, 1, 2, 3).contiguous()

        return features

    @classmethod
    def from_config(cls, cfg):
        device = cfg.get('model', {}).get('device', 'cpu')
        peft_cfg = cfg.get('model', {}).get('peft')
        return cls(device=device, peft_cfg=peft_cfg)
