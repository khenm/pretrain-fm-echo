import logging
import os
import sys

import torch
import torch.nn as nn

from src.registry import register_model

logger = logging.getLogger(__name__)

def load_panecho_isolated(clip_len=16):
    """Loads PanEcho bypassing torch.hub to prevent namespace collisions using importlib."""
    import sys
    import os
    import torch
    import torch.distributed as dist
    import importlib.util
    
    import timm
    try:
        import wandb
        import pydantic
    except ImportError:
        pass
    # ---------------------------------------------------------

    hub_dir = os.path.expanduser('~/.cache/torch/hub/CarDS-Yale_PanEcho_main')
    
    if not os.path.exists(hub_dir):
        try:
            torch.hub.load('CarDS-Yale/PanEcho', 'PanEcho', pretrained=False, trust_repo=True)
        except Exception as e:
            logger.warning(f"Failed to pre-download PanEcho: {e}")

    # Aggressively remove EVERY local path that might shadow the repo's 'src'
    orig_path = list(sys.path)
    cwd = os.getcwd()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    sys.path = [p for p in sys.path if p and p not in (cwd, project_root, '')]
    sys.path.insert(0, hub_dir)

    stashed_modules = {}
    for mod_name in list(sys.modules.keys()):
        if mod_name == 'src' or mod_name.startswith('src.'):
            stashed_modules[mod_name] = sys.modules.pop(mod_name, None)

    try:
        is_distributed = dist.is_initialized()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        if is_distributed and local_rank != 0:
            dist.barrier()

        # Load PanEcho from its hubconf directly
        pt_path = os.path.join(hub_dir, 'hubconf.py')
        spec = importlib.util.spec_from_file_location("panecho_hubconf", pt_path)
        hubconf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hubconf)
        model = hubconf.PanEcho(pretrained=True, clip_len=clip_len)

        if is_distributed and local_rank == 0:
            dist.barrier()

        return model
    finally:
        sys.path = orig_path
        
        for mod_name in list(sys.modules.keys()):
            if mod_name == 'src' or mod_name.startswith('src.'):
                sys.modules.pop(mod_name, None)
                
        for mod_name, mod_info in stashed_modules.items():
            if mod_info is not None:
                sys.modules[mod_name] = mod_info


@register_model("PanEchoWrapper")
class PanEchoWrapper(nn.Module):
    """Integrates PanEcho spatiotemporal feature extraction into the main pipeline."""

    def __init__(self, clip_len=16, peft_cfg=None):
        super().__init__()
        self.model = load_panecho_isolated(clip_len=clip_len)
        self.clip_len = clip_len
        self.feature_dim = self.model.encoder.encoder.n_features

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        if peft_cfg and peft_cfg.get('enabled', True):
            self._apply_peft(peft_cfg)

    def _apply_peft(self, peft_cfg):
        """Applies LoRA adapters to PanEcho transformer attention layers."""
        try:
            from peft import get_peft_model, LoraConfig

            target_modules = peft_cfg.get(
                'panecho_target_modules',
                peft_cfg.get('target_modules', ['out_proj', 'in_proj_weight'])
            )
            lora_config = LoraConfig(
                r=peft_cfg.get('rank', 8),
                lora_alpha=peft_cfg.get('alpha', 16),
                lora_dropout=peft_cfg.get('dropout', 0.05),
                target_modules=target_modules,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)

            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"PanEcho PEFT applied: {trainable:,} trainable / {total:,} total "
                f"({100 * trainable / total:.2f}%)"
            )
        except ImportError:
            logger.warning("peft library not installed. Skipping LoRA for PanEcho.")

    def forward(self, frames):
        """
        Extracts spatiotemporal sequences from video frames.

        Args:
            frames (Tensor): Video frames (batch_size, channels, time, height, width)

        Returns:
            Tensor: Spatiotemporal features (batch_size, feature_dim, time, 1, 1)
        """
        batch_size, channels, length, height, width = frames.shape
        frames_reshaped = frames.reshape(batch_size * length, channels, height, width)

        embeddings_spatial = self.model.encoder.encoder(frames_reshaped)
        embeddings_temporal = embeddings_spatial.reshape(batch_size, length, self.feature_dim)
        embeddings_temporal = self.model.encoder.time_encoder(embeddings_temporal)

        features = self.model.encoder.transformer(embeddings_temporal)
        features = features.permute(0, 2, 1)

        return features.unsqueeze(-1).unsqueeze(-1)

    @classmethod
    def from_config(cls, cfg):
        clip_len = cfg.get('model', {}).get('max_clip_len', 16)
        peft_cfg = cfg.get('model', {}).get('peft')
        return cls(clip_len=clip_len, peft_cfg=peft_cfg)

