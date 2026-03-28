import os
import json
import logging
import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

class SelfAuditor:
    """
    PyTorch implementation of SelfAuditor for OOD detection via Martingale Wealth.
    Calculates Martingale Wealth based on Entropy and Feature Drift.
    """
    def __init__(self, alpha=0.5, delta=0.05, lambda_val=0.5, device="cpu"):
        self.alpha = alpha
        self.delta = delta
        self.lambda_val = lambda_val
        self.threshold = 1.0 / delta
        self.device = device
        
        self.martingale = 1.0
        self.collapsed = False
        self.wealth_history = []
        
        # Calibration Stats
        self.mu_source = None
        self.epsilon = 0.5  # Default conservative value
        self.max_ent = torch.log(torch.tensor(2.0, device=device))
        
    def load_stats(self, stats_path: str):
        """Load calibration statistics from JSON."""
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                if stats.get('mu_source') is not None:
                    self.mu_source = torch.tensor(stats['mu_source'], device=self.device)
                else:
                    self.mu_source = None
                self.epsilon = stats.get('epsilon', 0.5)
                self.max_ent = torch.tensor(stats.get('max_ent', np.log(2)), device=self.device)
                logger.info(f"Loaded auditor stats from {stats_path}")
            except Exception as e:
                logger.warning(f"Failed to load stats: {e}. Using defaults.")
        else:
            logger.warning(f"Stats file {stats_path} not found. Using defaults.")

    def _compute_entropy(self, inputs: torch.Tensor, is_logits=True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean entropy of logits or probabilities.
        Args:
            inputs: (C, H, W) or (1, H, W) - Logits or Probabilities
        Returns:
            entropy: scalar mean entropy
            max_ent: scalar max entropy
        """
        if is_logits:
            if inputs.ndim == 3 and inputs.shape[0] > 1:
                probs = F.softmax(inputs, dim=0)
                max_ent = torch.log(torch.tensor(inputs.shape[0], dtype=torch.float32, device=self.device))
            else:
                probs = torch.sigmoid(inputs)
                max_ent = torch.log(torch.tensor(2.0, device=self.device))
        else:
            probs = inputs
            if inputs.ndim == 3 and inputs.shape[0] > 1:
                max_ent = torch.log(torch.tensor(inputs.shape[0], dtype=torch.float32, device=self.device))
            else:
                max_ent = torch.log(torch.tensor(2.0, device=self.device))

        eps = 1e-10
        if inputs.ndim == 3 and inputs.shape[0] > 1:
             ent = -torch.sum(probs * torch.log(probs + eps), dim=0)
        else:
             ent = -(probs * torch.log(probs + eps) + (1.0 - probs) * torch.log(1.0 - probs + eps))
             
        return torch.mean(ent), max_ent

    def update(self, inputs: torch.Tensor, features: torch.Tensor = None, is_logits=True) -> float:
        """
        Update Martingale Wealth.
        Args:
            inputs: (C, H, W) for current frame
            features: (D,) optional feature vector for drift tracking
        Returns:
            wealth: Current martingale value
        """
        inputs = inputs.to(self.device)
        
        # 1. Entropy
        entropy, max_ent = self._compute_entropy(inputs, is_logits=is_logits)
        norm_entropy = (entropy / max_ent) if max_ent > 0 else torch.tensor(0.0, device=self.device)
        
        # 2. Drift
        drift = torch.tensor(0.0, device=self.device)
        if self.mu_source is not None and features is not None:
            features = features.to(self.device)
            feat_flat = features.flatten()
            if self.mu_source.shape == feat_flat.shape:
                feat_norm = feat_flat / (torch.norm(feat_flat) + 1e-10)
                drift = 1.0 - torch.dot(feat_norm, self.mu_source)
        
        # 3. Composite Score
        if self.mu_source is None:
             score = float(norm_entropy.cpu().item())
        else:
             score = float((self.alpha * norm_entropy + (1.0 - self.alpha) * drift).cpu().item())
        
        # 4. Update Wealth
        bet = 1.0 + self.lambda_val * (score - self.epsilon)
        bet = max(0.1, bet)
        
        self.martingale *= bet
        self.wealth_history.append(self.martingale)
        
        if self.martingale > self.threshold:
            self.collapsed = True
            
        return self.martingale
