import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.registry import register_tta_component

logger = logging.getLogger(__name__)


def extract_batch_data(batch: Any, device: torch.device) -> Tuple[torch.Tensor, Any, Optional[torch.Tensor]]:
    """
    Standardizes data extraction from various batch formats.
    
    Returns:
        inputs, targets, frame_mask (optional)
    """
    inputs = None
    targets = None
    frame_mask = None

    if isinstance(batch, dict):
        # Extract Inputs
        if "video" in batch: 
            inputs = batch["video"]
        elif "image" in batch: 
            inputs = batch["image"]
        else: 
            inputs = list(batch.values())[0]

        # Extract Targets
        if "target" in batch: 
            targets = batch["target"]
        elif "label" in batch: 
            targets = batch["label"]
        elif "ef" in batch: 
            targets = batch["ef"]
        elif "EF" in batch: 
            targets = batch["EF"]
        else:
            # Fallback for target extraction
            vals = list(batch.values())
            targets = vals[1] if len(vals) > 1 else None

        frame_mask = batch.get("frame_mask")

    elif isinstance(batch, (list, tuple)):
        inputs, targets = batch[0], batch[1]
        if len(batch) > 2:
            frame_mask = batch[2]
            
    else:
        raise ValueError(f"Unsupported batch type: {type(batch)}")

    if inputs is not None:
        inputs = inputs.to(device)
    if targets is not None:
        targets = targets.to(device)
    if frame_mask is not None:
        frame_mask = frame_mask.to(device)
        
    return inputs, targets, frame_mask


def parse_model_output(out: Any, task: str = 'any') -> torch.Tensor:
    """
    Extracts the relevant tensor from complex model outputs (tuple, dict, etc).
    """
    if isinstance(out, torch.Tensor):
        return out
        
    if isinstance(out, tuple):
        # Heuristic Search
        seg = None
        reg = None
        
        for item in out:
            if isinstance(item, torch.Tensor):
                if item.ndim >= 4: 
                    seg = item
                elif item.ndim == 2 and item.shape[1] == 1:
                    reg = item

        if task == 'segmentation':
            if seg is not None: return seg
            return out[0] if len(out) > 0 else out
            
        elif task == 'regression':
            if reg is not None: return reg
            return out[1] if len(out) > 1 else out[0]
            
        # Default fallback
        return out[0]

    if isinstance(out, dict):
        if task == 'segmentation':
            return out.get("mask_logits", out.get("logits", out.get("segmentation")))
        elif task == 'regression':
            return out.get("pred_ef", out.get("ef", out.get("regression")))
            
    return out


class BaseConformalCalibrator(ABC):
    """Abstract Base Class for Conformal Calibrators."""
    
    def __init__(self, alpha: float = 0.05, gamma: float = 0.01, num_classes: Optional[int] = None, **kwargs: Any):
        self.target_alpha = alpha
        self.current_alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        
        self.cal_scores = np.array([])
        self.q_hat: Optional[float] = None
        
        logger.info(f"Initialized {self.__class__.__name__}(alpha={alpha}, gamma={gamma})")

    @abstractmethod
    def calibrate(self, valid_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device) -> None:
        pass

    @abstractmethod
    def predict(self, logits: torch.Tensor, martingale_stable: bool = True, audit_score: Optional[float] = None, audit_epsilon: Optional[float] = None) -> Tuple[Any, float]:
        pass
    
    def quantile_from_alpha(self, alpha: float) -> float:
        """Helper to look up quantile from calibration table dynamically."""
        if len(self.cal_scores) == 0:
            return 0.95
            
        q_level = 1.0 - alpha
        q_level = min(max(q_level, 0.0), 1.0)
        return np.quantile(self.cal_scores, q_level, method='linear')

    def _update_alpha(self, error_rate: float) -> None:
        """Adaptive Conformal Inference (ACI) update rule."""
        self.current_alpha = self.current_alpha - self.gamma * (error_rate - self.target_alpha)
        self.current_alpha = np.clip(self.current_alpha, 0.001, 0.999)


@register_tta_component("classification_calibrator")
class ClassificationCalibrator(BaseConformalCalibrator):
    """Conformal Prediction for classification (Prediction Sets) using ACI."""

    def __init__(self, alpha: float = 0.05, gamma: float = 0.01, num_classes: int = 10, **kwargs: Any):
        super().__init__(alpha=alpha, gamma=gamma, num_classes=num_classes, **kwargs)

    def calibrate(self, valid_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device) -> None:
        logger.info("Starting Classification Conformal Calibration...")
        model.eval()
        scores_list = []
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Calibration"):
                inputs, targets, _ = extract_batch_data(batch, device)
                if targets is not None:
                    targets = targets.long()
                
                out = model(inputs)
                logits = parse_model_output(out)
                
                probs = F.softmax(logits, dim=1)
                
                # Get probability of the TRUE class
                true_class_probs = probs.gather(1, targets.unsqueeze(1)).squeeze()
                batch_scores = 1.0 - true_class_probs
                
                batch_scores_np = batch_scores.cpu().numpy()
                if batch_scores_np.ndim == 0:
                    batch_scores_np = np.expand_dims(batch_scores_np, axis=0)
                    
                scores_list.append(batch_scores_np)
                
        if not scores_list:
            logger.warning("Calibration data empty.")
            return

        self.cal_scores = np.concatenate(scores_list)
        
        n = len(self.cal_scores)
        q_level = np.ceil((n + 1) * (1 - self.target_alpha)) / n
        q_level = min(max(q_level, 0.0), 1.0)
        
        self.q_hat = np.quantile(self.cal_scores, q_level, method='higher')
        logger.info(f"Calibration Complete. N={n}, Initial Quantile (q_hat): {self.q_hat:.4f}")

    def predict(self, logits: torch.Tensor, martingale_stable: bool = True, audit_score: Optional[float] = None, audit_epsilon: Optional[float] = None) -> Tuple[List[List[int]], float]:
        probs = F.softmax(logits, dim=1)
        batch_size = probs.shape[0]
        prediction_sets = []
        
        # 1. Determine Threshold
        if self.q_hat is None:
             logger.warning("Calibrator not calibrated! Using default 0.95.")
             current_q = 0.95
        else:
             current_q = self.quantile_from_alpha(self.current_alpha)

        # 2. Construct Sets
        prob_threshold = 1.0 - current_q
        pseudo_labels = torch.argmax(probs, dim=1)
        
        prob_np = probs.detach().cpu().numpy()
        pseudo_labels_np = pseudo_labels.detach().cpu().numpy()
        
        for i in range(batch_size):
            classes = np.nonzero(prob_np[i] >= prob_threshold)[0]
            if len(classes) == 0:
                classes = np.array([pseudo_labels_np[i]])
            prediction_sets.append(classes.tolist())
            
        # 3. ACI Update
        if martingale_stable:
            avg_err = 0.0
            for i, p_label in enumerate(pseudo_labels_np):
                if p_label not in prediction_sets[i]:
                    avg_err += 1.0
            avg_err /= batch_size
            self._update_alpha(avg_err)

        return prediction_sets, current_q


@register_tta_component("regression_calibrator")
class RegressionCalibrator(BaseConformalCalibrator):
    """Conformal Prediction for Regression (EF Estimation)."""

    def calibrate(self, valid_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device) -> None:
        logger.info("Starting Regression Conformal Calibration...")
        model.eval()
        residuals = []
        
        with torch.no_grad():
            for batch in valid_loader:
                inputs, targets, _ = extract_batch_data(batch, device)
                
                preds = parse_model_output(model(inputs), task='regression')
                
                # Non-conformity score: Absolute Error
                batch_residuals = torch.abs(preds.squeeze() - targets.squeeze())
                
                batch_residuals_np = batch_residuals.cpu().numpy()
                if batch_residuals_np.ndim == 0:
                     batch_residuals_np = np.expand_dims(batch_residuals_np, axis=0)
                     
                residuals.extend(batch_residuals_np)
                
        self.cal_scores = np.array(residuals)
        n = len(self.cal_scores)
        
        if n == 0:
             logger.warning("Regression calibration empty.")
             self.q_hat = 0.5
             return

        q_level = np.ceil((n + 1) * (1 - self.target_alpha)) / n
        q_level = min(max(q_level, 0.0), 1.0)
        self.q_hat = np.quantile(self.cal_scores, q_level)
        logger.info(f"Calibration Complete. Margin (q_hat): +/- {self.q_hat:.4f}")

    def predict(self, logits: torch.Tensor, martingale_stable: bool = True, audit_score: Optional[float] = None, audit_epsilon: Optional[float] = None) -> Tuple[Dict[str, Any], float]:
        preds = logits.detach().cpu().numpy().squeeze()
        if preds.ndim == 0: 
            preds = np.array([preds])
        
        if self.q_hat is None:
             current_q = 0.5
        elif len(self.cal_scores) == 0:
             current_q = self.q_hat
        else:
             current_q = self.quantile_from_alpha(self.current_alpha) 
        
        lower_bounds = np.maximum(0.0, preds - current_q)
        upper_bounds = np.minimum(1.0, preds + current_q)
        
        prediction_intervals = list(zip(lower_bounds, upper_bounds))

        if martingale_stable and audit_score is not None and audit_epsilon is not None:
            proxy_error = 1.0 if audit_score > audit_epsilon else 0.0
            self._update_alpha(proxy_error)

        return {'intervals': prediction_intervals, 'predictions': preds.tolist()}, current_q


@register_tta_component("segmentation_calibrator")
class SegmentationCalibrator(BaseConformalCalibrator):
    """Outputs a Core Mask (high confidence) and a Shadow Mask (uncertainty zone)."""

    def calibrate(self, valid_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device) -> None:
        logger.info("Starting Segmentation Conformal Calibration (Pixel-wise)...")
        model.eval()
        scores_list = []
        
        with torch.no_grad():
            for batch in valid_loader:
                inputs, targets, frame_mask = extract_batch_data(batch, device)
                if targets is None or frame_mask is None:
                    continue
                
                logits = parse_model_output(model(inputs), task='segmentation')
                probs = torch.sigmoid(logits)
                
                # Filter out UNLABELED frames
                _, C, _, H, W = probs.shape
                probs_flat = probs.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
                targets_flat = targets.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
                frame_mask_flat = frame_mask.view(-1)
                
                valid_indices = torch.nonzero(frame_mask_flat).squeeze()
                if valid_indices.numel() == 0:
                    continue
                    
                probs_valid = probs_flat[valid_indices]
                targets_valid = targets_flat[valid_indices]

                # Score = 1 - p(true_class)
                p_true = torch.where(targets_valid > 0.5, probs_valid, 1.0 - probs_valid)
                batch_scores = 1.0 - p_true
                
                # Subsample to avoid OOM
                flat_scores = batch_scores.flatten()
                if flat_scores.numel() > 10000:
                    idx = torch.randperm(flat_scores.numel())[:10000]
                    flat_scores = flat_scores[idx]
                    
                scores_list.append(flat_scores.cpu().numpy())

    def predict(self, logits: torch.Tensor, martingale_stable: bool = True, audit_score: Optional[float] = None, audit_epsilon: Optional[float] = None) -> Tuple[Tuple[torch.Tensor, torch.Tensor], float]:
        probs = torch.sigmoid(logits)
        base_q = self.q_hat if self.q_hat is not None else 0.1
        
        sensitivity = 0.05
        if martingale_stable and audit_score is not None:
            current_q = base_q * (1.0 + sensitivity * np.log1p(audit_score))
        else:
            current_q = 0.99

        current_q = min(current_q, 0.99)
        
        upper_thresh = 1.0 - current_q
        lower_thresh = current_q
        
        core_mask = (probs >= upper_thresh).int()
        shadow_mask = ((probs >= lower_thresh) & (probs < upper_thresh)).int()
        
        return (core_mask, shadow_mask), current_q


class _SmartShim(torch.nn.Module):
    """Shim to present a uniform interface for sub-calibrators."""
    def __init__(self, model: torch.nn.Module, task: str):
        super().__init__()
        self.model = model
        self.task = task

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return parse_model_output(out, self.task)


@register_tta_component("audit_calibrator")
class AuditCalibrator(BaseConformalCalibrator):
    """Wraps multiple calibrators for hybrid models."""

    def __init__(self, calibrators: Dict[str, BaseConformalCalibrator], **kwargs: Any):
        super().__init__(**kwargs)
        self.calibrators = calibrators
        logger.info(f"Initialized AuditCalibrator with: {list(calibrators.keys())}")

    def calibrate(self, valid_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device) -> None:
        logger.info("Starting Audit (Hybrid) Calibration...")
        for name, cal in self.calibrators.items():
            logger.info(f"Calibrating sub-component: {name}")
            shim_model = _SmartShim(model, name)
            cal.calibrate(valid_loader, shim_model, device)
            
    def predict(self, logits: Any, martingale_stable: bool = True, audit_score: Optional[float] = None, audit_epsilon: Optional[float] = None) -> Tuple[Dict[str, Any], float]:
        results = {}
        
        ef_logit = None
        seg_logit = None
        
        if isinstance(logits, (tuple, list)):
            ef_logit = logits[0]
            seg_logit = logits[1]
        elif isinstance(logits, dict):
             ef_logit = logits.get("pred_ef", logits.get("ef"))
             seg_logit = logits.get("mask_logits", logits.get("segmentation"))
        
        q_vals = []
        
        if 'regression' in self.calibrators and ef_logit is not None:
            res, q = self.calibrators['regression'].predict(ef_logit, martingale_stable, audit_score, audit_epsilon)
            results['regression'] = res
            q_vals.append(q)
            
        if 'segmentation' in self.calibrators and seg_logit is not None:
            res, q = self.calibrators['segmentation'].predict(seg_logit, martingale_stable, audit_score, audit_epsilon)
            results['segmentation'] = res
            q_vals.append(q)
            
        final_q = max(q_vals) if q_vals else 0.0
        return results, final_q

# Alias for backward compatibility
ConformalCalibrator = ClassificationCalibrator
