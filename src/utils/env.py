import torch
import numpy as np
import random
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Loads a configuration file and resolves component references robustly.
    """
    config_file = Path(config_path).resolve()
    
    if not config_file.exists():
        raise FileNotFoundError(f"Base config not found at {config_file}")
        
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
        
    base_dir = config_file.parent
    
    if isinstance(cfg.get('model'), str):
        model_name = cfg['model']
        model_config_path = base_dir / 'models' / f"{model_name}.yaml"
        if not model_config_path.exists():
             raise FileNotFoundError(f"Model config not found at {model_config_path}")
             
        with open(model_config_path, 'r') as f:
            model_cfg = yaml.safe_load(f)
        cfg['model'] = model_cfg
        
    if isinstance(cfg.get('data'), str):
        dataset_name = cfg['data']
        dataset_config_path = base_dir / 'datasets' / f"{dataset_name}.yaml"
        if not dataset_config_path.exists():
             raise FileNotFoundError(f"Dataset config not found at {dataset_config_path}")
             
        with open(dataset_config_path, 'r') as f:
            dataset_cfg = yaml.safe_load(f)
            
        cfg['data'] = dataset_cfg
    
    elif isinstance(cfg.get('data'), dict) and 'name' in cfg['data']:
        dataset_name = cfg['data']['name']
        dataset_config_path = base_dir / 'datasets' / f"{dataset_name}.yaml"
        if not dataset_config_path.exists():
            raise FileNotFoundError(f"Dataset config not found at {dataset_config_path}")
        
        with open(dataset_config_path, 'r') as f:
            dataset_cfg = yaml.safe_load(f)

        inline_overrides = cfg['data']
        dataset_cfg.update(inline_overrides)
        cfg['data'] = dataset_cfg
    
    return cfg

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device_id=0):
    return torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

def load_checkpoint(ckpt_path, model, optimizer=None, scaler=None, device='cpu', load_rng=False, strict=True):
    start_epoch = 1
    best_metric = -float('inf')
    if ckpt_path is None:
        raise ValueError("checkpoint path is None")
    
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    def restore_rng(rng_state):
        if rng_state is None: return
        try:
            torch.set_rng_state(rng_state["torch"])
            if torch.cuda.is_available() and "cuda" in rng_state and rng_state["cuda"] is not None:
                torch.cuda.set_rng_state_all(rng_state["cuda"])
            np.random.set_state(rng_state["numpy"])
            random.setstate(rng_state["python"])
            print("Restored RNG states")
        except Exception as e:
             print(f"Warning: Failed to restore RNG state: {e}")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
                
        if load_rng and "rng_state" in checkpoint:
            restore_rng(checkpoint["rng_state"])
            
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "best_metric" in checkpoint:
            best_metric = checkpoint["best_metric"]
            
        print(f"Loaded checkpoint (epoch {checkpoint.get('epoch')}, best_metric {best_metric:.4f})")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded legacy checkpoint (model weights only).")
        
    return start_epoch, best_metric

def save_checkpoint(model, optimizer, epoch, best_metric, save_path, scaler=None, save_rng=True):
    if save_rng:
        rng_state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
    else:
        rng_state = None
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "rng_state": rng_state,
    }
    
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")

class TrainerState:
    """
    Manages the lifecycle, configuration, and state persistence of a training run.
    """
    def __init__(self, config_path: str | Path, seed: int = 42, device_id: int = 0):
        self.config = load_config(config_path)
        
        self.device = get_device(device_id)
        seed_everything(seed)
        
        self.current_epoch = 1
        self.best_metric = -float('inf')
        
        self.run_dir = None
        self.vault_path = None

    def load(self, ckpt_path: str | Path, model: torch.nn.Module, 
             optimizer: Optional[torch.optim.Optimizer] = None, 
             scaler: Optional[torch.amp.GradScaler] = None):
        """Restores model, optimizer, and synchronizes the state manager's internal trackers."""
        start_epoch, best_metric = load_checkpoint(
            ckpt_path, model, optimizer=optimizer, scaler=scaler, 
            device=self.device, load_rng=True, strict=True
        )
        self.current_epoch = start_epoch
        self.best_metric = best_metric

    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
             scaler: Optional[torch.amp.GradScaler] = None, is_best: bool = False):
        """Saves current state, automatically handling standard and vault (best) saves."""
        if not self.run_dir:
            raise ValueError("TrainerState.run_dir is not set. Cannot save checkpoint.")
            
        last_path = os.path.join(self.run_dir, 'last.ckpt')
        
        save_checkpoint(
            model=model, optimizer=optimizer, epoch=self.current_epoch, 
            best_metric=self.best_metric, save_path=last_path, 
            scaler=scaler, save_rng=True
        )
        
        if is_best and self.vault_path:
            save_checkpoint(
                model=model, optimizer=optimizer, epoch=self.current_epoch, 
                best_metric=self.best_metric, save_path=self.vault_path, 
                scaler=scaler, save_rng=True
            )
            print(f"Saved Best Checkpoint to vault: {self.vault_path}")