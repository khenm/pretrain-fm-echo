import os
import random
import numpy as np
import torch
from tqdm import tqdm
from src.utils.logging import get_logger
from src.utils.dist import is_main_process, reduce_tensor, get_rank, setup_fsdp
import torch.distributed as dist

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = get_logger()

class Trainer:
    """
    Handles generic training and validation across unified architectures.
    """
    def __init__(self, model, loaders, state, criterions=None, metrics=None):
        self.model = model
        self.state = state
        self.cfg = state.config
        self.device = state.device
        
        # Unpack loaders. Typically trains, val. Test is optional.
        if len(loaders) == 2:
            self.ld_tr, self.ld_va = loaders
            self.ld_ts = None
        elif len(loaders) == 3:
            self.ld_tr, self.ld_va, self.ld_ts = loaders
            
        self.criterions = criterions or {'ce': torch.nn.CrossEntropyLoss()}
        self.metrics = metrics or {}
        
        # Fetch dynamic dataset keys to prevent magic string failures
        self.input_key = self.cfg.get('data', {}).get('input_key', 'image')
        self.target_key = self.cfg.get('data', {}).get('target_key', 'label')
        
        self.num_classes = self.cfg.get('data', {}).get('num_classes', 10)
        self._setup_optimization()
        
        # Setup metric devices
        for k, metric in self.metrics.items():
             if hasattr(metric, 'to'):
                  metric.to(self.device)

    def _setup_optimization(self):
        # Gradient Accumulation
        micro_batch = self.cfg['training'].get('batch_size', 32)
        effective_train = self.cfg['training'].get('train_batch_size', micro_batch)
        self.accum_steps = max(1, effective_train // micro_batch)
        if self.accum_steps > 1:
            logger.info(f"Gradient Accumulation: {self.accum_steps} steps (effective batch size: {effective_train})")

        base_lr = self.cfg['training'].get('lr', 1e-3)
        weight_decay = self.cfg['training'].get('weight_decay', 1e-4)

        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim < 2 or 'bias' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        opt_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]

        self.opt = torch.optim.AdamW(opt_groups, lr=base_lr)
        
        dev_type = self.device.type if hasattr(self.device, 'type') else str(self.device)
        self.scaler = torch.amp.GradScaler(device=dev_type, enabled=(dev_type == 'cuda'))

        if dist.is_available() and dist.is_initialized():
            use_fsdp = self.cfg['training'].get('use_fsdp', False)
            if use_fsdp:
                # FSDP wrapping
                self.model = setup_fsdp(self.model, self.device, self.cfg)
                logger.info(f"Wrapped model in FSDP (Rank {get_rank()})")
            else:
                # Standard DDP wrapping
                if hasattr(self.model, 'to'):
                    self.model = self.model.to(self.device)
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, 
                    device_ids=[self.device] if self.device.type == 'cuda' else None,
                    find_unused_parameters=self.cfg['training'].get('find_unused_parameters', False)
                )
                logger.info(f"Wrapped model in DDP (Rank {get_rank()})")

    def train(self):
        epochs = self.cfg['training'].get('epochs', 10)
        patience = self.cfg['training'].get('patience', 10)
        best_metric = -float('inf')
        wait = 0
        
        start_ep = 1
        if self.cfg['training'].get('resume_path'):
            start_ep, best_metric = self._load_checkpoint(self.cfg['training']['resume_path'])

        logger.info(f"Starting training from epoch {start_ep}")
        
        for ep in range(start_ep, epochs + 1):
            
            if hasattr(self.ld_tr, 'sampler') and hasattr(self.ld_tr.sampler, 'set_epoch'):
                self.ld_tr.sampler.set_epoch(ep)
                
            avg_loss, loss_comps = self._run_epoch(ep, epochs)
            val_result = self._validate()
            
            self._log_epoch(ep, avg_loss, loss_comps, val_result)
            
            # Assuming validation result is a scalar or the primary metric is the first element
            score = val_result if not isinstance(val_result, tuple) else val_result[0]
            is_best = False
            if score > best_metric:
                best_metric = score
                wait = 0
                is_best = True
            else:
                wait += 1
            
            self._save_checkpoint(ep, best_metric, is_best=is_best)
            
            if wait >= patience:
                logger.info(f"⏹ Early stop triggered. No improvement for {patience} epochs.")
                break

    def _run_epoch(self, ep, max_ep):
        self.model.train()
        run_loss = 0.0
        loss_components = {}
        
        pbar = tqdm(self.ld_tr, desc=f"Epoch {ep}/{max_ep}", mininterval=2.0)
        self.opt.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(pbar):
            loss, batch_comps = self._process_batch(batch)
            
            scaled_loss = loss / self.accum_steps
            self.scaler.scale(scaled_loss).backward()
            
            if (batch_idx + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
            
            run_loss += loss.item()
            for k, v in batch_comps.items():
                if k not in loss_components:
                    loss_components[k] = 0.0
                loss_components[k] += v
            
            postfix = {"loss": f"{loss.item():.4f}"}
            postfix.update({k: f"{v:.4f}" for k, v in batch_comps.items()})
            pbar.set_postfix(postfix)

            if is_main_process() and WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({"train/loss": loss.item(), **{f"train/{k}": v for k, v in batch_comps.items()}})

        if len(self.ld_tr) % self.accum_steps != 0:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)

        if dist.is_available() and dist.is_initialized():
             run_loss_t = torch.tensor(run_loss, device=self.device)
             run_loss = reduce_tensor(run_loss_t).item()
             for k in loss_components:
                 comp_t = torch.tensor(loss_components[k], device=self.device)
                 loss_components[k] = reduce_tensor(comp_t).item()

        avg_loss = run_loss / len(self.ld_tr)
        avg_comps = {k: v / len(self.ld_tr) for k, v in loss_components.items()}
        return avg_loss, avg_comps

    def _process_batch(self, batch):
        dev_type = self.device.type if hasattr(self.device, 'type') else str(self.device)
        with torch.amp.autocast(device_type=dev_type):

            if isinstance(batch, dict):
                inputs = batch[self.input_key].to(self.device)
                targets = batch[self.target_key].to(self.device)
            else:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)

            outputs = self.model(inputs)
            
            loss = 0.0
            comps = {}
            for loss_name, criterion in self.criterions.items():
                l = criterion(outputs, targets)
                loss += l
                comps[loss_name] = l.item()

        return loss, comps

    def _validate(self):
        self.model.eval()
        for m in self.metrics.values(): 
             if hasattr(m, 'reset'): m.reset()
        
        dev_type = self.device.type if hasattr(self.device, 'type') else str(self.device)
        total_loss = 0.0
        batches = 0
        
        with torch.no_grad(), torch.amp.autocast(device_type=dev_type):
            for batch in tqdm(self.ld_va, desc="Validating", mininterval=2.0, leave=False):
                if isinstance(batch, dict):
                    inputs = batch[self.input_key].to(self.device)
                    targets = batch[self.target_key].to(self.device)
                else:
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                
                outputs = self.model(inputs)
                
                # Compute loss purely for logging
                batch_loss = sum(c(outputs, targets).item() for c in self.criterions.values())
                total_loss += batch_loss
                batches += 1
                
                # Update metrics
                for metric in self.metrics.values():
                    metric.update(outputs, targets)
                    
        avg_loss = total_loss / max(1, batches)
        
        metric_results = {}
        for k, metric in self.metrics.items():
             if hasattr(metric, 'compute'):
                  metric_results[k] = float(metric.compute().cpu())
        
        if not metric_results:
             return -avg_loss
             
        return list(metric_results.values())[0]

    def _save_checkpoint(self, ep, metric, is_best=False):
        """Delegates all saving responsibility to the State Manager."""
        if not is_main_process():
            return
            
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        if self.state:
            self.state.current_epoch = ep
            self.state.best_metric = metric
            self.state.save(model_to_save, self.opt, self.scaler, is_best=is_best)

    def _load_checkpoint(self, path):
        """Delegates loading to the State Manager."""
        if self.state:
            self.state.load(path, self.model, optimizer=self.opt, scaler=self.scaler)
            return self.state.current_epoch, self.state.best_metric
        return 1, -float('inf')

    def _log_epoch(self, ep, loss, comps, val_res):
        if not is_main_process():
            return
            
        train_strs = [f"{k}={v:.4f}" for k, v in comps.items()]
        msg_train = f"E{ep:03d} Train loss={loss:.4f} " + " ".join(train_strs)
        
        if isinstance(val_res, dict):
            val_strs = [f"{k}={v:.4f}" for k, v in val_res.items()]
            msg_val = f"E{ep:03d} Valid " + " ".join(val_strs)
        else:
             msg_val = f"E{ep:03d} Valid Score={val_res:.4f}"
            
        logger.info(msg_train)
        logger.info(msg_val)
        
        if WANDB_AVAILABLE and wandb.run is not None:
            log_dict = {"val/loss": loss}
            if isinstance(val_res, dict):
                 for k, v in val_res.items():
                      log_dict[f"val/{k}"] = v
            else:
                 log_dict["val/score"] = val_res
                 
            wandb.log(log_dict)