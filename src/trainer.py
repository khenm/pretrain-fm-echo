import os
import random
import numpy as np
import torch
import torch.nn.functional as F
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

class LossWeightScheduler:
    """
    Dynamically scales loss component weights during training to prevent 
    unstable gradients from complex compound metrics (like ratios) early on.
    """
    def __init__(self, criterions_dict, target_attr='ef_weight', start_epoch=10, end_epoch=20, max_weight=1.0):
        self.criterions_dict = criterions_dict
        self.target_attr = target_attr
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.max_weight = max_weight
        self.current_weight = 0.0

    def step(self, epoch):
        # Calculate the linear warmup interpolation
        if epoch < self.start_epoch:
            new_weight = 0.0
        elif epoch >= self.end_epoch:
            new_weight = self.max_weight
        else:
            progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            new_weight = self.max_weight * progress

        # Apply the weight if it has changed
        if new_weight != self.current_weight:
            self.current_weight = new_weight
            for name, criterion in self.criterions_dict.items():
                if hasattr(criterion, self.target_attr):
                    # Handle both standard floats and registered buffer tensors
                    if isinstance(getattr(criterion, self.target_attr), torch.Tensor):
                        getattr(criterion, self.target_attr).fill_(self.current_weight)
                    else:
                        setattr(criterion, self.target_attr, self.current_weight)
                    
                    logger.info(f"\u2696\ufe0f Loss Topology Update: Set '{name}' {self.target_attr} to {self.current_weight:.4f}")

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
        scheduler_cfg = self.cfg.get('training', {}).get('ef_warmup', {})
        self.loss_scheduler = LossWeightScheduler(
            criterions_dict=self.criterions,
            target_attr='ef_weight',
            start_epoch=scheduler_cfg.get('start_epoch', 10),
            end_epoch=scheduler_cfg.get('end_epoch', 20),
            max_weight=self.cfg.get('loss', {}).get('kwargs', {}).get('ef_weight_target', 1.0)
        )
        self.val_metrics = metrics or {}
        import copy
        self.train_metrics = {k: v.clone() if hasattr(v, 'clone') else copy.deepcopy(v) for k, v in self.val_metrics.items()}

        # Fetch dynamic dataset keys to prevent magic string failures
        self.input_key = self.cfg.get('data', {}).get('input_key', 'image')
        self.target_key = self.cfg.get('data', {}).get('target_key', 'label')

        self.num_classes = self.cfg.get('data', {}).get('num_classes', 10)
        self._setup_optimization()

        for k, metric in self.train_metrics.items():
            if hasattr(metric, 'to'):
                metric.to(self.device)
        for k, metric in self.val_metrics.items():
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
                self.model = setup_fsdp(self.model, self.device, self.cfg)
                logger.info(f"Wrapped model in FSDP (Rank {get_rank()})")
            else:
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
            self.loss_scheduler.step(ep)
            
            if hasattr(self.ld_tr, 'sampler') and hasattr(self.ld_tr.sampler, 'set_epoch'):
                self.ld_tr.sampler.set_epoch(ep)

            avg_loss, loss_comps, train_metrics_res = self._run_epoch(ep, epochs)
            val_result = self._validate()

            self._log_epoch(ep, avg_loss, loss_comps, train_metrics_res, val_result)

            score = val_result.get('dice', list(val_result.values())[0])
            is_best = False
            if score > best_metric:
                best_metric = score
                wait = 0
                is_best = True
            else:
                wait += 1

            self._save_checkpoint(ep, best_metric, is_best=is_best)

            if wait >= patience:
                logger.info(f"\u23f9 Early stop triggered. No improvement for {patience} epochs.")
                break

    def _run_epoch(self, ep, max_ep):
        self.model.train()
        for m in self.train_metrics.values():
            if hasattr(m, 'reset'):
                m.reset()
        run_loss = 0.0
        loss_components = {}

        pbar = tqdm(self.ld_tr, desc=f"Epoch {ep}/{max_ep}", mininterval=2.0)
        self.opt.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar):
            with torch.amp.autocast(device_type=self.device.type if hasattr(self.device, 'type') else str(self.device)):
                loss, batch_comps, outputs, targets = self._process_batch(batch)
                self._update_metrics(outputs, targets, metrics_dict=self.train_metrics)

            scaled_loss = loss / self.accum_steps
            
            is_final_accum_step = (batch_idx + 1) % self.accum_steps == 0 or (batch_idx + 1) == len(self.ld_tr)

            if not is_final_accum_step and hasattr(self.model, "no_sync"):
                with self.model.no_sync():
                    self.scaler.scale(scaled_loss).backward()
            else:
                self.scaler.scale(scaled_loss).backward()

            if is_final_accum_step:
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

        if dist.is_available() and dist.is_initialized():
            run_loss_t = torch.tensor(run_loss, device=self.device)
            run_loss = reduce_tensor(run_loss_t).item()
            for k in loss_components:
                val = loss_components[k]
                if isinstance(val, torch.Tensor):
                    comp_t = val.detach().clone().to(self.device)
                else:
                    comp_t = torch.tensor(val, device=self.device)
                loss_components[k] = reduce_tensor(comp_t).item()

        avg_loss = run_loss / len(self.ld_tr)
        avg_comps = {k: v / len(self.ld_tr) for k, v in loss_components.items()}
        train_metric_results = self._aggregate_metrics(self.train_metrics)
        return avg_loss, avg_comps, train_metric_results

    def _process_batch(self, batch):
        if isinstance(batch, dict):
            inputs = batch[self.input_key].to(self.device)
            if self.target_key == 'batch':
                targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            else:
                targets = batch[self.target_key].to(self.device)
        else:
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)

        outputs = self.model(inputs)

        loss = 0.0
        comps = {}
        for loss_name, criterion in self.criterions.items():
            result = criterion(outputs, targets)
            if isinstance(result, tuple):
                l, sub_comps = result
                loss += l
                comps.update(sub_comps)
            else:
                loss += result
                comps[loss_name] = result.item()

        return loss, comps, outputs, targets

    def _validate(self):
        self.model.eval()
        for m in self.val_metrics.values():
            if hasattr(m, 'reset'):
                m.reset()

        dev_type = self.device.type if hasattr(self.device, 'type') else str(self.device)
        
        run_loss = 0.0
        loss_components = {}

        with torch.no_grad(), torch.amp.autocast(device_type=dev_type):
            for batch in tqdm(self.ld_va, desc="Validating", mininterval=2.0, leave=False):
                loss, batch_comps, outputs, targets = self._process_batch(batch)
                self._update_metrics(outputs, targets, metrics_dict=self.val_metrics)
                
                run_loss += loss.item()
                for k, v in batch_comps.items():
                    if k not in loss_components:
                        loss_components[k] = 0.0
                    loss_components[k] += v

        if dist.is_available() and dist.is_initialized():
            run_loss_t = torch.tensor(run_loss, device=self.device)
            run_loss = reduce_tensor(run_loss_t).item()
            for k in loss_components:
                val = loss_components[k]
                if isinstance(val, torch.Tensor):
                    comp_t = val.detach().clone().to(self.device)
                else:
                    comp_t = torch.tensor(val, device=self.device)
                loss_components[k] = reduce_tensor(comp_t).item()
                
        avg_loss = run_loss / max(1, len(self.ld_va))
        avg_comps = {k: v / max(1, len(self.ld_va)) for k, v in loss_components.items()}

        val_res = self._aggregate_metrics(self.val_metrics)
        val_res['loss'] = avg_loss
        for k, v in avg_comps.items():
            val_res[k] = v
            
        return val_res

    def _update_metrics(self, outputs, targets, metrics_dict):
        if not isinstance(outputs, dict) or not isinstance(targets, dict):
            return

        outputs = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}

        mask_logits = outputs.get('mask_logits')
        vol_curve = outputs.get('vol_curve')
        
        if mask_logits is None or vol_curve is None:
            return

        target_edv = targets.get('target_edv')
        target_esv = targets.get('target_esv')
        target_ef = targets.get('target_ef')
        frame_mask = targets.get('frame_mask')

        B, T = vol_curve.shape

        # --- EF metrics ---
        if target_ef is not None and 'mae' in metrics_dict:
            pred_edv = outputs.get('pred_edv')
            pred_esv = outputs.get('pred_esv')
            if pred_edv is not None and pred_esv is not None:
                pred_ef = torch.where(
                    pred_edv > 1e-6,
                    (pred_edv - pred_esv) / pred_edv,
                    torch.zeros_like(pred_edv)
                )
                valid_ef = (target_ef >= 0)
                if valid_ef.any():
                    metrics_dict['mae'](pred_ef[valid_ef], target_ef[valid_ef])
                    if 'rmse' in metrics_dict:
                        metrics_dict['rmse'](pred_ef[valid_ef], target_ef[valid_ef])
                    if 'r2' in metrics_dict:
                        metrics_dict['r2'](pred_ef[valid_ef], target_ef[valid_ef])

        # --- EDV / ESV volume metrics ---
        if target_edv is not None and 'mae_edv' in metrics_dict:
            pred_edv = outputs.get('pred_edv')
            pred_esv = outputs.get('pred_esv')
            
            if pred_edv is not None and pred_esv is not None:
                valid_edv = (target_edv >= 0)
                if valid_edv.any():
                    p_edv_ml = pred_edv[valid_edv] * 300.0
                    t_edv_ml = target_edv[valid_edv] * 300.0
                    metrics_dict['mae_edv'](p_edv_ml, t_edv_ml)
                    metrics_dict['rmse_edv'](p_edv_ml, t_edv_ml)
                    metrics_dict['r2_edv'](p_edv_ml, t_edv_ml)

                valid_esv = (target_esv >= 0)
                if valid_esv.any():
                    p_esv_ml = pred_esv[valid_esv] * 300.0
                    t_esv_ml = target_esv[valid_esv] * 300.0
                    metrics_dict['mae_esv'](p_esv_ml, t_esv_ml)
                    metrics_dict['rmse_esv'](p_esv_ml, t_esv_ml)
                    metrics_dict['r2_esv'](p_esv_ml, t_esv_ml)

        # --- Dice metric ---
        if 'dice' in metrics_dict:
            target_masks = targets.get('label')
            if target_masks is not None:
                pred_probs = torch.sigmoid(mask_logits)
                if pred_probs.shape[-2:] != target_masks.shape[-2:]:
                    Bm, C, Tm, H, W = pred_probs.shape
                    pred_probs = F.interpolate(
                        pred_probs.view(Bm, C * Tm, H, W),
                        size=target_masks.shape[-2:],
                        mode='bilinear', align_corners=False
                    ).view(Bm, C, Tm, *target_masks.shape[-2:])

                pred_binary = (pred_probs > 0.5).int()
                target_binary = target_masks.int()

                if frame_mask is not None:
                    Bf, Tf = frame_mask.shape[0], frame_mask.shape[1]
                    mask_flat = frame_mask.view(Bf * Tf)
                    valid_idx = torch.nonzero(mask_flat).squeeze(-1)
                    if valid_idx.numel() > 0:
                        pred_flat = pred_binary.permute(0, 2, 1, 3, 4).reshape(
                            Bf * Tf, pred_binary.shape[1], *pred_binary.shape[-2:]
                        )
                        target_flat = target_binary.permute(0, 2, 1, 3, 4).reshape(
                            Bf * Tf, target_binary.shape[1], *target_binary.shape[-2:]
                        )
                        metrics_dict['dice'](pred_flat[valid_idx], target_flat[valid_idx])
                else:
                    Bm2, Tm2 = pred_binary.shape[0], pred_binary.shape[2]
                    pred_flat = pred_binary.permute(0, 2, 1, 3, 4).reshape(
                        Bm2 * Tm2, pred_binary.shape[1], *pred_binary.shape[-2:]
                    )
                    target_flat = target_binary.permute(0, 2, 1, 3, 4).reshape(
                        Bm2 * Tm2, target_binary.shape[1], *target_binary.shape[-2:]
                    )
                    metrics_dict['dice'](pred_flat, target_flat)

        # --- Phase accuracy ---
        if 'phase_acc' in metrics_dict:
            phase_logits = outputs.get('phase_logits')
            if phase_logits is not None and frame_mask is not None:
                phase_targets = frame_mask.long()

                if phase_logits.shape[2] != phase_targets.shape[1]:
                    phase_logits = F.interpolate(
                        phase_logits, 
                        size=phase_targets.shape[1], 
                        mode='linear', 
                        align_corners=False
                    )

                phase_preds = phase_logits.argmax(dim=1)

                valid_phase = phase_targets > 0
                if valid_phase.any():
                    metrics_dict['phase_acc'](
                        phase_preds[valid_phase],
                        phase_targets[valid_phase]
                    )

    def _aggregate_metrics(self, metrics_dict):
        results = {}
        for k, metric in metrics_dict.items():
            if hasattr(metric, 'aggregate'):
                results[k] = float(metric.aggregate())
            elif hasattr(metric, 'compute'):
                try:
                    results[k] = float(metric.compute().cpu())
                except (RuntimeError, ValueError):
                    results[k] = 0.0
            else:
                results[k] = 0.0
        return results

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

    def _log_epoch(self, ep, loss, comps, train_metrics, val_res):
        if not is_main_process():
            return

        train_strs = [f"{k}={v:.4f}" for k, v in comps.items()]
        t_metric_strs = [f"{k.upper()}={v:.4f}" for k, v in train_metrics.items()]
        if t_metric_strs:
            msg_train = f"E{ep:03d} Train loss={loss:.4f} " + " ".join(train_strs) + " " + " ".join(t_metric_strs)
        else:
            msg_train = f"E{ep:03d} Train loss={loss:.4f} " + " ".join(train_strs)
        logger.info(msg_train)

        val_strs = [f"{k.upper()}={v:.4f}" for k, v in val_res.items()]
        msg_val = f"E{ep:03d} Valid " + " ".join(val_strs)
        logger.info(msg_val)

        if WANDB_AVAILABLE and wandb.run is not None:
            log_dict = {"train/loss": loss, "epoch": ep}
            if hasattr(self, 'opt') and self.opt is not None and len(self.opt.param_groups) > 0:
                log_dict["train/lr"] = self.opt.param_groups[0]["lr"]
            for k, v in comps.items():
                log_dict[f"train/{k}"] = v
            for k, v in train_metrics.items():
                log_dict[f"train/{k}"] = v
            for k, v in val_res.items():
                log_dict[f"val/{k}"] = v
            wandb.log(log_dict)