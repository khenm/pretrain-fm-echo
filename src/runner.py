import os
import yaml
import torch
import glob
from datetime import datetime
from src.utils.logging import get_logger
from src.utils.env import load_checkpoint
from src.registry import build_model, build_loss
from src.utils.dist import is_main_process

logger = get_logger()

def resolve_resume_path(resume_mode, runs_root, model_name, cfg_resume_path=None):
    if cfg_resume_path:
        if os.path.exists(cfg_resume_path):
             logger.info(f"Resuming from config path: {cfg_resume_path}")
             return cfg_resume_path
        else:
             logger.warning(f"Configured resume_path {cfg_resume_path} not found. Falling back to logic.")

    if not resume_mode:
        return None
        
    if isinstance(resume_mode, str) and resume_mode != 'auto' and "/" in resume_mode:
        if os.path.exists(resume_mode):
            return resume_mode
        else:
            logger.warning(f"Explicit resume path {resume_mode} not found. Starting fresh.")
            return None
    
    model_runs_dir = os.path.join(runs_root, model_name)
    if not os.path.exists(model_runs_dir):
        logger.info(f"No previous runs found for {model_name}. Starting fresh.")
        return None
    
    runs = sorted([
        os.path.join(model_runs_dir, d) for d in os.listdir(model_runs_dir) 
        if os.path.isdir(os.path.join(model_runs_dir, d))
    ], key=os.path.getmtime, reverse=True)
    
    for run in runs:
        last_ckpt = os.path.join(run, "last.ckpt")
        if os.path.exists(last_ckpt):
            logger.info(f"Auto-Discovery: Found resume point at {last_ckpt}")
            return last_ckpt
            
    logger.info("Auto-Discovery: No 'last.ckpt' found in recent history. Starting fresh.")
    return None

def _setup_workspace(cfg, args_resume):
    model_name = cfg['model'].get('name', 'GenericModel')
    runs_root = cfg['training'].get('save_dir', 'runs')
    os.makedirs(runs_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_root = cfg['training'].get('checkpoint_dir', 'checkpoints')
    os.makedirs(ckpt_root, exist_ok=True)
    vault_dir = os.path.join(ckpt_root, model_name)
    os.makedirs(vault_dir, exist_ok=True)
    resume_mode = 'auto' if args_resume else cfg['training'].get('resume_mode', None)
    cfg_resume_path = cfg['training'].get('resume_path')
    resume_path = resolve_resume_path(resume_mode, runs_root, model_name, cfg_resume_path)
    
    if resume_path:
        run_dir = os.path.dirname(resume_path)
        logger.info(f"Resuming workspace: {run_dir}")
        cfg['training']['resume_path'] = resume_path
        run_dir = os.path.join(runs_root, model_name, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Created new workspace for resume session: {run_dir}")
    else:
        run_dir = os.path.join(runs_root, model_name, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Created new workspace: {run_dir}")
    
    cfg['training']['run_dir'] = run_dir
    cfg['training']['vault_dir'] = vault_dir
    cfg['training']['run_id'] = timestamp
    
    return run_dir, vault_dir, timestamp

def _setup_wandb(cfg, run_dir, timestamp):
    if not is_main_process():
        return

    if cfg.get('wandb', {}).get('enable', True):
        try:
            import wandb
        except ImportError:
            logger.warning("wandb is not installed. Disabling wandb logging.")
            return

        project_name = cfg.get('wandb', {}).get('project', 'pytorch-template')
        
        if cfg['model'].get('name'):
            run_name = f"{cfg['model']['name']}_{timestamp}"
        else:
            run_name = f"run_{timestamp}"
            
        wandb.init(
            project=project_name,
            config=cfg,
            name=run_name,
            dir=run_dir,
            resume="allow",
            id=cfg['training'].get('run_id')
        )

def run_init(state, args_resume=False):
    cfg = state.config
    run_dir, vault_dir, timestamp = _setup_workspace(cfg, args_resume)

    state.run_dir = run_dir
    state.vault_path = os.path.join(vault_dir, f"{timestamp}_best.ckpt")
    
    log_file = os.path.join(run_dir, ".log")
    
    global logger
    logger = get_logger(log_file=log_file)
    _setup_wandb(cfg, run_dir, timestamp)
    
    with open(os.path.join(run_dir, "config.yaml"), 'w') as f:
        yaml.dump(cfg, f)
    
    logger.info("Initializing...")
    logger.info(f"Workspace: {run_dir}")
    logger.info(f"Vault: {vault_dir}")
    
    logger.info(f"Using device: {state.device}")
    
    cfg['training']['ckpt_save_path'] = os.path.join(run_dir, "last.ckpt")
    
    return state.device

def load_model_for_inference(cfg, device):
    model_name = cfg['model'].get('name', 'GenericModel')
    
    if cfg['training'].get('resume_path'):
         ckpt_path = cfg['training']['resume_path']
         if not os.path.exists(ckpt_path):
             logger.warning(f"Explicit path {ckpt_path} not found. Falling back to auto-discovery.")
         else:
             logger.info(f"Using explicit checkpoint: {ckpt_path}")
             model = build_model(cfg, device)
             load_checkpoint(model, ckpt_path, device)
             model.eval()
             return model

    vault_dir = os.path.join(cfg['training'].get('checkpoint_dir', 'checkpoints'), model_name)
    best_ckpts = glob.glob(os.path.join(vault_dir, "*best.ckpt"))
    
    if best_ckpts:
        best_ckpts.sort(key=os.path.getmtime, reverse=True)
        ckpt_path = best_ckpts[0]
        logger.info(f"Found BEST checkpoint: {ckpt_path}")
    else:
        logger.info("No best checkpoint found. Searching for latest last.ckpt...")
        runs_root = cfg['training'].get('save_dir', 'runs')
        search_pattern = os.path.join(runs_root, model_name, "*", "last.ckpt")
        last_ckpts = glob.glob(search_pattern)
        
        if last_ckpts:
            last_ckpts.sort(key=os.path.getmtime, reverse=True)
            ckpt_path = last_ckpts[0]
            logger.info(f"Found LATEST checkpoint: {ckpt_path}")
        else:
            logger.error(f"No checkpoints found for model {model_name} in {vault_dir} or {runs_root}.")
            return None
            
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint path determined but not found: {ckpt_path}")
        return None
        
    model = build_model(cfg, device)
    logger.info(f"Loading checkpoint: {ckpt_path}")
    load_checkpoint(model, ckpt_path, device)
    model.eval()
    return model

def get_criterions(cfg):
    """
    Returns a dictionary of criteria. By default we look up standard losses from our registry.
    """
    criterions = {}
    loss_cfg = cfg.get('loss', {})
    losses_to_build = loss_cfg.get('types', ['CrossEntropy'])
    for l_name in losses_to_build:
        try:
             criterions[l_name.lower()] = build_loss(l_name, **loss_cfg.get('kwargs', {}))
        except KeyError:
             logger.warning(f"Loss {l_name} not found in registry. Using default CrossEntropyLoss.")
             criterions[l_name.lower()] = torch.nn.CrossEntropyLoss()
             
    if not criterions:
        logger.info("No losses configured, defaulting to CrossEntropyLoss.")
        criterions['ce'] = torch.nn.CrossEntropyLoss()

    return criterions

def get_metrics(cfg):
    """
    Returns a dictionary of metrics for evaluation.
    """
    metrics = {}
    try:
        from torchmetrics import Accuracy
        num_classes = cfg['data'].get('num_classes', 10)
        task = "multiclass" if num_classes > 2 else "binary"
        metrics['acc'] = Accuracy(task=task, num_classes=num_classes)
    except ImportError:
        logger.warning("torchmetrics not installed. Metrics will be empty.")
    
    return metrics
