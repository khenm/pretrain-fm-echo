import os
import argparse
import sys
import yaml

# Add project root to sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging import get_logger
from src.runner import run_init, get_criterions, get_metrics
from src.registry import build_model, get_dataloaders
from src.trainer import Trainer
from src.utils.env import TrainerState
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser(description="PyTorch Generic Training Template")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the config file.")
    parser.add_argument("--resume", action="store_true", help="Auto-resume from the latest checkpoint.")
    
    # Simple DDP arguments
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", -1)))
    args = parser.parse_args()

    # DDP Initialization
    if args.local_rank != -1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
    
    # Init state
    state = TrainerState(args.config, seed=42, device_id=max(0, args.local_rank))
    cfg = state.config

    # Init workspace, logging, seed, wandb, returns device
    device = run_init(state, args_resume=args.resume)
    logger = get_logger()
    
    # Build Dataset/Dataloaders
    logger.info("Building dataloaders...")
    loaders = get_dataloaders(cfg)
    
    # Build Model
    logger.info("Initializing model...")
    model = build_model(cfg, device)
    
    # Build Losses and Metrics
    logger.info("Setting up losses and metrics...")
    criterions = get_criterions(cfg)
    metrics = get_metrics(cfg)
    
    # Create Trainer and start
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        loaders=loaders,
        cfg=cfg,
        device=device,
        criterions=criterions,
        metrics=metrics,
        state=state
    )
    
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed.")

if __name__ == "__main__":
    main()
