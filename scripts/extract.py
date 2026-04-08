import sys
import os
import argparse
import glob
import numpy as np
import tqdm
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

from src.inference.infer import setup_model, _preprocess_video, get_device
from src.inference.video import load_video, sliding_window_inference
from src.utils.logging import get_logger

logger = get_logger("EXTRACT")

def parse_args():
    parser = argparse.ArgumentParser(description="Extract pseudomasks for all EchoNet-Dynamic videos")
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration YAML file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.pt) file.")
    parser.add_argument("--data_dir", type=str, default="datasets/echonet-dynamic", help="Path to the dataset directory containing the Videos folder.")
    parser.add_argument("--output_dir", type=str, default="datasets/echonet-dynamic/pseudomasks", help="Path to the directory to save extracted masks.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for sliding window inference.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Distributed setup
    rank = 0
    world_size = 1
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            device_id = local_rank % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
        
    device = get_device()
    if rank == 0:
        logger.info(f"Using device: {device} (Total World Size: {world_size})")
    
    model, cfg = setup_model(args.config, args.checkpoint, device)
    
    img_size = tuple(cfg.get('data', {}).get('img_size', [224, 224]))
    clip_len = cfg.get('model', {}).get('max_clip_len', 16)
    
    videos_dir = os.path.join(args.data_dir, "Videos")
    if not os.path.exists(videos_dir):
        if rank == 0:
            logger.error(f"Videos directory not found at {videos_dir}")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_video_files = sorted(glob.glob(os.path.join(videos_dir, "*.avi")) + glob.glob(os.path.join(videos_dir, "*.mp4")))
    # Partition videos
    video_files = all_video_files[rank::world_size]
    
    if rank == 0:
        logger.info(f"Found {len(all_video_files)} videos total. Rank {rank} processing {len(video_files)} videos.")
    
    pbar = tqdm.tqdm(video_files, desc=f"Rank {rank} extracting", disable=False)
    
    for video_path in pbar:
        fname = Path(video_path).stem
        output_mask_path = os.path.join(args.output_dir, f"{fname}.npy")
        
        if os.path.exists(output_mask_path):
            continue
            
        try:
            frames, fps = load_video(video_path, img_size=img_size)
            if not frames:
                logger.warning(f"No frames loaded for {fname}, skipping.")
                continue
                
            video_tensor = _preprocess_video(frames, device)
            
            full_masks, _, _, _ = sliding_window_inference(
                model, video_tensor, 
                clip_len=clip_len, 
                overlap=0,
                batch_size=args.batch_size,
                device=device,
                auditor=None
            )
            
            np.save(output_mask_path, full_masks)
        except Exception as e:
            logger.error(f"Failed to process {fname} on rank {rank}: {e}")

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
