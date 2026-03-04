import os
import argparse
import torch
import cv2
import numpy as np
import io
from tqdm import tqdm
import torch.distributed as dist

from src.flow.raft_flow import RAFTFlowEstimator
from src.utils.dist import setup_dist, cleanup_dist, get_rank, get_world_size, is_main_process

def load_video_frames(video_path, img_size=(112, 112)):
    """Loads all frames of a video and resizes them identically to the training dataloader."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (frame.shape[0], frame.shape[1]) != img_size:
            frame = cv2.resize(frame, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    
    if not frames:
        return None
        
    # (T, H, W, C) -> (T, C, H, W) -> float32 [0, 1]
    video_tensor = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).float() / 255.0
    return video_tensor

def main():
    parser = argparse.ArgumentParser(description="Precompute RAFT Optical Flow for EchoNet (Distributed)")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing .avi files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .pt flow files")
    parser.add_argument("--img_size", type=int, nargs=2, default=[112, 112], help="Target size (H, W)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for RAFT model")
    
    args = parser.parse_args()
    
    # Initialize Distributed Processing
    has_dist = False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # We are running under torchrun/DDP
        has_dist = True
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        # Fallback to single process
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    rank = get_rank()
    world_size = get_world_size()
    
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Distributed Flow Extraction: World Size = {world_size}")
    
    if has_dist:
        dist.barrier() # Wait for main process to create directories
        
    if is_main_process():
        print(f"Initializing RAFT Estimator on {device} (Main Process)...")
        
    estimator = RAFTFlowEstimator(device=device)
    
    all_videos = sorted([f for f in os.listdir(args.video_dir) if f.lower().endswith('.avi')])
    
    # Split workload across ranks
    rank_videos = all_videos[rank::world_size]
    
    if is_main_process():
        print(f"Found {len(all_videos)} total videos. Rank 0 processing {len(rank_videos)} videos.")
        
    img_size = tuple(args.img_size)
    
    success_count = 0
    fail_count = 0
    
    # Only show tqdm on main process to avoid messy logs
    pbar = tqdm(rank_videos, desc=f"Rank {rank}", disable=not is_main_process())
    
    for video_name in pbar:
        vid_path = os.path.join(args.video_dir, video_name)
        out_path = os.path.join(args.output_dir, video_name.replace('.avi', '.pt'))
        
        if os.path.exists(out_path):
            continue # Skip already processed
            
        video_tensor = load_video_frames(vid_path, img_size=img_size)
        if video_tensor is None or len(video_tensor) < 2:
            fail_count += 1
            continue
            
        try:
            flow = estimator.compute_dense_flow(video_tensor, batch_size=args.batch_size)
            flow = flow.half() # Convert to float16 to save disk space
            
            # Workaround for PyTorch inline_container.cc error on network drives
            buffer = io.BytesIO()
            torch.save(flow, buffer)
            with open(out_path, 'wb') as f:
                f.write(buffer.getvalue())
                
            success_count += 1
        except Exception as e:
            if is_main_process():
                print(f"\nError processing {video_name}: {e}")
            fail_count += 1
            
    # Gather statistics
    stats = torch.tensor([success_count, fail_count], dtype=torch.int32, device=device if torch.cuda.is_available() else 'cpu')
    if has_dist and dist.is_initialized() and torch.cuda.is_available(): # Tensors must be on CUDA for NCCL reduce
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        
    if is_main_process():
        print(f"Extraction Complete. Successful: {stats[0].item()}, Failed: {stats[1].item()}.")

    if has_dist:
        cleanup_dist()

if __name__ == "__main__":
    main()
