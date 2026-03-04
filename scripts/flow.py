import os
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm

from src.flow.raft_flow import RAFTFlowEstimator

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
    parser = argparse.ArgumentParser(description="Precompute RAFT Optical Flow for EchoNet")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing .avi files (usually EchoNet-Dynamic/Videos)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .pt flow files")
    parser.add_argument("--img_size", type=int, nargs=2, default=[112, 112], help="Target size (H, W). Must match training config.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for RAFT model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Initializing RAFT Estimator on {args.device}...")
    estimator = RAFTFlowEstimator(device=args.device)
    
    videos = [f for f in os.listdir(args.video_dir) if f.lower().endswith('.avi')]
    print(f"Found {len(videos)} videos. Beginning flow extraction.")
    
    img_size = tuple(args.img_size)
    
    success_count = 0
    fail_count = 0
    
    for video_name in tqdm(videos):
        vid_path = os.path.join(args.video_dir, video_name)
        out_path = os.path.join(args.output_dir, video_name.replace('.avi', '.pt'))
        
        if os.path.exists(out_path):
            continue # Skip already processed
            
        video_tensor = load_video_frames(vid_path, img_size=img_size)
        if video_tensor is None or len(video_tensor) < 2:
            print(f"Skipping {video_name} - unreadable or too short (<2 frames).")
            fail_count += 1
            continue
            
        try:
            # Output will be on CPU
            flow = estimator.compute_dense_flow(video_tensor, batch_size=args.batch_size)
            
            # Convert to float16 to save disk space
            flow = flow.half()
            
            # Save
            torch.save(flow, out_path)
            success_count += 1
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            fail_count += 1
            
    print(f"Done. Successfully processed {success_count} videos. Failed: {fail_count}.")

if __name__ == "__main__":
    main()
