import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging import get_logger
from src.inference.infer import setup_model, _preprocess_video
from src.inference.video import load_video, sliding_window_inference
from src.tta.auditor import SelfAuditor

logger = get_logger("CALIBRATE")

def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate SelfAuditor threshold on validation set")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="datasets/echonet-dynamic", help="Dataset directory containing FileListwFrames112.csv and Videos/")
    parser.add_argument("--audit_stats", type=str, default="audit_stats.json", help="Output JSON for calibration stats")
    parser.add_argument("--max_samples", type=int, default=50, help="Maximum number of videos to use for calibration")
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Calibration starting on {device}")
    
    model, cfg = setup_model(args.config, args.checkpoint, device)
    img_size = tuple(cfg.get('data', {}).get('img_size', [224, 224]))
    clip_len = cfg.get('model', {}).get('max_clip_len', 16)
    
    csv_path = os.path.join(args.data_dir, "FileList.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(args.data_dir, "FileListwFrames112.csv") # Fallback
        
    if not os.path.exists(csv_path):
        logger.error(f"Dataset CSV not found in {args.data_dir}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Optional: Filter to val/test set
    if "Split" in df.columns:
        df = df[df["Split"] == "VAL"]
        
    video_dir = os.path.join(args.data_dir, "Videos")
    
    auditor = SelfAuditor(device=device)
    entropies_list = []
    max_ent_val = np.log(2.0)
    
    count = 0
    for _, row in tqdm(df.iterrows(), total=min(len(df), args.max_samples)):
        if args.max_samples > 0 and count >= args.max_samples:
            break
            
        fname = str(row['FileName'])
        if not fname.lower().endswith('.avi'):
             fname += ".avi"
             
        v_path = os.path.join(video_dir, fname)
        if not os.path.exists(v_path):
            continue
            
        try:
            frames, _ = load_video(v_path, img_size=img_size)
            if not frames: continue
            
            video_tensor = _preprocess_video(frames, device)
            model.eval()
            _, _, T, H, W = video_tensor.shape
            
            with torch.no_grad():
                for start_idx in range(0, T, clip_len):
                    end_idx = min(start_idx + clip_len, T)
                    chunk = video_tensor[:, :, start_idx:end_idx, :, :]
                    
                    if chunk.shape[2] < clip_len: # pad
                        pad_tensor = chunk[:, :, -1:].expand(-1, -1, clip_len - chunk.shape[2], -1, -1)
                        chunk = torch.cat([chunk, pad_tensor], dim=2)
                        
                    outputs = model(chunk)
                    mask_logits = outputs["mask_logits"].squeeze(0).squeeze(0) # (T_clip, H, W)
                    
                    # Compute entropy on raw logits
                    for t in range(min(clip_len, end_idx - start_idx)):
                        ent, mx = auditor._compute_entropy(mask_logits[t].unsqueeze(0), is_logits=True)
                        entropies_list.append(ent.cpu().item())
                        max_ent_val = mx.cpu().item()
                        
            count += 1
        except Exception as e:
            logger.warning(f"Error processing {fname}: {e}")
            continue

    if not entropies_list:
        logger.error("No data collected for calibration!")
        return
        
    all_entropies = np.array(entropies_list)
    norm_entropies = all_entropies / max_ent_val
    
    # Entropy-only formulation since features are not extracted yet
    scores = norm_entropies
    
    epsilon = float(np.mean(scores) + 3 * np.std(scores))
    
    stats = {
        "mu_source": None,
        "epsilon": epsilon,
        "max_ent": float(max_ent_val)
    }
    
    with open(args.audit_stats, 'w') as f:
        json.dump(stats, f, indent=4)
        
    logger.info(f"Calibration completed on {count} videos.")
    logger.info(f"Saved stats to {args.audit_stats}")
    logger.info(f"Calculated Epsilon: {epsilon:.4f}")

if __name__ == "__main__":
    main()
