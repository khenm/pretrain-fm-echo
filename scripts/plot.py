import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging import get_logger
from src.inference.infer import setup_model, _preprocess_video, _get_gt_data
from src.inference.video import load_video, sliding_window_inference, overlay_mask
from src.tta.auditor import SelfAuditor

logger = get_logger("PLOT")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference and Plotting Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--result", type=str, default="results/plot.png", help="Output plot path")
    parser.add_argument("--data_dir", type=str, default="datasets/echonet-dynamic", help="Dataset directory")
    parser.add_argument("--audit", action="store_true", help="Enable SelfAuditor to calculate Martingale Wealth.")
    parser.add_argument("--audit_stats", type=str, default=None, help="Path to audit stats JSON file for calibrated eps and max entropies.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model, cfg = setup_model(args.config, args.checkpoint, device)
    
    img_size = tuple(cfg.get('data', {}).get('img_size', [224, 224]))
    clip_len = cfg.get('model', {}).get('max_clip_len', 16)
    
    logger.info(f"Loading video from {args.video} at size {img_size}")
    frames, fps = load_video(args.video, img_size=img_size)
    if not frames:
        logger.error("No valid frames loaded.")
        return
        
    T = len(frames)
    logger.info(f"Loaded {T} frames at {fps} fps")
    
    video_tensor = _preprocess_video(frames, device)
    
    auditor = None
    if getattr(args, "audit", False):
        auditor = SelfAuditor(device=device)
        audit_stats = getattr(args, "audit_stats", None)
        if audit_stats:
            auditor.load_stats(audit_stats)
            
    logger.info("Running sliding window inference...")
    full_masks, full_vol_curve, _, wealth_curve = sliding_window_inference(
        model, video_tensor, clip_len=clip_len, overlap=0, device=device, auditor=auditor
    )
    
    logger.info("Fetching ground truth data if available...")
    gt_mask, ed_frame, es_frame, edv, esv = _get_gt_data(args.video, args.data_dir, img_size, T)
    
    if ed_frame is None or ed_frame < 0 or ed_frame >= T:
        ed_frame = int(np.argmax(full_vol_curve))
        logger.info(f"Using max volume frame {ed_frame} as ED")
    else:
        logger.info(f"Ground truth ED frame: {ed_frame}")
        
    if es_frame is None or es_frame < 0 or es_frame >= T:
        es_frame = int(np.argmin(full_vol_curve))
        logger.info(f"Using min volume frame {es_frame} as ES")
    else:
        logger.info(f"Ground truth ES frame: {es_frame}")
        
    # Sample 4-5 frames per second
    sample_rate = max(1, int(fps / 4.5))
    sampled_indices = list(range(0, T, sample_rate))
    
    if ed_frame not in sampled_indices:
        sampled_indices.append(ed_frame)
    if es_frame not in sampled_indices:
        sampled_indices.append(es_frame)
        
    sampled_indices = sorted(list(set(sampled_indices)))
    logger.info(f"Sampling {len(sampled_indices)} frames for the top plot")
    
    # Prepare top image row
    small_frames = []
    target_h = 112
    for idx in sampled_indices:
        frame = frames[idx].copy()
        
        # Overlay predicted mask
        if idx < len(full_masks):
            frame = overlay_mask(frame, full_masks[idx], color=(0, 255, 0), alpha=0.5)
            

            
        cv2.putText(frame, f"F:{idx}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        if idx == ed_frame:
            cv2.putText(frame, "ED", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        if idx == es_frame:
            cv2.putText(frame, "ES", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
            
        h, w = frame.shape[:2]
        target_w = int(w * (target_h / h))
        frame_small = cv2.resize(frame, (target_w, target_h))
        small_frames.append(frame_small)
        
    if small_frames:
        top_image = np.hstack(small_frames)
    else:
        top_image = np.zeros((target_h, target_h, 3), dtype=np.uint8)
        
    # Plotting
    plot_width = max(12, len(sampled_indices) * 1.5)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(plot_width, 8), gridspec_kw={'height_ratios': [1, 2]})
    
    ax1.imshow(top_image)
    ax1.axis('off')
    ax1.set_title("Sampled Annotated Frames", fontsize=14)
    
    vol_curve_scaled = full_vol_curve * 300.0
    
    ax2.plot(range(T), vol_curve_scaled, color='blue', label='Predicted Volume (* 300)', linewidth=2.5)
    
    ax2.axvline(x=ed_frame, color='green', linestyle='--', label=f'ED (Frame {ed_frame})')
    ax2.axvline(x=es_frame, color='purple', linestyle='--', label=f'ES (Frame {es_frame})')
    
    pred_edv = vol_curve_scaled[ed_frame]
    pred_esv = vol_curve_scaled[es_frame]
    
    ax2.plot([ed_frame], [pred_edv], marker='o', color='green', markersize=10, zorder=5)
    ax2.annotate(f"{pred_edv:.1f}", (ed_frame, pred_edv), textcoords="offset points", xytext=(0, 10), ha='center', color='green', fontweight='bold')
    
    ax2.plot([es_frame], [pred_esv], marker='o', color='purple', markersize=10, zorder=5)
    ax2.annotate(f"{pred_esv:.1f}", (es_frame, pred_esv), textcoords="offset points", xytext=(0, 10), ha='center', color='purple', fontweight='bold')
    
    if edv is not None and esv is not None:
        ax2.plot([ed_frame], [edv], marker='X', color='darkgreen', markersize=12, zorder=5, label='GT EDV')
        ax2.plot([es_frame], [esv], marker='X', color='indigo', markersize=12, zorder=5, label='GT ESV')
        ax2.annotate(f"GT: {edv:.1f}", (ed_frame, edv), textcoords="offset points", xytext=(0, -15), ha='center', color='darkgreen', fontweight='bold')
        ax2.annotate(f"GT: {esv:.1f}", (es_frame, esv), textcoords="offset points", xytext=(0, -15), ha='center', color='indigo', fontweight='bold')

    ax2.set_xlabel("Frame Number", fontsize=12)
    ax2.set_ylabel("Volume", fontsize=12)
    ax2.set_title("Inferred Volume Curve", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right')
    
    ax2.set_xlim(0, max(T-1, 1))
    
    if wealth_curve is not None:
        ax3 = ax2.twinx()
        ax3.plot(range(T), wealth_curve, color='#9467bd', label='Wealth', alpha=0.5, linestyle=':', linewidth=2)
        ax3.set_ylabel('Wealth (OOD)', color='#9467bd', fontsize=12)
        ax3.tick_params(axis='y', labelcolor='#9467bd')
        ax3.set_ylim(0, max(max(wealth_curve), 2.0) * 1.2)
        ax3.legend(loc='lower right')
        
    plt.tight_layout()
    
    out_dir = os.path.dirname(os.path.abspath(args.result))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    plt.savefig(args.result, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved successfully to {args.result}")

if __name__ == "__main__":
    main()
