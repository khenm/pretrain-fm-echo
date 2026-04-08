import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def load_video(video_path, img_size=(224, 224)):
    """
    Loads a video and resizes it to the required img_size.
    Returns:
        frames: list of numpy arrays (H, W, 3) in RGB
        fps: float
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps):
        fps = 30.0
        
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
    return frames, fps

def _compute_vol_curve(mask_logits: torch.Tensor) -> torch.Tensor:
    import math
    masks_for_vol = (torch.sigmoid(mask_logits) > 0.5).squeeze(0).squeeze(0)
    masks_np = masks_for_vol.cpu().numpy().astype(np.uint8)
    
    vol_curve_list = []
    for mask_idx in range(masks_np.shape[0]):
        area = float(masks_np[mask_idx].sum())
        if area == 0:
            vol_curve_list.append(0.0)
            continue
            
        contours, _ = cv2.findContours(masks_np[mask_idx], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            length = 1.0
        else:
            contour = max(contours, key=cv2.contourArea)
            if len(contour) >= 5:
                _, (major_axis, minor_axis), _ = cv2.fitEllipse(contour)
                length = max(major_axis, minor_axis)
            else:
                rect = cv2.minAreaRect(contour)
                length = max(rect[1][0], rect[1][1])
            length = max(length, 1e-3)
            
        volume = (8.0 * (area ** 2)) / (3.0 * math.pi * length)
        vol_curve_list.append(volume)
        
    return torch.tensor(vol_curve_list, device=mask_logits.device, dtype=torch.float32).unsqueeze(0)

def sliding_window_inference(model, video_tensor, clip_len=16, overlap=0, batch_size=1, device="cuda", auditor=None):
    """
    Runs sliding window inference on the full video tensor.
    
    Args:
        model: PyTorch model.
        video_tensor: (1, 3, T, H, W) tensor on device.
        clip_len: length of each temporal clip.
        overlap: temporal overlap (not yet implemented fully, using non-overlapping or simple chunking).
        batch_size: number of clips to process in a single forward pass.
        device: typical device to run.
        auditor: Optional SelfAuditor instance for OOD detection via Martingale Wealth.
        
    Returns:
        full_masks: (T, H, W) numpy array of mask probabilities or binary masks.
        full_vol_curve: (T,) numpy array of volume estimates.
        clip_starts: list of integers.
        wealth_curve: (T,) numpy array of Martingale Wealth values (or None).
    """
    model.eval()
    
    _, _, T, H, W = video_tensor.shape
    stride = max(1, clip_len - overlap)
    
    full_mask_logits = torch.zeros((T, H, W), device=device)
    full_vol_curve = torch.zeros((T,), device=device)
    counts = torch.zeros((T,), device=device)
    
    clip_starts = []
    
    with torch.no_grad():
        pending_chunks = []
        pending_indices = []
        
        for start_idx in range(0, T, stride):
            clip_starts.append(start_idx)
            end_idx = min(start_idx + clip_len, T)
            
            chunk = video_tensor[:, :, start_idx:end_idx, :, :]
            actual_len = chunk.shape[2]
            
            if actual_len < clip_len:
                pad_len = clip_len - actual_len
                pad_tensor = chunk[:, :, -1:].expand(-1, -1, pad_len, -1, -1)
                chunk = torch.cat([chunk, pad_tensor], dim=2)
            
            pending_chunks.append(chunk)
            pending_indices.append((start_idx, end_idx, actual_len))
            
            if len(pending_chunks) >= batch_size or (start_idx + stride >= T):
                if not pending_chunks:
                    continue
                    
                batch_input = torch.cat(pending_chunks, dim=0) # (B, 3, clip_len, H, W)
                outputs = model(batch_input)
                
                mask_logits_batch = outputs["mask_logits"] # (B, 1, clip_len, H, W)
                
                vol_curve_batch = None
                if "volume" in outputs and outputs["volume"] is not None:
                    vol_curve_batch = outputs["volume"]
                    if vol_curve_batch.dim() == 3:
                        vol_curve_batch = vol_curve_batch.squeeze(-1) # (B, clip_len)
                
                for i, (s, e, a_len) in enumerate(pending_indices):
                    mask_logits = mask_logits_batch[i:i+1] # (1, 1, clip_len, H, W)
                    
                    if vol_curve_batch is not None:
                        vol_curve = vol_curve_batch[i:i+1] # (1, clip_len)
                    else:
                        vol_curve = _compute_vol_curve(mask_logits) # (1, clip_len)
                    
                    if mask_logits.shape[2:] != (clip_len, H, W):
                        mask_logits = F.interpolate(mask_logits, size=(clip_len, H, W), mode='trilinear', align_corners=False)
                    mask_logits = mask_logits.squeeze(0).squeeze(0) # (clip_len, H, W)
                    
                    vol_curve = vol_curve.unsqueeze(0) # (1, 1, clip_len)
                    if vol_curve.shape[2] != clip_len:
                        vol_curve = F.interpolate(vol_curve, size=clip_len, mode='linear', align_corners=False)
                    vol_curve = vol_curve.squeeze(0).squeeze(0) # (clip_len,)
                    
                    full_mask_logits[s:e] += mask_logits[:a_len]
                    full_vol_curve[s:e] += vol_curve[:a_len]
                    counts[s:e] += 1
                
                pending_chunks = []
                pending_indices = []

    full_mask_logits = full_mask_logits / counts.unsqueeze(-1).unsqueeze(-1)
    full_vol_curve = full_vol_curve / counts
    
    full_masks = (torch.sigmoid(full_mask_logits) > 0.5).cpu().numpy().astype(np.uint8)
    
    wealth_curve = None
    if auditor is not None:
        wealth_history = []
        for t in range(T):
            wealth = auditor.update(full_mask_logits[t].unsqueeze(0), is_logits=True)
            wealth_history.append(wealth)
        wealth_curve = np.array(wealth_history)
        
    return full_masks, full_vol_curve.cpu().numpy(), clip_starts, wealth_curve

def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (0, 255, 0), alpha: float = 0.4) -> np.ndarray:
    """Overlays a binary mask on an RGB image."""
    overlay = image.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(mask > 0, image[:, :, c] * (1 - alpha) + color[c] * alpha, image[:, :, c])
    return overlay.astype(np.uint8)

def render_live_plot(frames: list[np.ndarray], masks: np.ndarray, vol_curve: np.ndarray, output_path: str, fps: float = 30.0, video_size: tuple[int, int] = (224, 224), gt_mask: np.ndarray | None = None, ed_frame: int | None = None, es_frame: int | None = None, edv: float | None = None, esv: float | None = None, clip_starts: list[int] | None = None, wealth_curve: np.ndarray | None = None) -> None:
    T = len(frames)
    if T == 0:
        return
        
    H, W = frames[0].shape[:2]
    
    if ed_frame is None or ed_frame < 0 or ed_frame >= T:
        ed_frame = np.argmax(vol_curve)
    if es_frame is None or es_frame < 0 or es_frame >= T:
        es_frame = np.argmin(vol_curve)
    
    plot_h, plot_w = 200, 448 
    out_w = W * 2 
    out_h = H + plot_h
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    
    for t in range(T):
        frame = frames[t]
        mask = masks[t]
        
        left_frame = frame.copy()
        if gt_mask is not None and t < len(gt_mask):
            gt_frame_mask = gt_mask[t]
            if np.any(gt_frame_mask):
                left_frame = overlay_mask(left_frame, gt_frame_mask, color=(255, 0, 0), alpha=0.5)
                
        overlaid_frame = overlay_mask(frame, mask, color=(0, 255, 0), alpha=0.5)
        top_row = np.concatenate([left_frame, overlaid_frame], axis=1)
        
        fig, ax = plt.subplots(figsize=(out_w / 100, plot_h / 100), dpi=100)
        ax.plot(range(T), vol_curve, color='blue', label='Volume Curve', alpha=0.5)
        ax.plot(range(t+1), vol_curve[:t+1], color='red', linewidth=2)
        ax.scatter([t], [vol_curve[t]], color='red', s=50, zorder=5)
        
        if clip_starts:
            for start in clip_starts:
                ax.axvline(x=start, color='lightgray', linestyle=':', zorder=1)
                
        if wealth_curve is not None:
            ax2 = ax.twinx()
            ax2.plot(range(T), wealth_curve, color='#9467bd', label='Wealth', alpha=0.5, linestyle=':')
            ax2.plot(range(t+1), wealth_curve[:t+1], color='#9467bd', linewidth=2, linestyle=':')
            ax2.scatter([t], [wealth_curve[t]], color='#9467bd', s=50, zorder=5)
            ax2.set_ylabel('Wealth (OOD)', color='#9467bd')
            ax2.tick_params(axis='y', labelcolor='#9467bd')
            ax2.set_ylim(0, max(max(wealth_curve), 2.0) * 1.2)
                
        ax.axvline(x=ed_frame, color='green', linestyle='--', label=f'ED (Frame {ed_frame})')
        ax.axvline(x=es_frame, color='purple', linestyle='--', label=f'ES (Frame {es_frame})')
        
        if ed_frame is not None and edv is not None:
            ax.scatter([ed_frame], [edv / 300.0], color='green', marker='o', s=80, zorder=6)
        if es_frame is not None and esv is not None:
            ax.scatter([es_frame], [esv / 300.0], color='purple', marker='o', s=80, zorder=6)
        
        ax.set_xlim(0, max(T-1, 1))
        
        vols_to_plot = list(vol_curve)
        if ed_frame is not None and edv is not None: vols_to_plot.append(edv / 300.0)
        if es_frame is not None and esv is not None: vols_to_plot.append(esv / 300.0)
        min_vol, max_vol = np.min(vols_to_plot), np.max(vols_to_plot)
        margin = max((max_vol - min_vol) * 0.1, 1e-3)
        ax.set_ylim(min_vol - margin, max_vol + margin)
        
        ax.set_title(f"Volume: {vol_curve[t]:.2f}")
        ax.set_ylabel("Volume")
        ax.set_xlabel("Frame")
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        canvas = FigureCanvas(fig)
        canvas.draw()
        plot_img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plot_img = plot_img[:, :, :3]
        plt.close(fig)
        
        if plot_img.shape[:2] != (plot_h, out_w):
            plot_img = cv2.resize(plot_img, (out_w, plot_h))
            
        final_frame = np.concatenate([top_row, plot_img], axis=0) 
        
        final_frame_bgr = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
        out_video.write(final_frame_bgr)
        
    out_video.release()
    print(f"Saved visualization to {output_path}")
