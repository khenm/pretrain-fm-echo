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

def sliding_window_inference(model, video_tensor, clip_len=16, overlap=0, device="cuda"):
    """
    Runs sliding window inference on the full video tensor.
    
    Args:
        model: PyTorch model.
        video_tensor: (1, 3, T, H, W) tensor on device.
        clip_len: length of each temporal clip.
        overlap: temporal overlap (not yet implemented fully, using non-overlapping or simple chunking).
        device: typical device to run.
        
    Returns:
        full_masks: (T, H, W) numpy array of mask probabilities or binary masks.
        full_vol_curve: (T,) numpy array of volume estimates.
    """
    model.eval()
    
    B, C, T, H, W = video_tensor.shape
    stride = max(1, clip_len - overlap)
    
    # Store results
    full_mask_logits = torch.zeros((T, H, W), device=device) # (T, H, W)
    full_vol_curve = torch.zeros((T,), device=device) # (T,)
    counts = torch.zeros((T,), device=device)
    
    with torch.no_grad():
        for start_idx in range(0, T, stride):
            end_idx = min(start_idx + clip_len, T)
            
            # If the chunk is smaller than clip_len, we might need to pad it to fit the model's expected clip_len.
            chunk = video_tensor[:, :, start_idx:end_idx, :, :]
            actual_len = chunk.shape[2]
            
            pad_len = 0
            if actual_len < clip_len:
                pad_len = clip_len - actual_len
                # Pad repeating the last frame or zero padding
                # Depending on how the model was trained, zero-padding might be safer, or edge padding.
                # using replication of last frame:
                pad_tensor = chunk[:, :, -1:].expand(-1, -1, pad_len, -1, -1)
                chunk = torch.cat([chunk, pad_tensor], dim=2)
                
            outputs = model(chunk)
            
            mask_logits = outputs["mask_logits"] # (1, 1, T_clip, H, W)
            vol_curve = outputs["vol_curve"]     # (1, T_clip)
            
            mask_logits_req = mask_logits # (1, 1, T_clip, H', W')
            if mask_logits_req.shape[2:] != (clip_len, H, W):
                mask_logits = F.interpolate(mask_logits_req, size=(clip_len, H, W), mode='trilinear', align_corners=False)
            else:
                mask_logits = mask_logits_req
            mask_logits = mask_logits.squeeze(0).squeeze(0) # (clip_len, H, W)
            
            vol_curve = vol_curve.unsqueeze(0) # (1, 1, T_clip)
            if vol_curve.shape[2] != clip_len:
                vol_curve = F.interpolate(vol_curve, size=clip_len, mode='linear', align_corners=False)
            vol_curve = vol_curve.squeeze(0).squeeze(0) # (clip_len,)
            
            # Aggregate
            valid_len = actual_len
            full_mask_logits[start_idx:end_idx] += mask_logits[:valid_len]
            full_vol_curve[start_idx:end_idx] += vol_curve[:valid_len]
            counts[start_idx:end_idx] += 1

    # Average overlaps
    full_mask_logits = full_mask_logits / counts.unsqueeze(-1).unsqueeze(-1)
    full_vol_curve = full_vol_curve / counts
    
    full_masks = torch.sigmoid(full_mask_logits) > 0.5
    full_masks = full_masks.cpu().numpy().astype(np.uint8)
    
    return full_masks, full_vol_curve.cpu().numpy()

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.4):
    """
    Overlays a binary mask on an RGB image.
    """
    overlay = image.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(mask > 0, image[:, :, c] * (1 - alpha) + color[c] * alpha, image[:, :, c])
    return overlay.astype(np.uint8)

def render_live_plot(frames, masks, vol_curve, output_path, fps=30.0, video_size=(224, 224)):
    """
    Generates a live video plot with segmentation masks and a volume curve.
    
    Args:
        frames: list of numpy arrays (H, W, 3) representing the original video frames.
        masks: (T, H, W) numpy array of binary masks.
        vol_curve: (T,) numpy array of volume estimates.
        output_path: path to save the output video (.mp4).
        fps: output frames per second.
        video_size: spatial size of the video.
    """
    T = len(frames)
    if T == 0:
        return
        
    H, W = frames[0].shape[:2]
    
    # Identify ED (End-Diastole, max volume) and ES (End-Systole, min volume)
    ed_frame = np.argmax(vol_curve)
    es_frame = np.argmin(vol_curve)
    
    # Plot configuration
    plot_h, plot_w = 200, 448 # W*2 typically for side-by-side or matched width
    out_w = W * 2 # Original | Overlaid
    out_h = H + plot_h
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    
    for t in range(T):
        frame = frames[t]
        mask = masks[t]
        
        # Side-by-side: original vs mask overlaid
        overlaid_frame = overlay_mask(frame, mask, color=(0, 255, 0), alpha=0.5)
        top_row = np.concatenate([frame, overlaid_frame], axis=1) # (H, W*2, 3)
        
        # Bottom row: Volume curve
        fig, ax = plt.subplots(figsize=(out_w / 100, plot_h / 100), dpi=100)
        ax.plot(range(T), vol_curve, color='blue', label='Volume Curve', alpha=0.5)
        ax.plot(range(t+1), vol_curve[:t+1], color='red', linewidth=2)
        ax.scatter([t], [vol_curve[t]], color='red', s=50, zorder=5)
        
        # Annotate ED and ES frames
        ax.axvline(x=ed_frame, color='green', linestyle='--', label=f'ED (Frame {ed_frame})')
        ax.axvline(x=es_frame, color='purple', linestyle='--', label=f'ES (Frame {es_frame})')
        ax.legend(loc='upper right', fontsize='small')
        
        ax.set_xlim(0, max(T-1, 1))
        
        min_vol, max_vol = np.min(vol_curve), np.max(vol_curve)
        margin = (max_vol - min_vol) * 0.1
        if margin == 0: margin = 10
        ax.set_ylim(min_vol - margin, max_vol + margin)
        
        ax.set_title(f"Volume: {vol_curve[t]:.2f}")
        ax.set_ylabel("Volume")
        ax.set_xlabel("Frame")
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Render plot to numpy array
        canvas = FigureCanvas(fig)
        canvas.draw()
        plot_img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plot_img = plot_img[:, :, :3] # keep RGB
        plt.close(fig)
        
        # resize plot_img to exactly (plot_h, out_w) if not already
        if plot_img.shape[:2] != (plot_h, out_w):
            plot_img = cv2.resize(plot_img, (out_w, plot_h))
            
        # Combine
        final_frame = np.concatenate([top_row, plot_img], axis=0) # (H+plot_h, W*2, 3)
        
        # OpenCV uses BGR for writing
        final_frame_bgr = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
        out_video.write(final_frame_bgr)
        
    out_video.release()
    print(f"Saved visualization to {output_path}")
