import os
import torch
import yaml
import numpy as np
import pandas as pd
import cv2

from src.registry import get_model_class
from src.utils.logging import get_logger
from src.utils.config import load_config
import src.models
from src.inference.video import load_video, sliding_window_inference, render_live_plot
from src.datasets.echonet import EchoNetVideoDataset

logger = get_logger("INFERENCE")

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_model_weights(model: torch.nn.Module, checkpoint_path: str) -> None:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not provided or not found: {checkpoint_path}. Using untrained weights.")
        return

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    unpacked_state_dict = {
        k[7:] if k.startswith("module.") else k: v 
        for k, v in state_dict.items()
    }
            
    model.load_state_dict(unpacked_state_dict, strict=False)

def setup_model(config_path: str, checkpoint_path: str, device: torch.device) -> tuple[torch.nn.Module, dict]:
    cfg = load_config(config_path)
    model_name = cfg.get("model", {}).get("name", "SpatiotemporalEchoModel")
         
    model_cls = get_model_class(model_name)
    if not model_cls:
         raise ValueError(f"Model class for {model_name} not registered.")
         
    model = model_cls.from_config(cfg)
    _load_model_weights(model, checkpoint_path)
        
    model = model.to(device)
    model.eval()
    return model, cfg

def _preprocess_video(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    video_array = np.stack(frames, axis=0).astype(np.float32) / 255.0
    video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    return video_tensor.to(device)

def _get_gt_mask(video_path: str, data_dir: str, img_size: tuple[int, int], T: int) -> np.ndarray | None:
    tracings_path = os.path.join(data_dir, "VolumeTracings.csv")
    if not os.path.exists(tracings_path):
        return None

    fname = os.path.basename(video_path)
    if fname.lower().endswith(('.avi', '.mp4')):
        fname = fname[:-4]
        
    try:
        df = pd.read_csv(tracings_path)
        df["FileName"] = df["FileName"].astype(str).apply(lambda x: x[:-4] if x.lower().endswith('.avi') else x)
        file_tracings = df[df["FileName"] == fname]
        
        if file_tracings.empty:
            logger.warning(f"No GT tracings found for {fname} in VolumeTracings.csv.")
            return None

        logger.info(f"Generating GT mask for {fname}...")
        H, W = img_size
        gt_mask = np.zeros((T, H, W), dtype=np.uint8)
        
        for t in range(T):
            t_subset = file_tracings[file_tracings["Frame"] == t]
            if not t_subset.empty:
                gt_mask[t] = EchoNetVideoDataset._generate_mask(t_subset, H, W)
                
        return gt_mask
    except Exception as e:
        logger.error(f"Error loading GT mask: {e}")
        return None

def _save_results(args, full_masks: np.ndarray, full_vol_curve: np.ndarray) -> None:
    if args.return_masks:
        masks_path = os.path.join(args.result, "masks.npy")
        vol_path = os.path.join(args.result, "volume.npy")
        np.save(masks_path, full_masks)
        np.save(vol_path, full_vol_curve)
        logger.info(f"Saved masks to {masks_path} and volume to {vol_path}")

def run_inference(args) -> tuple[np.ndarray, np.ndarray] | None:
    device = get_device()
    logger.info(f"Using device: {device}")
    
    model, cfg = setup_model(args.config, args.checkpoint, device)
    
    img_size = tuple(cfg.get('data', {}).get('img_size', [224, 224]))
    clip_len = cfg.get('model', {}).get('max_clip_len', 16)
    
    logger.info(f"Loading video: {args.video} at size {img_size}")
    frames, fps = load_video(args.video, img_size=img_size)
    if not frames:
        logger.error("No valid frames loaded from video.")
        return None
        
    T = len(frames)
    logger.info(f"Video loaded: {T} frames, {fps} fps")
    
    video_tensor = _preprocess_video(frames, device)
    
    logger.info("Running sliding window inference...")
    full_masks, full_vol_curve = sliding_window_inference(
        model, video_tensor, 
        clip_len=clip_len, 
        overlap=0,
        device=device
    )
    
    os.makedirs(args.result, exist_ok=True)
    output_video_path = os.path.join(args.result, "output.mp4")
    
    data_dir = getattr(args, "data_dir", "datasets/echonet-dynamic")
    gt_mask = _get_gt_mask(args.video, data_dir, img_size, T)
            
    logger.info(f"Rendering live video plot to {output_video_path}")
    render_live_plot(frames, full_masks, full_vol_curve, output_video_path, fps=fps, video_size=img_size, gt_mask=gt_mask)
    
    _save_results(args, full_masks, full_vol_curve)
    
    if args.return_masks:
        return full_masks, full_vol_curve
    return None
