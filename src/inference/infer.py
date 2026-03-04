import os
import torch
import yaml
from src.registry import get_model_class
from src.utils.logging import get_logger
import src.models  # Import to trigger model registrations

logger = get_logger("INFERENCE")

from src.utils.config import load_config

def setup_model(config_path, checkpoint_path, device="cuda"):
    """
    Sets up the model from configuration and checkpoint.
    """
    cfg = load_config(config_path)
        
    model_name = cfg.get("model", {}).get("name")
    if not model_name:
         # fallback or default
         model_name = "SpatiotemporalEchoModel"
         logger.info(f"Model name not found in config, using default: {model_name}")
         
    model_cls = get_model_class(model_name)
    if not model_cls:
         raise ValueError(f"Model class for {model_name} not registered.")
         
    model_cfg = cfg.get('model', {})
    fm_configs = model_cfg.get('foundation_models', {})
    fusion_space_cfg = model_cfg.get('fusion_space', {})
    peft_cfg = model_cfg.get('peft')

    model = model_cls.from_config(cfg)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        # Unpack DDP
        unpacked_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                unpacked_state_dict[k[7:]] = v
            else:
                unpacked_state_dict[k] = v
                
        model.load_state_dict(unpacked_state_dict, strict=False)
    else:
        logger.warning(f"Checkpoint not provided or not found: {checkpoint_path}. Using untrained weights.")
        
    model = model.to(device)
    model.eval()
    return model, cfg

def run_inference(args):
    """
    Main entry point for running inference on a single video.
    """
    import numpy as np
    from src.inference.video import load_video, sliding_window_inference, render_live_plot
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 1. Setup Model
    model, cfg = setup_model(args.config, args.checkpoint, device)
    
    # 2. Load Video
    # Use image size from config, default 224
    img_size = tuple(cfg.get('data', {}).get('img_size', [224, 224]))
    clip_len = cfg.get('model', {}).get('max_clip_len', 16)
    
    logger.info(f"Loading video: {args.video} at size {img_size}")
    frames, fps = load_video(args.video, img_size=img_size)
    if not frames:
        logger.error("No valid frames loaded from video.")
        return
        
    T = len(frames)
    logger.info(f"Video loaded: {T} frames, {fps} fps")
    
    # Preprocess video: float32 normalization (0-1), correct axes
    # (T, H, W, C) -> (C, T, H, W) -> (1, C, T, H, W)
    video_array = np.stack(frames, axis=0).astype(np.float32) / 255.0
    video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    video_tensor = video_tensor.to(device)
    
    # 3. Sliding Window Inference
    logger.info("Running sliding window inference...")
    full_masks, full_vol_curve = sliding_window_inference(
        model, video_tensor, 
        clip_len=clip_len, 
        overlap=0, # can adjust if you implement overlapping
        device=device
    )
    
    # 4. Result preparation
    import os
    os.makedirs(args.result, exist_ok=True)
    
    output_video_path = os.path.join(args.result, "output.mp4")
    
    # 5. Render Plot
    logger.info(f"Rendering live video plot to {output_video_path}")
    render_live_plot(frames, full_masks, full_vol_curve, output_video_path, fps=fps, video_size=img_size)
    
    # 6. Optional Returns
    if args.return_masks:
        masks_path = os.path.join(args.result, "masks.npy")
        vol_path = os.path.join(args.result, "volume.npy")
        np.save(masks_path, full_masks)
        np.save(vol_path, full_vol_curve)
        logger.info(f"Saved masks to {masks_path} and volume to {vol_path}")
        return full_masks, full_vol_curve

