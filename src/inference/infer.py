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
from src.tta.auditor import SelfAuditor

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

def _get_gt_data(video_path: str, data_dir: str, img_size: tuple[int, int], T: int) -> tuple[np.ndarray | None, int | None, int | None, float | None, float | None]:
    tracings_path = os.path.join(data_dir, "VolumeTracings.csv")
    filelist_path = os.path.join(data_dir, "FileListwFrames112.csv")
    
    gt_mask = None
    ed_frame = None
    es_frame = None
    edv = None
    esv = None

    fname = os.path.basename(video_path).lower()
    if fname.endswith('.avi') or fname.endswith('.mp4'):
        fname = fname[:-4]
        
    if os.path.exists(filelist_path):
        try:
            df_file = pd.read_csv(filelist_path)
            df_file["FileNameLower"] = df_file["FileName"].astype(str).str.lower().str.replace('.avi', '', regex=False)
            file_meta = df_file[df_file["FileNameLower"] == fname]
            if not file_meta.empty:
                ed_frame = file_meta["EDFrame"].values[0]
                es_frame = file_meta["ESFrame"].values[0]
                edv = file_meta["EDV"].values[0] if "EDV" in file_meta else None
                esv = file_meta["ESV"].values[0] if "ESV" in file_meta else None
                if not np.isnan(ed_frame): ed_frame = int(ed_frame)
                else: ed_frame = None
                if not np.isnan(es_frame): es_frame = int(es_frame)
                else: es_frame = None
                if edv is not None and np.isnan(edv): edv = None
                if esv is not None and np.isnan(esv): esv = None
        except Exception as e:
            logger.error(f"Error loading ED/ES frames from FileListwFrames112.csv: {e}")
            
    if not os.path.exists(tracings_path):
        logger.warning(f"VolumeTracings.csv not found at {tracings_path}. Cannot generate GT mask.")
        return gt_mask, ed_frame, es_frame, edv, esv

    try:
        df = pd.read_csv(tracings_path)
        df["FileNameLower"] = df["FileName"].astype(str).str.lower().str.replace('.avi', '', regex=False)
        file_tracings = df[df["FileNameLower"] == fname]
        
        if file_tracings.empty:
            logger.warning(f"No GT tracings found for {fname} in VolumeTracings.csv.")
            return gt_mask, ed_frame, es_frame, edv, esv

        logger.info(f"Generating GT mask for {fname}...")
        H, W = img_size
        gt_mask = np.zeros((T, H, W), dtype=np.uint8)
        
        for t in range(T):
            t_subset = file_tracings[file_tracings["Frame"] == t]
            if not t_subset.empty:
                gt_mask[t] = EchoNetVideoDataset._generate_mask(t_subset, H, W)
                
        return gt_mask, ed_frame, es_frame, edv, esv
    except Exception as e:
        logger.error(f"Error loading GT mask: {e}")
        return None, ed_frame, es_frame, edv, esv

def _save_results(args, full_masks: np.ndarray, full_vol_curve: np.ndarray, wealth_curve: np.ndarray | None = None) -> None:
    if args.return_masks:
        masks_path = os.path.join(args.result, "masks.npy")
        vol_path = os.path.join(args.result, "volume.npy")
        np.save(masks_path, full_masks)
        np.save(vol_path, full_vol_curve)
        if wealth_curve is not None:
            wealth_path = os.path.join(args.result, "wealth.npy")
            np.save(wealth_path, wealth_curve)
            logger.info(f"Saved masks to {masks_path}, volume to {vol_path}, and wealth to {wealth_path}")
        else:
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
    
    auditor = None
    if getattr(args, "audit", False):
        auditor = SelfAuditor(device=device)
        audit_stats = getattr(args, "audit_stats", None)
        if audit_stats:
            auditor.load_stats(audit_stats)
    
    logger.info("Running sliding window inference...")
    full_masks, full_vol_curve, clip_starts, wealth_curve = sliding_window_inference(
        model, video_tensor, 
        clip_len=clip_len, 
        overlap=0,
        device=device,
        auditor=auditor
    )
    
    os.makedirs(args.result, exist_ok=True)
    output_video_path = os.path.join(args.result, "output.mp4")
    
    data_dir = getattr(args, "data_dir", "datasets/echonet-dynamic")
    gt_mask, ed_frame, es_frame, edv, esv = _get_gt_data(args.video, data_dir, img_size, T)
            
    logger.info(f"Rendering live video plot to {output_video_path}")
    render_live_plot(frames, full_masks, full_vol_curve, output_video_path, fps=fps, video_size=img_size, gt_mask=gt_mask, ed_frame=ed_frame, es_frame=es_frame, edv=edv, esv=esv, clip_starts=clip_starts, wealth_curve=wealth_curve)
    
    _save_results(args, full_masks, full_vol_curve, wealth_curve)
    
    if args.return_masks:
        return full_masks, full_vol_curve, wealth_curve
    return None
