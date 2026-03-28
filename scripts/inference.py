import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.inference.infer import run_inference

def parse_args():
    parser = argparse.ArgumentParser(description="Spatiotemporal Echo Model Inference Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration YAML file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.pt) file.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file (e.g., .avi or .mp4).")
    parser.add_argument("--result", type=str, default="results", help="Directory to save the resulting output.mp4. Default is /results")
    parser.add_argument("--data_dir", type=str, default="datasets/echonet-dynamic", help="Path to the dataset directory containing VolumeTracings.csv.")
    parser.add_argument("--return_masks", action="store_true", help="If passed, will save the segmentation masks and volume curve to disk.")
    parser.add_argument("--audit", action="store_true", help="Enable SelfAuditor to calculate Martingale Wealth.")
    parser.add_argument("--audit_stats", type=str, default=None, help="Path to audit stats JSON file for calibrated eps and max entropies.")
    
    return parser.parse_args()

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    args = parse_args()
    run_inference(args)
