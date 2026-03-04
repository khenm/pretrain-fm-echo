import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from src.utils.dist import is_main_process, get_world_size, get_rank
from src.utils.logging import get_logger
from src.registry import register_dataset

logger = get_logger("ECHONET")

@register_dataset("ECHONET")
class EchoNetVideoDataset(Dataset):
    """
    Dataset class for EchoNet-Dynamic Video Classification/Regression.
    Loads video clips for R(2+1)D model.
    """
    def __init__(self, root_dir, split="TRAIN", max_clip_len=32, img_size=(112, 112), sampling_rate=1, transform=None, pretrain=False, flow_dir=None):
        self.root_dir = root_dir
        self.split = split.upper()
        self.max_clip_len = max_clip_len
        self.overlap = 0
        self.img_size = img_size
        self.sampling_rate = sampling_rate
        self.transform = transform
        self.pretrain = pretrain
        self.max_retries = 5
        self.flow_dir = flow_dir
        self.max_retries = 5
        self.videos_dir = os.path.join(root_dir, "Videos")

        self.file_list = self._load_file_list()
        
        # Meta lookup for ED/ES frames
        self.meta_lookup = self._create_meta_lookup()
        self.tracings = self._load_tracings()
        
        # Pre-calculate all valid clips
        self.clips = self._generate_clips()
        
        # Targets lookup (still by filename, but accessed via clip index)
        self.ef_targets = self._get_normalized_column("EF", 100.0)
        self.edv_targets = self._get_normalized_column("EDV", 300.0)
        self.esv_targets = self._get_normalized_column("ESV", 300.0)
        
        # Map filename to index in file_list for target retrieval
        self.fname_to_idx = {fname: i for i, fname in enumerate(self.file_list["FileName"].values)}
        
        logger.info(f"EchoNetVideoDataset initialized: Split={self.split}, Clips={len(self.clips)}, Videos={len(self.file_list)}")

    def _load_file_list(self):
        fname_w_frames = "FileListwFrames112.csv"
        path = os.path.join(self.root_dir, fname_w_frames)
        if not os.path.exists(path):
            path = os.path.join(self.root_dir, "FileList.csv")
            logger.warning(f"{fname_w_frames} not found, falling back to FileList.csv.")

        if not os.path.exists(path):
            raise FileNotFoundError(f"FileList.csv not found at {path}")

        df = pd.read_csv(path)
        if "Split" in df.columns:
            df = df[df["Split"].str.upper() == self.split]

        df["FileName"] = df["FileName"].astype(str).apply(lambda x: x[:-4] if x.lower().endswith('.avi') else x)

        if os.path.exists(self.videos_dir):
            available_files = {f[:-4] if f.lower().endswith('.avi') else f for f in os.listdir(self.videos_dir)}
            original_len = len(df)
            df = df[df["FileName"].isin(available_files)]
            if len(df) < original_len:
                logger.warning(f"Filtered {original_len - len(df)} missing videos. Remaining: {len(df)}")
        
        return df

    def _get_normalized_column(self, col_name, scale):
        if col_name in self.file_list.columns:
            return self.file_list[col_name].values / scale
        return np.full(len(self.file_list), -1.0 if "V" in col_name else 0.0)

    def _create_meta_lookup(self):
        if "EDFrame" in self.file_list.columns and "ESFrame" in self.file_list.columns:
            return self.file_list.set_index("FileName")[["EDFrame", "ESFrame"]].to_dict('index')
        logger.warning("EDFrame/ESFrame columns missing. Beat-centric sampling disabled.")
        return {}

    def _load_tracings(self):
        path = os.path.join(self.root_dir, "VolumeTracings.csv")
        if not os.path.exists(path):
            logger.warning(f"VolumeTracings.csv not found at {path}. No masks will be generated.")
            return None
            
        df = pd.read_csv(path)
        df["FileName"] = df["FileName"].astype(str).apply(lambda x: x[:-4] if x.lower().endswith('.avi') else x)
        
        # Only keep tracings for available files
        available_files = set(self.file_list["FileName"].values)
        return df[df["FileName"].isin(available_files)]

    def _generate_clips(self):
        """
        Generates a list of all valid clips using a sliding window approach.
        Returns: List of (filename, start_frame, end_frame, total_frames)
        """
        clips = []
        stride = max(1, self.max_clip_len - self.overlap)
        
        for _, row in self.file_list.iterrows():
            fname = row["FileName"]
            total_frames = int(row["NumberOfFrames"])
            
            # Generate clips: [0, 32), [16, 48), ...
            for start in range(0, total_frames, stride):
                end = start + self.max_clip_len
                
                if fname not in self.meta_lookup:
                    continue
                        
                meta = self.meta_lookup[fname]
                ed_frame = int(meta.get("EDFrame", -1))
                es_frame = int(meta.get("ESFrame", -1))
                    
                if ed_frame == -1 or es_frame == -1:
                    continue
                    
                # Check if BOTH frames are strictly within (start, end-1)
                has_ed = (start < ed_frame < end - 1)
                has_es = (start < es_frame < end - 1)
                    
                if not (has_ed and has_es):
                    continue
                        
                # Smart Padding Logic
                if end > total_frames:
                    pad_len = end - total_frames
                    if pad_len > (0.2 * self.max_clip_len):
                        continue # Drop clip if padding exceeds 20%
                        
                # Stop if we went way past (redundant with continue, but keeps logic clean)
                if start >= total_frames:
                    break
                    
                clips.append((fname, start, end, total_frames))
                
        return clips

    def __len__(self):
        return len(self.clips)

    @staticmethod
    def _generate_mask(points, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        points = points.iloc[1:] 
        scale_x = width / 112.0
        scale_y = height / 112.0
        pts = np.stack([
            np.concatenate([points["X1"].values * scale_x, points["X2"].values[::-1] * scale_x]),
            np.concatenate([points["Y1"].values * scale_y, points["Y2"].values[::-1] * scale_y])
        ], axis=1).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
        return mask

    def _pad_video_tensor(self, video_chunk, start_idx, end_idx, valid_start, valid_end):
        pad_left = max(0, valid_start - start_idx)
        pad_right = max(0, end_idx - valid_end)
        
        if pad_left > 0 or pad_right > 0:
            video_chunk = np.pad(
                video_chunk, 
                ((pad_left, pad_right), (0,0), (0,0), (0,0)), 
                mode='edge'
            )
        
        # Ensure exact length
        if video_chunk.shape[0] != self.max_clip_len:
             current_len = video_chunk.shape[0]
             if current_len < self.max_clip_len:
                 video_chunk = np.pad(video_chunk, ((0, self.max_clip_len - current_len), (0,0), (0,0), (0,0)), mode='edge')
             else:
                 video_chunk = video_chunk[:self.max_clip_len]
        return video_chunk

    def _load_video_clip(self, video_path, fname, start_idx, end_idx, total_frames):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        valid_start = max(0, start_idx)
        valid_end = min(total_frames, end_idx)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, valid_start)
        
        frames = []
        for _ in range(valid_end - valid_start):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if (frame.shape[0], frame.shape[1]) != self.img_size:
                frame = cv2.resize(frame, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        cap.release()
        
        if not frames:
            return np.zeros((self.max_clip_len, *self.img_size, 3), dtype=np.uint8)
            
        video_chunk = np.stack(frames, axis=0)
        video_chunk = self._pad_video_tensor(video_chunk, start_idx, end_idx, valid_start, valid_end)

        return video_chunk

    def __getitem__(self, idx, _retry_count=0):
        fname, start_idx, end_idx, total_frames = self.clips[idx]
        file_idx = self.fname_to_idx[fname]
        
        video_path = os.path.join(self.videos_dir, fname + (".avi" if not fname.lower().endswith(".avi") else ""))
        
        video = self._load_video_clip(video_path, fname, start_idx, end_idx, total_frames)
        
        if video is None:
            logger.warning(f"Failed to load video clip: {video_path}")
            if _retry_count >= self.max_retries:
                raise RuntimeError(f"Failed to load {self.max_retries} consecutive samples from {idx}")
            return self.__getitem__((idx+1)%len(self), _retry_count + 1)
        
        T_clip, H, W, _ = video.shape
        
        mask_clip = np.zeros((T_clip, H, W), dtype=np.uint8)
        frame_mask = np.zeros((T_clip,), dtype=np.float32)

        if fname in self.meta_lookup:
            meta = self.meta_lookup[fname]
            ed_frame = int(meta.get("EDFrame", -1))
            es_frame = int(meta.get("ESFrame", -1))
        else:
            ed_frame, es_frame = -1, -1

        for t in range(T_clip):
            original_frame_idx = start_idx + t
            if original_frame_idx == ed_frame:
                frame_mask[t] = 2.0
            elif original_frame_idx == es_frame:
                frame_mask[t] = 1.0

        if self.tracings is not None:
            file_tracings = self.tracings[self.tracings["FileName"] == fname]
            if not file_tracings.empty:
                for t in range(T_clip):
                    original_frame_idx = start_idx + t
                    if 0 <= original_frame_idx < total_frames:
                        t_subset = file_tracings[file_tracings["Frame"] == original_frame_idx]
                        if not t_subset.empty:
                            mask_clip[t] = self._generate_mask(t_subset, H, W)

        video = video.transpose(3, 0, 1, 2).astype(np.float32) / 255.0  # (C, T, H, W)
        mask_clip = mask_clip[None, ...].astype(np.float32)  # (1, T, H, W)

        video = torch.from_numpy(video)
        mask_clip = torch.from_numpy(mask_clip)

        if self.transform:
            data = self.transform({"video": video, "label": mask_clip})
            video, mask_clip = data["video"], data.get("label", mask_clip)
            
        flow_chunk = None
        if self.flow_dir and os.path.exists(self.flow_dir):
            flow_path = os.path.join(self.flow_dir, fname + ".pt")
            if os.path.exists(flow_path):
                # Shape: (TotalFrames-1, 2, H, W)
                full_flow = torch.load(flow_path, weights_only=True).float()
                
                # We need flow from start_idx to end_idx - 1
                f_start = max(0, start_idx) 
                # Note: full_flow has length total_frames - 1.
                f_end = min(total_frames - 1, end_idx) 
                
                valid_flow = full_flow[f_start:f_end]
                
                # Flow length should be T_clip - 1 ideally. Let's pad just like video if necessary
                target_flow_len = self.max_clip_len - 1
                curr_flow_len = valid_flow.shape[0]
                
                if curr_flow_len < target_flow_len:
                    pad_left = max(0, f_start - start_idx)
                    pad_right = target_flow_len - curr_flow_len - pad_left
                    valid_flow = F.pad(valid_flow, (0,0,0,0, 0,0, pad_left, pad_right), mode='constant', value=0.0)
                else:
                    valid_flow = valid_flow[:target_flow_len]
                    
                flow_chunk = valid_flow
            else:
                flow_chunk = torch.zeros((self.max_clip_len - 1, 2, *self.img_size), dtype=torch.float32)
        else:
            flow_chunk = torch.zeros((self.max_clip_len - 1, 2, *self.img_size), dtype=torch.float32)

        output = {
            "video": video,
            "label": mask_clip, 
            "case": fname,
            "frame_mask": torch.tensor(frame_mask, dtype=torch.float32),
            "lengths": torch.tensor(T_clip, dtype=torch.long),
            "target_ef": torch.tensor(self.ef_targets[file_idx], dtype=torch.float32),
            "target_edv": torch.tensor(self.edv_targets[file_idx], dtype=torch.float32),
            "target_esv": torch.tensor(self.esv_targets[file_idx], dtype=torch.float32)
        }
            
        if flow_chunk is not None:
            output["flow"] = flow_chunk

        return output

@register_dataset("ECHONET")
class EchoNet:
    @staticmethod
    def get_dataloaders(cfg):
        root_dir = cfg['data']['root_dir']
        batch_size = cfg['training'].get('batch_size', 8)
        num_workers = cfg['training'].get('num_workers', 4)
        model_name = cfg['model'].get('name')

        max_clip_len = cfg['model'].get('max_clip_len', 16)
        img_size = tuple(cfg['data'].get('img_size', [112, 112]))
        flow_dir = cfg['data'].get('flow_dir', None)
        
        pretrain = cfg['training'].get('pretrain', False)
        if pretrain:
            logger.info("Pretraining Mode: Filtering clips to contain both ED and ES frames.")

        ds_tr = EchoNetVideoDataset(root_dir, "TRAIN", max_clip_len=max_clip_len, img_size=img_size, pretrain=pretrain, flow_dir=flow_dir)
        ds_va = EchoNetVideoDataset(root_dir, "VAL", max_clip_len=max_clip_len, img_size=img_size, pretrain=pretrain, flow_dir=flow_dir)
        ds_ts = EchoNetVideoDataset(root_dir, "TEST", max_clip_len=max_clip_len, img_size=img_size, pretrain=pretrain, flow_dir=flow_dir)

        if cfg['data'].get('subset_size'):
            subset_size = int(cfg['data']['subset_size'])
            logger.info(f"Using subset of size {subset_size} for all splits")
            ds_tr = Subset(ds_tr, range(min(len(ds_tr), subset_size)))
            ds_va = Subset(ds_va, range(min(len(ds_va), subset_size)))
            ds_ts = Subset(ds_ts, range(min(len(ds_ts), subset_size)))

        # DDP Sampler
        sampler_tr = None
        sampler_va = None
        sampler_ts = None
        
        shuffle_tr = True
        shuffle_va = False
        
        if get_world_size() > 1:
            sampler_tr = DistributedSampler(ds_tr, shuffle=True)
            sampler_va = DistributedSampler(ds_va, shuffle=False)
            sampler_ts = DistributedSampler(ds_ts, shuffle=False)
            shuffle_tr = False # Sampler handles shuffling
            shuffle_va = False

        return (
            DataLoader(ds_tr, batch_size=batch_size, shuffle=shuffle_tr, num_workers=num_workers, sampler=sampler_tr, pin_memory=True),
            DataLoader(ds_va, batch_size=batch_size, shuffle=shuffle_va, num_workers=num_workers, sampler=sampler_va, pin_memory=True),
            DataLoader(ds_ts, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=sampler_ts, pin_memory=True)
        )
