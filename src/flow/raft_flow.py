import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import torch.nn.functional as F

class RAFTFlowEstimator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=self.weights, progress=False).to(self.device)
        self.model.eval()
        self.transforms = self.weights.transforms()

    @torch.no_grad()
    def compute_dense_flow(self, video_tensor: torch.Tensor, batch_size: int = 8) -> torch.Tensor:
        """
        Computes flow between consecutive frames in a video tensor.
        Args:
            video_tensor: Shape (T, C, H, W). Values should be [0, 1] or [0, 255].
                          Expects RGB channels.
            batch_size: To avoid OOM, how many frame-pairs to process at once.
        Returns:
            flow_tensor: Shape (T-1, 2, H, W)
        """
        T, C, H, W = video_tensor.shape
        if T < 2:
            return torch.zeros((0, 2, H, W), device=video_tensor.device)
        
        # Ensure tensor is scaled [0, 1] if it looks like [0, 255]
        if video_tensor.max() > 1.0:
            video_tensor = video_tensor.float() / 255.0
            
        video_tensor = video_tensor.to(self.device)
        
        # Prepare pairs: (frame_t, frame_{t+1})
        img1_batch = video_tensor[:-1] # [0, 1, ..., T-2]
        img2_batch = video_tensor[1:]  # [1, 2, ..., T-1]
        
        img1_batch, img2_batch = self.transforms(img1_batch, img2_batch)

        pad_h = max(128 - H, 0)
        pad_w = max(128 - W, 0)
        pad_h = pad_h + (8 - (H + pad_h) % 8) % 8
        pad_w = pad_w + (8 - (W + pad_w) % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            img1_batch = F.pad(img1_batch, (0, pad_w, 0, pad_h), mode='replicate')
            img2_batch = F.pad(img2_batch, (0, pad_w, 0, pad_h), mode='replicate')
        
        flows = []
        num_pairs = len(img1_batch)
        
        for i in range(0, num_pairs, batch_size):
            b_img1 = img1_batch[i:i + batch_size]
            b_img2 = img2_batch[i:i + batch_size]
            
            # RAFT returns a list of flow estimates per iteration.
            # We take the final one (index -1)
            flow_predictions = self.model(b_img1, b_img2)[-1]
            flows.append(flow_predictions.cpu()) # Move to CPU immediately to save VRAM
            
        flow_tensor = torch.cat(flows, dim=0) # (T-1, 2, H', W')
        
        # Crop away the padding we added
        if pad_h > 0 or pad_w > 0:
            flow_tensor = flow_tensor[..., :flow_tensor.shape[-2]-pad_h, :flow_tensor.shape[-1]-pad_w]
            
        if flow_tensor.shape[-2:] != (H, W):
            curr_h, curr_w = flow_tensor.shape[-2:]
            
            # Resize
            flow_tensor = F.interpolate(flow_tensor, size=(H, W), mode='bilinear', align_corners=False)
            
            # Scale magnitudes
            scale_x = W / curr_w
            scale_y = H / curr_h
            
            flow_tensor[:, 0, :, :] *= scale_x
            flow_tensor[:, 1, :, :] *= scale_y
            
        return flow_tensor
