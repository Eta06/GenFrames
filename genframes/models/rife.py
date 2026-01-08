"""
RIFE (Real-Time Intermediate Flow Estimation) model implementation
Based on: https://github.com/hzwer/ECCV2022-RIFE
"""

from typing import Optional
import numpy as np
from pathlib import Path
import requests
from tqdm import tqdm
from genframes.models.base import BaseModel


class RIFEModel(BaseModel):
    """RIFE model for frame interpolation"""

    # Official model URLs
    MODEL_URLS = {
        "rife-v4.6": "https://github.com/hzwer/Practical-RIFE/releases/download/4.6/flownet.pkl",
        "rife-v4.15-lite": "https://github.com/hzwer/Practical-RIFE/releases/download/4.15.lite/flownet.pkl",
    }

    def __init__(
        self,
        backend: "BaseBackend",
        model_version: str = "rife-v4.6",
        model_dir: Optional[Path] = None,
        scale: float = 1.0,
    ):
        """
        Initialize RIFE model

        Args:
            backend: Computation backend
            model_version: Model version to use
            model_dir: Directory to store model weights
            scale: Scale factor for input resolution (1.0 = original resolution)
        """
        super().__init__(model_version, backend, model_dir)
        self.scale = scale
        self.version = model_version

        if model_version not in self.MODEL_URLS:
            raise ValueError(
                f"Unknown model version: {model_version}. "
                f"Available: {list(self.MODEL_URLS.keys())}"
            )

    def download_weights(self) -> Path:
        """
        Download RIFE model weights

        Returns:
            Path to downloaded weights
        """
        url = self.MODEL_URLS[self.version]
        weights_path = self.model_dir / f"{self.version}.pkl"

        if weights_path.exists():
            return weights_path

        print(f"Downloading {self.version} from {url}...")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(weights_path, "wb") as f, tqdm(
            desc=self.version,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)

        print(f"Downloaded to {weights_path}")
        return weights_path

    def load_model(self):
        """Load RIFE model"""
        weights_path = self.model_dir / f"{self.version}.pkl"

        if not weights_path.exists():
            weights_path = self.download_weights()

        # Import PyTorch for model architecture
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("PyTorch is required for RIFE model. Install with: pip install torch")

        # Load the IFNet architecture
        self._model = IFNet()

        # Load weights
        checkpoint = torch.load(weights_path, map_location="cpu")
        self._model.load_state_dict(checkpoint, strict=False)

        # Move to device
        if hasattr(self.backend, "_get_device"):
            device = self.backend._get_device()
            self._model = self._model.to(device)

        self._model.eval()

    def interpolate(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        timestep: float = 0.5,
    ) -> np.ndarray:
        """
        Interpolate between two frames using RIFE

        Args:
            frame1: First frame (H, W, 3) RGB [0, 255]
            frame2: Second frame (H, W, 3) RGB [0, 255]
            timestep: Interpolation timestep (0.5 = middle)

        Returns:
            Interpolated frame (H, W, 3) RGB [0, 255]
        """
        self.ensure_model_loaded()

        import torch
        import torch.nn.functional as F

        # Preprocess
        img0 = self.preprocess(frame1)
        img1 = self.preprocess(frame2)

        # Convert to tensor (B, C, H, W)
        img0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).float()
        img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()

        # Move to device
        if hasattr(self.backend, "_get_device"):
            device = self.backend._get_device()
            img0 = img0.to(device)
            img1 = img1.to(device)

        # Pad to multiple of 32
        h, w = img0.shape[2:]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h != 0 or pad_w != 0:
            img0 = F.pad(img0, (0, pad_w, 0, pad_h), mode="reflect")
            img1 = F.pad(img1, (0, pad_w, 0, pad_h), mode="reflect")

        # Inference
        with torch.no_grad():
            output = self._model(img0, img1, timestep=timestep)

        # Remove padding
        if pad_h != 0 or pad_w != 0:
            output = output[:, :, :h, :w]

        # Convert back to numpy
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Postprocess
        return self.postprocess(output)


# RIFE IFNet architecture
class IFNet(torch.nn.Module):
    """
    IFNet architecture for RIFE v4.x
    Simplified version - actual implementation should match official RIFE
    """

    def __init__(self):
        super().__init__()
        import torch.nn as nn

        # Feature pyramid network
        self.encoder = nn.ModuleList([
            ConvBlock(3, 32, 3, 2, 1),
            ConvBlock(32, 64, 3, 2, 1),
            ConvBlock(64, 128, 3, 2, 1),
            ConvBlock(128, 256, 3, 2, 1),
        ])

        # Flow estimation
        self.decoder = nn.ModuleList([
            ConvBlock(256, 128, 3, 1, 1),
            ConvBlock(128, 64, 3, 1, 1),
            ConvBlock(64, 32, 3, 1, 1),
            ConvBlock(32, 16, 3, 1, 1),
        ])

        # Final flow prediction
        self.flow_head = nn.Conv2d(16, 4, 3, 1, 1)  # 4 channels for bidirectional flow

    def forward(self, img0, img1, timestep=0.5):
        """
        Forward pass

        Args:
            img0: First frame (B, 3, H, W)
            img1: Second frame (B, 3, H, W)
            timestep: Interpolation timestep

        Returns:
            Interpolated frame (B, 3, H, W)
        """
        import torch.nn.functional as F

        # Concatenate frames
        x = torch.cat([img0, img1], dim=1)

        # Encode
        features = []
        feat = torch.cat([img0, img1], dim=1)
        for layer in self.encoder:
            feat = layer(feat)
            features.append(feat)

        # Decode and estimate flow
        x = features[-1]
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            if i < len(features) - 1:
                x = x + F.interpolate(features[-(i + 2)], size=x.shape[2:], mode="bilinear")

        # Predict flow
        flow = self.flow_head(x)
        flow = F.interpolate(flow, size=img0.shape[2:], mode="bilinear", align_corners=False)

        # Warp frames using flow
        flow0 = flow[:, :2] * timestep
        flow1 = flow[:, 2:] * (1 - timestep)

        warped0 = self.warp(img0, flow0)
        warped1 = self.warp(img1, flow1)

        # Blend warped frames
        output = warped0 * (1 - timestep) + warped1 * timestep

        return output

    def warp(self, img, flow):
        """Warp image using optical flow"""
        import torch
        import torch.nn.functional as F

        B, C, H, W = img.shape
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if img.is_cuda:
            grid = grid.cuda()

        vgrid = grid + flow
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)

        output = F.grid_sample(img, vgrid, align_corners=True)
        return output


class ConvBlock(torch.nn.Module):
    """Convolutional block with activation"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        import torch.nn as nn

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        return self.prelu(self.conv(x))
