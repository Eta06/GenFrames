"""
Base model class for frame interpolation
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from pathlib import Path


class BaseModel(ABC):
    """Abstract base class for frame interpolation models"""

    def __init__(
        self,
        model_name: str,
        backend: "BaseBackend",
        model_dir: Optional[Path] = None,
    ):
        """
        Initialize model

        Args:
            model_name: Name of the model (e.g., 'rife-v4.6')
            backend: Backend for computation
            model_dir: Directory to store/load model weights
        """
        self.model_name = model_name
        self.backend = backend
        self.model_dir = model_dir or Path.home() / ".cache" / "genframes" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    @abstractmethod
    def download_weights(self) -> Path:
        """
        Download model weights if not available

        Returns:
            Path to downloaded weights
        """
        pass

    @abstractmethod
    def load_model(self):
        """Load model weights into memory"""
        pass

    @abstractmethod
    def interpolate(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        timestep: float = 0.5,
    ) -> np.ndarray:
        """
        Interpolate between two frames

        Args:
            frame1: First frame (H, W, 3) in RGB format, values [0, 255]
            frame2: Second frame (H, W, 3) in RGB format, values [0, 255]
            timestep: Interpolation timestep (0.5 = middle frame)

        Returns:
            Interpolated frame (H, W, 3) in RGB format, values [0, 255]
        """
        pass

    def interpolate_recursive(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        num_frames: int,
    ) -> list[np.ndarray]:
        """
        Recursively interpolate multiple frames between two frames

        Args:
            frame1: First frame
            frame2: Second frame
            num_frames: Number of intermediate frames to generate

        Returns:
            List of interpolated frames (not including frame1 and frame2)
        """
        if num_frames == 0:
            return []
        elif num_frames == 1:
            return [self.interpolate(frame1, frame2, 0.5)]
        else:
            # Recursively interpolate
            mid_frame = self.interpolate(frame1, frame2, 0.5)
            left_frames = self.interpolate_recursive(frame1, mid_frame, num_frames // 2)
            right_frames = self.interpolate_recursive(
                mid_frame, frame2, num_frames - num_frames // 2 - 1
            )
            return left_frames + [mid_frame] + right_frames

    def ensure_model_loaded(self):
        """Ensure model is loaded (download if necessary)"""
        if self._model is None:
            weights_path = self.model_dir / f"{self.model_name}.pth"
            if not weights_path.exists():
                print(f"Downloading {self.model_name} weights...")
                weights_path = self.download_weights()
            print(f"Loading {self.model_name} on {self.backend.get_device_name()}...")
            self.load_model()

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for model input

        Args:
            frame: Frame in RGB format, uint8 [0, 255]

        Returns:
            Preprocessed frame, float32 [0, 1]
        """
        return frame.astype(np.float32) / 255.0

    def postprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Postprocess model output to frame

        Args:
            frame: Model output, float32 [0, 1]

        Returns:
            Frame in RGB format, uint8 [0, 255]
        """
        frame = np.clip(frame * 255.0, 0, 255)
        return frame.astype(np.uint8)
