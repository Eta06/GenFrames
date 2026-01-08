"""
MLX backend for Apple Silicon
"""

from typing import Any, Optional
import numpy as np
from genframes.backends.base import BaseBackend


class MLXBackend(BaseBackend):
    """MLX backend optimized for Apple Silicon"""

    def __init__(self, device: Optional[str] = None):
        """
        Initialize MLX backend

        Args:
            device: Device to use (MLX manages GPU/CPU automatically)
        """
        super().__init__(device)
        self.mx = None

    def is_available(self) -> bool:
        """Check if MLX is available (Apple Silicon only)"""
        try:
            import mlx.core as mx
            import mlx.nn as nn
            self.mx = mx
            self.nn = nn
            # Test if MLX is working
            _ = mx.array([1.0])
            return True
        except (ImportError, Exception):
            return False

    def get_device_name(self) -> str:
        """Get human-readable device name"""
        if not self.is_available():
            return "MLX (unavailable)"

        import platform
        chip = platform.processor()
        return f"MLX (Apple Silicon - {chip})"

    def to_tensor(self, array: np.ndarray) -> Any:
        """
        Convert numpy array to MLX array

        Args:
            array: Numpy array

        Returns:
            MLX array
        """
        return self.mx.array(array)

    def to_numpy(self, tensor: Any) -> np.ndarray:
        """
        Convert MLX array to numpy array

        Args:
            tensor: MLX array

        Returns:
            Numpy array
        """
        return np.array(tensor)

    def load_model(self, model_path: str, model_class: Any) -> Any:
        """
        Load MLX model from checkpoint

        Args:
            model_path: Path to model checkpoint
            model_class: Model class to instantiate

        Returns:
            Loaded model
        """
        import pickle

        model = model_class()

        # MLX models typically use their own format or converted PyTorch weights
        # For now, we'll implement a converter from PyTorch to MLX
        try:
            # Try to load as pickle (MLX native format)
            with open(model_path, "rb") as f:
                weights = pickle.load(f)
            model.load_weights(weights)
        except Exception:
            # Try to load PyTorch weights and convert
            import torch
            state_dict = torch.load(model_path, map_location="cpu")

            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

            # Convert PyTorch tensors to MLX arrays
            mlx_weights = {}
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                    mlx_weights[key] = self.mx.array(value.cpu().numpy())
                else:
                    mlx_weights[key] = value

            model.load_weights(mlx_weights)

        return model

    def initialize(self):
        """Initialize backend"""
        super().initialize()
        if not self.is_available():
            raise RuntimeError(
                "MLX is not available. Please install mlx on an Apple Silicon device."
            )
