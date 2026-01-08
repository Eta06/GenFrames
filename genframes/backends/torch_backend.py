"""
PyTorch backend for CUDA, MPS, and CPU
"""

from typing import Any, Optional
import numpy as np
from genframes.backends.base import BaseBackend


class TorchBackend(BaseBackend):
    """PyTorch backend supporting CUDA, MPS, and CPU"""

    def __init__(self, device: Optional[str] = None):
        """
        Initialize PyTorch backend

        Args:
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
        """
        super().__init__(device)
        self.torch = None
        self._device = None

    def is_available(self) -> bool:
        """Check if PyTorch is available"""
        try:
            import torch
            self.torch = torch
            return True
        except ImportError:
            return False

    def get_device_name(self) -> str:
        """Get human-readable device name"""
        if not self.is_available():
            return "PyTorch (unavailable)"

        device = self._get_device()
        if device.type == "cuda":
            return f"PyTorch CUDA ({self.torch.cuda.get_device_name(0)})"
        elif device.type == "mps":
            return "PyTorch MPS (Apple Silicon)"
        else:
            return "PyTorch CPU"

    def _get_device(self):
        """Get torch device object"""
        if self._device is not None:
            return self._device

        if self.device is None:
            # Auto-detect best device
            if self.torch.cuda.is_available():
                self._device = self.torch.device("cuda")
            elif hasattr(self.torch.backends, "mps") and self.torch.backends.mps.is_available():
                self._device = self.torch.device("mps")
            else:
                self._device = self.torch.device("cpu")
        else:
            self._device = self.torch.device(self.device)

        return self._device

    def to_tensor(self, array: np.ndarray) -> Any:
        """
        Convert numpy array to PyTorch tensor

        Args:
            array: Numpy array

        Returns:
            PyTorch tensor on the appropriate device
        """
        tensor = self.torch.from_numpy(array)
        return tensor.to(self._get_device())

    def to_numpy(self, tensor: Any) -> np.ndarray:
        """
        Convert PyTorch tensor to numpy array

        Args:
            tensor: PyTorch tensor

        Returns:
            Numpy array
        """
        return tensor.detach().cpu().numpy()

    def load_model(self, model_path: str, model_class: Any) -> Any:
        """
        Load PyTorch model from checkpoint

        Args:
            model_path: Path to model checkpoint
            model_class: Model class to instantiate

        Returns:
            Loaded model
        """
        device = self._get_device()
        model = model_class()

        # Load state dict
        state_dict = self.torch.load(model_path, map_location=device)

        # Handle different checkpoint formats
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        return model

    def initialize(self):
        """Initialize backend"""
        super().initialize()
        if not self.is_available():
            raise RuntimeError("PyTorch is not available. Please install torch.")

        # Set optimization flags
        if self._get_device().type == "cuda":
            self.torch.backends.cudnn.benchmark = True

        # Disable gradient computation for inference
        self.torch.set_grad_enabled(False)
