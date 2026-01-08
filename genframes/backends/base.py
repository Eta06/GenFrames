"""
Base backend class for hardware acceleration
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class BaseBackend(ABC):
    """Abstract base class for computation backends"""

    def __init__(self, device: Optional[str] = None):
        """
        Initialize backend

        Args:
            device: Device specification (e.g., 'mps', 'cuda', 'cpu')
        """
        self.device = device
        self._initialized = False

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the system"""
        pass

    @abstractmethod
    def get_device_name(self) -> str:
        """Get human-readable device name"""
        pass

    @abstractmethod
    def to_tensor(self, array: np.ndarray) -> Any:
        """Convert numpy array to backend-specific tensor"""
        pass

    @abstractmethod
    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert backend-specific tensor to numpy array"""
        pass

    @abstractmethod
    def load_model(self, model_path: str, model_class: Any) -> Any:
        """Load model weights for this backend"""
        pass

    def initialize(self):
        """Initialize backend (called once)"""
        if not self._initialized:
            self._initialized = True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device}, available={self.is_available()})"


def detect_best_backend() -> str:
    """
    Detect the best available backend for the current system

    Returns:
        Backend name: 'mlx', 'mps', 'cuda', or 'cpu'
    """
    import platform
    import sys

    # Check for MLX (Apple Silicon only)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx.core as mx
            # Test if MLX is working
            _ = mx.array([1.0])
            return "mlx"
        except (ImportError, Exception):
            pass

        # Fall back to MPS if MLX not available
        try:
            import torch
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return "mps"
        except ImportError:
            pass

    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    # Default to CPU
    return "cpu"


def get_backend(backend_name: str = "auto", device: Optional[str] = None) -> BaseBackend:
    """
    Get backend instance by name

    Args:
        backend_name: Backend to use ('auto', 'mlx', 'mps', 'cuda', 'cpu')
        device: Optional device specification

    Returns:
        Backend instance
    """
    if backend_name == "auto":
        backend_name = detect_best_backend()

    if backend_name == "mlx":
        from genframes.backends.mlx_backend import MLXBackend
        return MLXBackend(device=device)
    elif backend_name == "mps":
        from genframes.backends.torch_backend import TorchBackend
        return TorchBackend(device="mps")
    elif backend_name == "cuda":
        from genframes.backends.torch_backend import TorchBackend
        return TorchBackend(device="cuda")
    elif backend_name == "cpu":
        from genframes.backends.torch_backend import TorchBackend
        return TorchBackend(device="cpu")
    else:
        raise ValueError(f"Unknown backend: {backend_name}")
