"""
GenFrames - AI-powered video frame interpolation
"""

__version__ = "0.1.0"

from genframes.interpolator import FrameInterpolator
from genframes.models.base import BaseModel
from genframes.backends.base import BaseBackend

__all__ = ["FrameInterpolator", "BaseModel", "BaseBackend"]
