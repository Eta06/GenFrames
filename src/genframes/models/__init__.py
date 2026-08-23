"""GenFrames model contracts and research candidates."""

from .base import FrameInterpolator, InterpolationOutput, prepare_time
from .baselines import LinearBlend

__all__ = ["FrameInterpolator", "InterpolationOutput", "LinearBlend", "prepare_time"]

