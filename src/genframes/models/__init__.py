"""GenFrames model contracts and research candidates."""

from .base import FrameInterpolator, InterpolationOutput, prepare_time
from .baselines import LinearBlend
from .bilateral_flow import BilateralFlowConfig, GenFramesBilateralFlow
from .mini import GenFramesMini, MiniConfig
from .raft_guided import GenFramesRaftGuided, RaftGuidedConfig

__all__ = [
    "FrameInterpolator",
    "BilateralFlowConfig",
    "GenFramesBilateralFlow",
    "GenFramesMini",
    "GenFramesRaftGuided",
    "InterpolationOutput",
    "LinearBlend",
    "MiniConfig",
    "RaftGuidedConfig",
    "prepare_time",
]
