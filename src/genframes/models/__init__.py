"""GenFrames model contracts and research candidates."""

from .base import FrameInterpolator, InterpolationOutput, prepare_time
from .baselines import LinearBlend
from .bilateral_flow import BilateralFlowConfig, GenFramesBilateralFlow
from .mini import GenFramesMini, MiniConfig
from .raft_guided import GenFramesRaftGuided, RaftGuidedConfig
from .raft_multifield import GenFramesRaftMultiField, RaftMultiFieldConfig

__all__ = [
    "FrameInterpolator",
    "BilateralFlowConfig",
    "GenFramesBilateralFlow",
    "GenFramesMini",
    "GenFramesRaftGuided",
    "GenFramesRaftMultiField",
    "InterpolationOutput",
    "LinearBlend",
    "MiniConfig",
    "RaftGuidedConfig",
    "RaftMultiFieldConfig",
    "prepare_time",
]
