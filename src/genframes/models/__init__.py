"""GenFrames model contracts and research candidates."""

from .base import FrameInterpolator, InterpolationOutput, prepare_time
from .baselines import LinearBlend
from .bilateral_flow import BilateralFlowConfig, GenFramesBilateralFlow
from .mini import GenFramesMini, MiniConfig
from .raft_evidence_selector import GenFramesRaftEvidenceSelector, RaftEvidenceSelectorConfig
from .raft_guided import GenFramesRaftGuided, RaftGuidedConfig
from .raft_multifield import GenFramesRaftMultiField, RaftMultiFieldConfig
from .raft_region_assignment import GenFramesRaftRegionAssignment, RaftRegionAssignmentConfig

__all__ = [
    "FrameInterpolator",
    "BilateralFlowConfig",
    "GenFramesBilateralFlow",
    "GenFramesMini",
    "GenFramesRaftGuided",
    "GenFramesRaftEvidenceSelector",
    "GenFramesRaftMultiField",
    "GenFramesRaftRegionAssignment",
    "InterpolationOutput",
    "LinearBlend",
    "MiniConfig",
    "RaftGuidedConfig",
    "RaftEvidenceSelectorConfig",
    "RaftMultiFieldConfig",
    "RaftRegionAssignmentConfig",
    "prepare_time",
]
