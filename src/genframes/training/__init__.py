"""Training components for GenFrames candidates."""

from .losses import InterpolationLoss, LossConfig
from .privileged_teacher import PrivilegedRaftSmallTeacher
from .trainer import TrainConfig, TrainResult, train_steps

__all__ = [
    "InterpolationLoss",
    "LossConfig",
    "PrivilegedRaftSmallTeacher",
    "TrainConfig",
    "TrainResult",
    "train_steps",
]
