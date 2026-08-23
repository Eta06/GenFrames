"""Training components for GenFrames candidates."""

from .losses import InterpolationLoss, LossConfig
from .trainer import TrainConfig, TrainResult, train_steps

__all__ = ["InterpolationLoss", "LossConfig", "TrainConfig", "TrainResult", "train_steps"]

