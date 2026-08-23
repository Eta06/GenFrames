"""Learning-free lower bounds for experiment validation."""

from torch import Tensor

from .base import FrameInterpolator, InterpolationOutput, prepare_time


class LinearBlend(FrameInterpolator):
    """Blend endpoints in proportion to target time."""

    def forward(self, frame0: Tensor, frame1: Tensor, time: Tensor | float) -> InterpolationOutput:
        self.validate_frames(frame0, frame1)
        target_time = prepare_time(time, frame0)
        frame = (1.0 - target_time) * frame0 + target_time * frame1
        return InterpolationOutput(frame=frame)

