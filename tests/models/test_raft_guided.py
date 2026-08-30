import torch
from torch import nn

from genframes.models import GenFramesRaftGuided, LinearBlend, RaftGuidedConfig


class _ZeroFlow(nn.Module):
    def forward(self, first, second):
        return [first.new_zeros((first.shape[0], 2, first.shape[2], first.shape[3]))]


def test_raft_guided_initializes_to_time_blend_with_zero_flow() -> None:
    frame0 = torch.rand((2, 3, 31, 39))
    frame1 = torch.rand_like(frame0)
    time = torch.tensor((0.25, 0.75))
    model = GenFramesRaftGuided(
        RaftGuidedConfig(fusion_channels=8), flow_estimator=_ZeroFlow()
    )
    output = model(frame0, frame1, time)
    expected = LinearBlend()(frame0, frame1, time).frame
    assert torch.allclose(output.frame, expected, atol=1e-6)
    assert output.auxiliary["flow_t0"].shape == (2, 2, 31, 39)
    assert output.auxiliary["weight1"].shape == (2, 1, 31, 39)


def test_raft_guided_fusion_has_finite_gradients() -> None:
    model = GenFramesRaftGuided(
        RaftGuidedConfig(fusion_channels=8), flow_estimator=_ZeroFlow()
    )
    frame0 = torch.rand((1, 3, 24, 32))
    frame1 = torch.rand_like(frame0)
    output = model(frame0, frame1, 0.5)
    output.frame.mean().backward()
    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if not name.startswith("flow_estimator.") and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
