import torch
from torch import nn

from genframes.models import GenFramesRaftMultiField, LinearBlend, RaftMultiFieldConfig


class _ZeroFlow(nn.Module):
    def forward(self, first, second):
        return [first.new_zeros((first.shape[0], 2, first.shape[2], first.shape[3]))]


def test_multifield_initializes_to_quadratic_time_blend() -> None:
    frame0 = torch.rand((2, 3, 24, 32))
    frame1 = torch.rand_like(frame0)
    time = torch.tensor((0.25, 0.75))
    model = GenFramesRaftMultiField(
        RaftMultiFieldConfig(selector_channels=8), flow_estimator=_ZeroFlow()
    )
    output = model(frame0, frame1, time)
    expected = LinearBlend()(frame0, frame1, time).frame
    assert torch.allclose(output.frame, expected, atol=2e-3)
    assert output.auxiliary["candidate_stack"].shape == (2, 10, 3, 24, 32)
    assert output.auxiliary["candidate_weights"].shape == (2, 10, 24, 32)


def test_multifield_selector_checkpoint_excludes_flow_oracle() -> None:
    source = GenFramesRaftMultiField(flow_estimator=_ZeroFlow())
    state = source.selector_state_dict()
    assert state and not any(name.startswith("flow_estimator.") for name in state)
    target = GenFramesRaftMultiField(flow_estimator=_ZeroFlow())
    target.load_selector_state_dict(state)
