import torch
from torch import nn

from genframes.models import GenFramesRaftRegionAssignment, RaftRegionAssignmentConfig


class _ConstantFlow(nn.Module):
    def forward(self, first, second):
        flow = first.new_zeros((first.shape[0], 2, first.shape[2], first.shape[3]))
        flow[:, 0] = 2.0
        return [flow]


def test_region_assignment_is_discrete_and_preserves_candidates() -> None:
    model = GenFramesRaftRegionAssignment(
        RaftRegionAssignmentConfig(selector_channels=8), flow_estimator=_ConstantFlow()
    ).eval()
    frame0 = torch.rand((1, 3, 24, 32))
    frame1 = torch.rand_like(frame0)
    output = model(frame0, frame1, 0.5)
    auxiliary = output.auxiliary
    assert auxiliary["candidate_stack"].shape == (1, 10, 3, 24, 32)
    assert auxiliary["assignment_probabilities"].shape == (1, 11, 24, 32)
    assert auxiliary["assignment_index"].shape == (1, 24, 32)
    assert auxiliary["source0_probability"].shape == (1, 1, 24, 32)
    appearances = torch.cat(
        (auxiliary["candidate_stack"], auxiliary["fallback_frame"][:, None]), dim=1
    )
    chosen = appearances.gather(
        1, auxiliary["assignment_index"][:, None, None].expand(-1, 1, 3, -1, -1)
    ).squeeze(1)
    assert torch.allclose(output.frame, chosen)


def test_region_assignment_checkpoint_excludes_flow_oracle() -> None:
    model = GenFramesRaftRegionAssignment(flow_estimator=_ConstantFlow())
    state = model.selector_state_dict()
    assert state and not any(name.startswith("flow_estimator.") for name in state)
