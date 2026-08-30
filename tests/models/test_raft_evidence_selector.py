import torch
from torch import nn

from genframes.models import GenFramesRaftEvidenceSelector, RaftEvidenceSelectorConfig


class _ConstantFlow(nn.Module):
    def forward(self, first, second):
        flow = first.new_zeros((first.shape[0], 2, first.shape[2], first.shape[3]))
        flow[:, 0] = 2.0
        return [flow]


def test_evidence_selector_preserves_ten_field_representation() -> None:
    model = GenFramesRaftEvidenceSelector(
        RaftEvidenceSelectorConfig(selector_channels=8), flow_estimator=_ConstantFlow()
    )
    frame0 = torch.rand((1, 3, 24, 32))
    frame1 = torch.rand_like(frame0)
    output = model(frame0, frame1, 0.5)
    assert output.frame.shape == frame0.shape
    assert output.auxiliary["candidate_stack"].shape == (1, 10, 3, 24, 32)
    assert output.auxiliary["candidate_evidence"].shape == (1, 10, 6, 24, 32)
    assert output.auxiliary["candidate_fields"].shape == (1, 10, 2, 24, 32)
    assert torch.allclose(
        output.auxiliary["candidate_weights"].sum(1), torch.ones((1, 24, 32))
    )


def test_evidence_selector_checkpoint_excludes_flow_oracle() -> None:
    source = GenFramesRaftEvidenceSelector(flow_estimator=_ConstantFlow())
    state = source.selector_state_dict()
    assert state and not any(name.startswith("flow_estimator.") for name in state)
    target = GenFramesRaftEvidenceSelector(flow_estimator=_ConstantFlow())
    target.load_selector_state_dict(state)
