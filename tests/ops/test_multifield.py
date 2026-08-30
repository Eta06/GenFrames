import torch

from genframes.ops import inverse_flow_hypotheses


def test_constant_endpoint_flow_has_stable_inverse_hypotheses() -> None:
    flow = torch.zeros((2, 2, 8, 9))
    flow[:, 0] = 6.0
    fields = inverse_flow_hypotheses(flow, torch.tensor([0.25, 0.5]))
    assert len(fields) == 4
    for field in fields:
        assert torch.allclose(field[0, 0], torch.full((8, 9), -1.5))
        assert torch.allclose(field[1, 0], torch.full((8, 9), -3.0))
        assert torch.count_nonzero(field[:, 1]) == 0


def test_inverse_hypotheses_validate_configuration() -> None:
    flow = torch.zeros((1, 2, 4, 4))
    try:
        inverse_flow_hypotheses(flow, 0.5, refinement_steps=())
    except ValueError as error:
        assert "refinement_steps" in str(error)
    else:
        raise AssertionError("empty steps must fail")
