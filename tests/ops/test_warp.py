import torch

from genframes.ops import backward_warp


def test_zero_flow_is_identity() -> None:
    source = torch.rand((2, 3, 7, 9))
    flow = torch.zeros((2, 2, 7, 9))
    assert torch.allclose(backward_warp(source, flow), source, atol=1e-6)


def test_positive_horizontal_flow_samples_from_the_right() -> None:
    source = torch.arange(5, dtype=torch.float32).view(1, 1, 1, 5)
    flow = torch.zeros((1, 2, 1, 5))
    flow[:, 0] = 1.0
    expected = torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 4.0]]]])
    assert torch.allclose(backward_warp(source, flow), expected, atol=1e-6)


def test_warp_has_finite_flow_gradients() -> None:
    source = torch.rand((1, 3, 6, 6))
    flow = torch.zeros((1, 2, 6, 6), requires_grad=True)
    backward_warp(source, flow).mean().backward()
    assert flow.grad is not None
    assert torch.isfinite(flow.grad).all()

