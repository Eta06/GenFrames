import torch

from genframes.ops import correlation_shift_index, correlation_soft_argmax, local_correlation


def test_local_correlation_finds_known_positive_x_shift() -> None:
    first = torch.zeros((1, 1, 5, 6))
    second = torch.zeros_like(first)
    first[:, :, 2, 2] = 1.0
    second[:, :, 2, 3] = 1.0
    costs = local_correlation(first, second, radius=2)
    expected = correlation_shift_index(0, 1, radius=2)
    assert costs[0, :, 2, 2].argmax().item() == expected
    assert costs[0, expected, 2, 2] == 1.0


def test_local_correlation_shape_and_gradients() -> None:
    first = torch.rand((2, 4, 7, 9), requires_grad=True)
    second = torch.rand_like(first, requires_grad=True)
    output = local_correlation(first, second, radius=1)
    assert output.shape == (2, 9, 7, 9)
    output.mean().backward()
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert second.grad is not None and torch.isfinite(second.grad).all()


def test_soft_argmax_decodes_explicit_displacement() -> None:
    costs = torch.full((1, 25, 2, 3), -10.0)
    costs[:, correlation_shift_index(-1, 2, radius=2)] = 10.0
    displacement = correlation_soft_argmax(costs, radius=2, temperature=0.1)
    assert torch.allclose(displacement[:, 0], torch.full((1, 2, 3), 2.0))
    assert torch.allclose(displacement[:, 1], torch.full((1, 2, 3), -1.0))
