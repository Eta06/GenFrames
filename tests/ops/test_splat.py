import torch

from genframes.ops import forward_splat


def test_forward_splat_identity() -> None:
    source = torch.rand((2, 3, 5, 7))
    output, mass = forward_splat(source, torch.zeros((2, 2, 5, 7)))
    assert torch.allclose(output, source)
    assert torch.allclose(mass, torch.ones_like(mass))


def test_forward_splat_translates_and_exposes_holes() -> None:
    source = torch.arange(4, dtype=torch.float32).view(1, 1, 1, 4)
    flow = torch.zeros((1, 2, 1, 4))
    flow[:, 0] = 1.0
    output, mass = forward_splat(source, flow)
    assert mass[0, 0, 0, 0] == 0
    assert torch.equal(output[0, 0, 0, 1:], source[0, 0, 0, :-1])


def test_forward_splat_importance_resolves_collision() -> None:
    source = torch.tensor([[[[0.0, 1.0]]]])
    flow = torch.tensor([[[[1.0, 0.0]], [[0.0, 0.0]]]])
    importance = torch.tensor([[[[1.0, 3.0]]]])
    output, mass = forward_splat(source, flow, importance=importance)
    assert torch.allclose(output[0, 0, 0, 1], torch.tensor(0.75))
    assert torch.allclose(mass[0, 0, 0, 1], torch.tensor(4.0))
