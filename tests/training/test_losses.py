import torch

from genframes.training.losses import balanced_flow_loss


def test_balanced_flow_loss_does_not_dilute_sparse_motion() -> None:
    target = torch.zeros((1, 2, 16, 16))
    target[:, 0, 8, 8] = 4.0
    prediction = torch.zeros_like(target)
    loss = balanced_flow_loss(prediction, target, static_weight=0.0)
    assert torch.allclose(loss, torch.tensor(4.0), atol=1e-5)


def test_balanced_flow_loss_penalizes_stationary_region_leakage() -> None:
    target = torch.zeros((1, 2, 4, 4))
    prediction = torch.ones_like(target)
    loss = balanced_flow_loss(prediction, target, static_weight=0.5)
    assert torch.allclose(loss, torch.tensor(2**0.5 * 0.5), atol=1e-5)


def test_balanced_flow_loss_ignores_real_samples_without_flow() -> None:
    target = torch.ones((2, 2, 4, 4))
    prediction = torch.zeros_like(target)
    valid = torch.tensor([True, False])
    loss = balanced_flow_loss(prediction, target, static_weight=0.0, valid_samples=valid)
    assert torch.allclose(loss, torch.tensor(2**0.5), atol=1e-5)
