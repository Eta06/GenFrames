import torch

from genframes.training.losses import (
    balanced_flow_loss,
    candidate_selection_loss,
    oracle_warp_loss,
    visibility_blend_loss,
)


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


def test_balanced_flow_loss_respects_teacher_spatial_mask() -> None:
    target = torch.zeros((1, 2, 4, 4))
    target[:, 0, 1, 1] = 3.0
    target[:, 0, 2, 2] = 9.0
    prediction = torch.zeros_like(target)
    mask = torch.zeros((1, 1, 4, 4), dtype=torch.bool)
    mask[:, :, 1, 1] = True
    loss = balanced_flow_loss(
        prediction,
        target,
        static_weight=0.0,
        spatial_mask=mask,
    )
    assert torch.allclose(loss, torch.tensor(3.0), atol=1e-5)


def test_oracle_warp_loss_accepts_one_aligned_endpoint_per_pixel() -> None:
    target = torch.rand((1, 3, 8, 8))
    wrong = 1.0 - target
    loss = oracle_warp_loss(target, wrong, target)
    assert loss < 0.0011
    assert oracle_warp_loss(wrong, wrong, target) > loss


def test_visibility_blend_loss_prefers_only_visible_endpoint() -> None:
    reference = torch.zeros((1, 3, 2, 2))
    batch = {
        "visibility0": torch.zeros((1, 1, 2, 2)),
        "visibility1": torch.ones((1, 1, 2, 2)),
        "visibility_valid": torch.tensor([True]),
        "time": torch.tensor([0.5]),
    }
    assert visibility_blend_loss(torch.ones((1, 1, 2, 2)), batch, reference) == 0
    assert visibility_blend_loss(torch.zeros((1, 1, 2, 2)), batch, reference) > 0


def test_candidate_selection_loss_prefers_best_fixed_candidate() -> None:
    target = torch.ones((1, 3, 2, 2))
    candidates = torch.stack((torch.zeros_like(target), target), dim=1)
    correct = torch.empty((1, 2, 2, 2))
    correct[:, 0] = -5.0
    correct[:, 1] = 5.0
    wrong = -correct
    assert candidate_selection_loss(correct, candidates, target) < candidate_selection_loss(
        wrong, candidates, target
    )
