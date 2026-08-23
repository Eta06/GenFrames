import torch

from genframes.eval import batch_masked_mae, batch_masked_psnr


def test_masked_metrics_exclude_unselected_pixels() -> None:
    target = torch.zeros((1, 3, 2, 2))
    prediction = torch.ones_like(target)
    prediction[:, :, 0, 0] = 0.5
    mask = torch.zeros((1, 1, 2, 2), dtype=torch.bool)
    mask[:, :, 0, 0] = True
    mae, valid = batch_masked_mae(prediction, target, mask)
    psnr, _ = batch_masked_psnr(prediction, target, mask)
    assert valid.tolist() == [True]
    assert torch.allclose(mae, torch.tensor([0.5]))
    assert torch.allclose(psnr, torch.tensor([6.0206]), atol=1e-4)
