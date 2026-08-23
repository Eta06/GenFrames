import pytest
import torch

from genframes.models import LinearBlend


def test_linear_blend_supports_batched_arbitrary_time() -> None:
    frame0 = torch.zeros((2, 3, 4, 5))
    frame1 = torch.ones_like(frame0)
    result = LinearBlend()(frame0, frame1, torch.tensor([0.25, 0.75])).frame
    assert torch.allclose(result[0], torch.full_like(result[0], 0.25))
    assert torch.allclose(result[1], torch.full_like(result[1], 0.75))


def test_linear_blend_rejects_endpoints() -> None:
    frame = torch.zeros((1, 3, 4, 5))
    with pytest.raises(ValueError, match="strictly inside"):
        LinearBlend()(frame, frame, 0.0)

