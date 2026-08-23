import torch

from genframes.models import GenFramesMini, LinearBlend, MiniConfig


def test_mini_initializes_to_linear_blend_for_odd_shapes() -> None:
    torch.manual_seed(4)
    frame0 = torch.rand((2, 3, 31, 37))
    frame1 = torch.rand_like(frame0)
    time = torch.tensor([0.25, 0.75])
    expected = LinearBlend()(frame0, frame1, time).frame
    output = GenFramesMini(MiniConfig(base_channels=8))(frame0, frame1, time)
    assert output.frame.shape == frame0.shape
    assert torch.equal(output.frame, expected)
    assert output.auxiliary["confidence"].shape == (2, 1, 31, 37)


def test_mini_has_finite_gradients() -> None:
    model = GenFramesMini(MiniConfig(base_channels=8))
    frame0 = torch.rand((1, 3, 24, 24))
    frame1 = torch.rand_like(frame0)
    target = torch.rand_like(frame0)
    loss = (model(frame0, frame1, 0.5).frame - target).abs().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

