import torch

from genframes.models import BilateralFlowConfig, GenFramesBilateralFlow, LinearBlend


def test_bilateral_flow_initializes_to_linear_blend() -> None:
    torch.manual_seed(9)
    frame0 = torch.rand((2, 3, 31, 35))
    frame1 = torch.rand_like(frame0)
    time = torch.tensor([0.25, 0.75])
    expected = LinearBlend()(frame0, frame1, time).frame
    model = GenFramesBilateralFlow(BilateralFlowConfig(base_channels=8))
    output = model(frame0, frame1, time)
    # A zero-flow grid_sample round trip is accurate to roughly one float32 ULP.
    assert torch.allclose(output.frame, expected, atol=1e-6)
    assert output.auxiliary["flow_t0"].shape == (2, 2, 31, 35)
    assert output.auxiliary["weight1"].shape == (2, 1, 31, 35)


def test_bilateral_flow_has_finite_gradients() -> None:
    model = GenFramesBilateralFlow(BilateralFlowConfig(base_channels=8))
    frame0 = torch.rand((1, 3, 24, 24))
    frame1 = torch.rand_like(frame0)
    target = torch.rand_like(frame0)
    output = model(frame0, frame1, 0.5)
    (output.frame - target).abs().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_coarse_velocity_variant_starts_from_same_function() -> None:
    frame0 = torch.rand((1, 3, 32, 40))
    frame1 = torch.rand_like(frame0)
    model = GenFramesBilateralFlow(
        BilateralFlowConfig(base_channels=8, coarse_velocity=True)
    )
    output = model(frame0, frame1, 0.25)
    expected = LinearBlend()(frame0, frame1, 0.25).frame
    assert torch.allclose(output.frame, expected, atol=1e-6)
    assert output.auxiliary["coarse_velocity_logits"].shape == (1, 2, 32, 40)
