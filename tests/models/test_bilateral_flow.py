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


def test_local_correspondence_variant_starts_from_same_function() -> None:
    frame0 = torch.rand((1, 3, 32, 40))
    frame1 = torch.rand_like(frame0)
    model = GenFramesBilateralFlow(
        BilateralFlowConfig(
            base_channels=8,
            coarse_velocity=True,
            correlation_radius=2,
            correspondence_channels=4,
        )
    )
    output = model(frame0, frame1, 0.5)
    expected = LinearBlend()(frame0, frame1, 0.5).frame
    assert torch.allclose(output.frame, expected, atol=1e-6)
    assert output.auxiliary["correspondence_velocity"].shape == (1, 2, 32, 40)


def test_explicit_moment_variant_starts_from_same_function() -> None:
    frame0 = torch.rand((1, 3, 32, 40))
    frame1 = torch.rand_like(frame0)
    model = GenFramesBilateralFlow(
        BilateralFlowConfig(
            base_channels=8,
            coarse_velocity=True,
            correlation_radius=2,
            correspondence_channels=4,
            correlation_moments=True,
        )
    )
    output = model(frame0, frame1, 0.5)
    expected = LinearBlend()(frame0, frame1, 0.5).frame
    assert torch.allclose(output.frame, expected, atol=1e-6)
    assert output.auxiliary["correspondence_moment_scale"].shape == ()


def test_pyramid_refinement_starts_from_same_function_on_odd_shape() -> None:
    frame0 = torch.rand((1, 3, 35, 43))
    frame1 = torch.rand_like(frame0)
    model = GenFramesBilateralFlow(
        BilateralFlowConfig(
            base_channels=8,
            coarse_velocity=True,
            pyramid_refinement=True,
            pyramid_channels=(4, 6, 8, 10),
            pyramid_radii=(1, 1, 1, 2),
        )
    )
    output = model(frame0, frame1, 0.25)
    expected = LinearBlend()(frame0, frame1, 0.25).frame
    assert torch.allclose(output.frame, expected, atol=1e-6)
    assert output.auxiliary["pyramid_velocity_correction"].shape == (1, 2, 35, 43)
    assert torch.count_nonzero(output.auxiliary["pyramid_velocity_correction"]) == 0


def test_pyramid_refinement_has_finite_gradients() -> None:
    model = GenFramesBilateralFlow(
        BilateralFlowConfig(
            base_channels=8,
            pyramid_refinement=True,
            pyramid_channels=(4, 6, 8, 10),
            pyramid_radii=(1, 1, 1, 2),
        )
    )
    frame0 = torch.rand((1, 3, 32, 40))
    frame1 = torch.rand_like(frame0)
    output = model(frame0, frame1, 0.5)
    output.frame.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in model.pyramid_refiner.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_pyramid_refinement_supports_cuda_autocast() -> None:
    if not torch.cuda.is_available():
        return
    model = GenFramesBilateralFlow(
        BilateralFlowConfig(
            base_channels=8,
            pyramid_refinement=True,
            pyramid_channels=(4, 6, 8, 10),
            pyramid_radii=(1, 1, 1, 2),
        )
    ).cuda()
    frame0 = torch.rand((1, 3, 32, 40), device="cuda")
    frame1 = torch.rand_like(frame0)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(frame0, frame1, 0.5)
        output.frame.mean().backward()
    assert torch.isfinite(output.frame).all()


def test_independent_endpoint_flows_start_from_same_function() -> None:
    frame0 = torch.rand((1, 3, 31, 39))
    frame1 = torch.rand_like(frame0)
    model = GenFramesBilateralFlow(
        BilateralFlowConfig(
            base_channels=8,
            coarse_velocity=True,
            independent_endpoint_flows=True,
        )
    )
    output = model(frame0, frame1, 0.25)
    expected = LinearBlend()(frame0, frame1, 0.25).frame
    assert torch.allclose(output.frame, expected, atol=1e-6)
    assert output.auxiliary["independent_velocity"].shape == (1, 4, 31, 39)
    assert torch.count_nonzero(output.auxiliary["independent_velocity"]) == 0


def test_independent_endpoint_head_can_break_shared_velocity_constraint() -> None:
    model = GenFramesBilateralFlow(
        BilateralFlowConfig(base_channels=8, independent_endpoint_flows=True)
    )
    assert model.independent_flow_head is not None
    with torch.no_grad():
        model.independent_flow_head.bias.copy_(torch.tensor((0.1, 0.0, -0.1, 0.0)))
    frame0 = torch.rand((1, 3, 24, 24))
    frame1 = torch.rand_like(frame0)
    output = model(frame0, frame1, 0.5)
    assert not torch.allclose(
        output.auxiliary["endpoint_velocity0"],
        output.auxiliary["endpoint_velocity1"],
    )
