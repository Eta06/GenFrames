import torch

from genframes.data.analytic import AnalyticMotionDataset


def test_analytic_sample_contract() -> None:
    sample = AnalyticMotionDataset(length=2, height=32, width=40, seed=5)[0]
    assert sample["frame0"].shape == (3, 32, 40)
    assert sample["frame1"].shape == (3, 32, 40)
    assert sample["target"].shape == (3, 32, 40)
    assert sample["flow_t0"].shape == (2, 32, 40)
    assert sample["visibility0"].shape == (1, 32, 40)
    assert 0.0 < sample["time"].item() < 1.0


def test_analytic_generation_is_deterministic() -> None:
    dataset = AnalyticMotionDataset(length=1, height=32, width=32, seed=99)
    first = dataset[0]
    second = dataset[0]
    for key in ("frame0", "frame1", "target", "flow_t0", "flow_t1"):
        assert torch.equal(first[key], second[key])


def test_midpoint_mode_is_exact() -> None:
    sample = AnalyticMotionDataset(
        length=1, height=32, width=32, seed=1, midpoint_only=True
    )[0]
    assert sample["time"].item() == 0.5

