from torch.utils.data import DataLoader

from genframes.data.analytic import AnalyticMotionDataset
from genframes.models import GenFramesMini, MiniConfig
from genframes.training import TrainConfig, train_steps


def test_trainer_updates_mini_model_on_cpu() -> None:
    dataset = AnalyticMotionDataset(length=4, height=16, width=16, seed=7)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = GenFramesMini(MiniConfig(base_channels=4))
    result = train_steps(
        model,
        loader,
        device="cpu",
        config=TrainConfig(steps=2, amp=False, log_every=1),
    )
    assert result.steps == 2
    assert result.minimum_loss > 0

