from torch.utils.data import DataLoader

from genframes.data.analytic import AnalyticMotionDataset
from genframes.eval import evaluate_model
from genframes.models import LinearBlend


def test_evaluation_runner_reports_learning_free_baseline() -> None:
    loader = DataLoader(
        AnalyticMotionDataset(length=4, height=24, width=24, seed=3),
        batch_size=2,
    )
    result = evaluate_model(LinearBlend(), loader, device="cpu")
    assert result.samples == 4
    assert result.parameters == 0
    assert result.mae > 0
    assert result.psnr > 0
    assert result.moving_mae is not None
    assert result.moving_psnr is not None
    assert result.occlusion_mae is not None
    assert result.flow_epe is None
