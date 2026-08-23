"""Train GenFramesMini on deterministic analytic motion."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from genframes.data.analytic import AnalyticMotionDataset
from genframes.eval import evaluate_model
from genframes.models import (
    BilateralFlowConfig,
    GenFramesBilateralFlow,
    GenFramesMini,
    LinearBlend,
    MiniConfig,
)
from genframes.training import TrainConfig, train_steps


def main() -> None:
    arguments = parse_arguments()
    device = torch.device(arguments.device)
    if arguments.model == "mini":
        model_config = MiniConfig(base_channels=arguments.channels)
        model = GenFramesMini(model_config)
    else:
        model_config = BilateralFlowConfig(
            base_channels=arguments.channels,
            max_flow=arguments.max_flow,
        )
        model = GenFramesBilateralFlow(model_config)
    train_config = TrainConfig(
        steps=arguments.steps,
        learning_rate=arguments.learning_rate,
        amp=not arguments.no_amp,
        seed=arguments.seed,
        log_every=arguments.log_every,
    )
    training = AnalyticMotionDataset(
        length=arguments.train_samples,
        height=arguments.size,
        width=arguments.size,
        seed=arguments.seed,
    )
    validation = AnalyticMotionDataset(
        length=arguments.validation_samples,
        height=arguments.size,
        width=arguments.size,
        seed=arguments.seed + 1_000_000,
    )
    training_loader = DataLoader(
        training,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(validation, batch_size=arguments.batch_size, shuffle=False)
    baseline = evaluate_model(LinearBlend(), validation_loader, device=device)

    def report(step: int, losses: dict[str, float]) -> None:
        print(json.dumps({"step": step, **losses}), flush=True)

    training_result = train_steps(
        model,
        training_loader,
        device=device,
        config=train_config,
        callback=report,
    )
    evaluation = evaluate_model(model, validation_loader, device=device)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.safetensors"
    checkpoint_tensors = {
        name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
    }
    save_file(checkpoint_tensors, checkpoint)
    result = {
        "model": asdict(model_config),
        "training": asdict(train_config),
        "dataset": {
            "train_samples": arguments.train_samples,
            "validation_samples": arguments.validation_samples,
            "size": arguments.size,
            "batch_size": arguments.batch_size,
        },
        "baseline": asdict(baseline),
        "train_result": asdict(training_result),
        "evaluation": asdict(evaluation),
        "checkpoint": str(checkpoint),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=("mini", "bilateral-flow"), default="mini")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=2_048)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--max-flow", type=float, default=20.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
