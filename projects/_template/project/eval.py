"""Evaluation logic template."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml_portfolio.common.logging import get_logger
from ml_portfolio.eval.slicing import SliceEvaluator

logger = get_logger(__name__)


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    batch_size: int = 32
    metrics: list[str] | None = None
    slices: dict[str, list[str]] | None = None


@dataclass
class EvalResult:
    """Evaluation result."""

    metrics: dict[str, float]
    slices: list[dict[str, Any]]
    predictions: list[Any]
    targets: list[Any]


def evaluate(
    model: nn.Module,
    dataloader: DataLoader[Any],
    config: EvalConfig,
    device: str = "cpu",
) -> EvalResult:
    """Evaluate model on dataset.

    Args:
        model: Model to evaluate.
        dataloader: Evaluation dataloader.
        config: Evaluation configuration.
        device: Device to use.

    Returns:
        Evaluation results.
    """
    model.eval()
    model = model.to(device)

    all_predictions: list[Any] = []
    all_targets: list[Any] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device and run inference
            # predictions = model(batch["input"])
            # targets = batch["target"]

            # Placeholder - implement your evaluation logic
            predictions: list[Any] = []
            targets: list[Any] = []

            all_predictions.extend(predictions)
            all_targets.extend(targets)

    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets, config.metrics)

    # Compute slice metrics (placeholder)
    slices: list[dict[str, Any]] = []

    return EvalResult(
        metrics=metrics,
        slices=slices,
        predictions=all_predictions,
        targets=all_targets,
    )


def compute_metrics(
    predictions: list[Any],
    targets: list[Any],
    metric_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute evaluation metrics.

    Args:
        predictions: Model predictions.
        targets: Ground truth targets.
        metric_names: List of metrics to compute.

    Returns:
        Dictionary of metric name to value.
    """
    metrics: dict[str, float] = {}

    # Implement metric computation based on your problem type
    # Example for classification:
    # if "accuracy" in (metric_names or []):
    #     metrics["accuracy"] = accuracy_score(targets, predictions)

    return metrics


def save_results(
    result: EvalResult,
    output_dir: Path,
    run_id: str,
) -> None:
    """Save evaluation results.

    Args:
        result: Evaluation results.
        output_dir: Output directory.
        run_id: MLflow run ID.
    """
    report_dir = output_dir / run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_path = report_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result.metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Save slices
    if result.slices:
        slices_path = report_dir / "slices.csv"
        # Write slices to CSV
        logger.info(f"Saved slices to {slices_path}")

    # Create plots directory
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Generate and save plots
    # save_calibration_plot(result, plots_dir / "calibration.png")
    # save_confusion_matrix(result, plots_dir / "confusion.png")


def run_robustness_evaluation(
    model: nn.Module,
    dataloader: DataLoader[Any],
    perturbations: list[dict[str, Any]],
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Run robustness evaluation with perturbations.

    Args:
        model: Model to evaluate.
        dataloader: Evaluation dataloader.
        perturbations: List of perturbation configs.
        device: Device to use.

    Returns:
        List of results per perturbation.
    """
    results = []

    for perturbation in perturbations:
        # Apply perturbation and evaluate
        # perturbed_result = evaluate_with_perturbation(model, dataloader, perturbation)

        results.append(
            {
                "perturbation": perturbation["name"],
                "intensity": perturbation.get("intensity", 1.0),
                # "metrics": perturbed_result.metrics,
            }
        )

    return results
