#!/usr/bin/env python
"""Evaluate TrOCR model and generate reports."""

from pathlib import Path

import mlflow
import torch
import typer
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ml_portfolio.common.logging import get_logger, setup_logging
from ml_portfolio.common.paths import get_project_paths
from ml_portfolio.tracking.mlflow_utils import setup_mlflow

from projects.llm_ocr_trocr.project.data import SROIEDataset
from projects.llm_ocr_trocr.project.eval import (
    evaluate,
    evaluate_robustness,
    save_results,
    EvalConfig,
)

app = typer.Typer()
console = Console()
logger = get_logger(__name__)

PROJECT_NAME = "llm_ocr_trocr"


def get_device() -> str:
    """Get best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@app.command()
def main(
    run_id: str = typer.Option(
        None,
        "--run-id",
        "-r",
        help="MLflow run ID to evaluate",
    ),
    model_path: Path = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Path to model checkpoint (alternative to run-id)",
    ),
    config_path: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config YAML for evaluation settings",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for reports",
    ),
    robustness: bool = typer.Option(
        False,
        "--robustness",
        help="Run robustness evaluation (slower)",
    ),
    device: str = typer.Option(
        None,
        "--device",
        "-d",
        help="Device to use (cuda, mps, cpu)",
    ),
) -> None:
    """Evaluate TrOCR model and generate reports."""
    setup_logging()

    if run_id is None and model_path is None:
        console.print("[red]Error: Must provide either --run-id or --model-path[/red]")
        raise typer.Exit(1)

    paths = get_project_paths(PROJECT_NAME)
    output_dir = output_dir or paths.reports_dir
    device = device or get_device()

    # Load config
    config_path = config_path or paths.default_config()
    config = OmegaConf.load(config_path)

    # Load model
    if run_id:
        setup_mlflow(PROJECT_NAME)
        logger.info(f"Loading model from run {run_id}")

        # Get model path from MLflow
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)

        # Download model artifact
        artifact_path = client.download_artifacts(run_id, "model")
        model_path = Path(artifact_path) / "best_model"

    logger.info(f"Loading model from {model_path}")
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    model = model.to(device)

    # Create test dataset
    test_dataset = SROIEDataset(
        paths.data_dir,
        split="test",
        processor=processor,
    )

    if len(test_dataset) == 0:
        console.print("[red]No test data found. Run download_data.py first.[/red]")
        raise typer.Exit(1)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Create eval config
    eval_config = EvalConfig(
        batch_size=config.evaluation.batch_size,
        metrics=list(config.evaluation.metrics),
        slices=list(config.evaluation.slices),
        robustness=dict(config.evaluation.robustness) if robustness else {},
    )

    # Run evaluation
    logger.info("Running evaluation")
    result = evaluate(model, processor, test_loader, eval_config, device)

    # Run robustness evaluation if requested
    if robustness:
        logger.info("Running robustness evaluation")
        result.robustness_results = evaluate_robustness(
            model, processor, test_dataset, eval_config, device
        )

    # Save results
    result_run_id = run_id or "local"
    save_results(result, output_dir, result_run_id)

    # Log to MLflow if we have a run
    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({f"test_{k}": v for k, v in result.metrics.items()})
            mlflow.log_artifacts(str(output_dir / result_run_id), "evaluation")

    # Print results
    console.print("\n[bold]Evaluation Results[/bold]\n")

    # Primary metrics table
    metrics_table = Table(title="Primary Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")

    for name, value in result.metrics.items():
        if isinstance(value, float):
            metrics_table.add_row(name, f"{value:.4f}")
        else:
            metrics_table.add_row(name, str(value))

    console.print(metrics_table)

    # Slice metrics table
    if result.slice_results:
        console.print()
        slice_table = Table(title="Slice Metrics")
        slice_table.add_column("Slice", style="cyan")
        slice_table.add_column("Value", style="cyan")
        slice_table.add_column("CER", style="green")
        slice_table.add_column("WER", style="green")
        slice_table.add_column("Count", style="dim")

        for slice_result in result.slice_results:
            slice_table.add_row(
                slice_result["slice_name"],
                slice_result["slice_value"],
                f"{slice_result.get('cer', 0):.4f}",
                f"{slice_result.get('wer', 0):.4f}",
                str(slice_result["sample_count"]),
            )

        console.print(slice_table)

    # Robustness table
    if result.robustness_results:
        console.print()
        robust_table = Table(title="Robustness Analysis")
        robust_table.add_column("Perturbation", style="cyan")
        robust_table.add_column("Intensity", style="cyan")
        robust_table.add_column("Baseline CER", style="green")
        robust_table.add_column("Perturbed CER", style="yellow")
        robust_table.add_column("Degradation %", style="red")

        for r in result.robustness_results:
            robust_table.add_row(
                r["perturbation"],
                str(r["intensity"]),
                f"{r['baseline_value']:.4f}",
                f"{r['perturbed_value']:.4f}",
                f"{r['degradation_pct']:.1f}%",
            )

        console.print(robust_table)

    console.print(f"\n[green]Results saved to: {output_dir / result_run_id}[/green]")


if __name__ == "__main__":
    app()
