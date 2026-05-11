#!/usr/bin/env python
"""Evaluate SSL representations and downstream classifiers."""

from pathlib import Path

import mlflow
import typer
from omegaconf import OmegaConf
from rich.console import Console

from ml_portfolio.common.logging import get_logger, setup_logging
from ml_portfolio.common.paths import get_project_paths
from ml_portfolio.tracking.mlflow_utils import setup_mlflow

app = typer.Typer()
console = Console()
logger = get_logger(__name__)

PROJECT_NAME = "vision_ssl_transfer"


def get_device() -> str:
    """Determine best available device."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@app.command()
def knn(
    ssl_run_id: str = typer.Option(
        ...,
        "--ssl-run-id",
        "-s",
        help="MLflow run ID of pretrained SSL model",
    ),
    config_path: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config YAML",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for results",
    ),
) -> None:
    """Evaluate SSL representations using k-NN (no training required)."""
    setup_logging()

    paths = get_project_paths(PROJECT_NAME)
    config_path = config_path or paths.default_config()
    config = OmegaConf.load(config_path)
    output_dir = output_dir or paths.reports_dir

    experiment_name = config.get("mlflow", {}).get("experiment_name", PROJECT_NAME)
    setup_mlflow(experiment_name)

    device = get_device()

    with mlflow.start_run(run_name="knn_evaluation"):
        mlflow.set_tags(
            {
                "project": PROJECT_NAME,
                "stage": "evaluation",
                "method": "knn",
                "ssl_run_id": ssl_run_id,
            }
        )

        from projects.vision_ssl_transfer.project.data import (
            DataConfig,
            create_supervised_dataloaders,
        )
        from projects.vision_ssl_transfer.project.eval import (
            EvalConfig,
            evaluate_representation_quality,
            save_evaluation_results,
        )
        from projects.vision_ssl_transfer.project.model import SSLBackbone

        # Load SSL backbone
        logger.info(f"Loading SSL model from run {ssl_run_id}")
        artifact_path = mlflow.artifacts.download_artifacts(run_id=ssl_run_id, artifact_path="best_ssl_model.pt")
        backbone = SSLBackbone.from_pretrained(Path(artifact_path))

        # Create data loaders
        data_config = DataConfig(
            data_dir=paths.data_dir,
            dataset=config.data.dataset,
            image_size=config.data.image_size,
        )

        train_loader, _, test_loader = create_supervised_dataloaders(
            paths.data_dir,
            data_config,
            batch_size=config.evaluation.batch_size if hasattr(config, "evaluation") else 64,
        )

        # Evaluate
        eval_config = EvalConfig(
            knn_k=config.transfer.knn.k,
            knn_temperature=config.transfer.knn.temperature,
        )

        results = evaluate_representation_quality(
            backbone,
            train_loader,
            test_loader,
            eval_config,
            device=device,
        )

        # Log and save
        mlflow.log_metrics(
            {
                "knn_accuracy": results["knn_accuracy"],
                "knn_f1": results["knn_f1"],
            }
        )

        save_evaluation_results(results, output_dir, ssl_run_id)

        console.print("[green]k-NN evaluation complete![/green]")
        console.print(f"Accuracy: {results['knn_accuracy']:.4f}")
        console.print(f"F1 Score: {results['knn_f1']:.4f}")


@app.command()
def robustness(
    run_id: str = typer.Option(
        ...,
        "--run-id",
        "-r",
        help="MLflow run ID of trained classifier",
    ),
    config_path: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config YAML",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for results",
    ),
) -> None:
    """Evaluate model robustness under corruptions."""
    setup_logging()

    paths = get_project_paths(PROJECT_NAME)
    config_path = config_path or paths.default_config()
    config = OmegaConf.load(config_path)
    output_dir = output_dir or paths.reports_dir

    experiment_name = config.get("mlflow", {}).get("experiment_name", PROJECT_NAME)
    setup_mlflow(experiment_name)

    _device = get_device()  # Reserved for future use

    with mlflow.start_run(run_name="robustness_evaluation"):
        mlflow.set_tags(
            {
                "project": PROJECT_NAME,
                "stage": "evaluation",
                "method": "robustness",
                "source_run_id": run_id,
            }
        )

        from projects.vision_ssl_transfer.project.data import (
            DataConfig,
            create_supervised_dataloaders,
        )

        # Load classifier model
        logger.info(f"Loading model from run {run_id}")
        # model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")

        console.print("[red]Not fully implemented: need to load classifier model[/red]")

        # Create test loader
        data_config = DataConfig(
            data_dir=paths.data_dir,
            dataset=config.data.dataset,
            image_size=config.data.image_size,
        )

        _, _, test_loader = create_supervised_dataloaders(
            paths.data_dir,
            data_config,
            batch_size=64,
        )

        # Run robustness evaluation
        _corruptions = OmegaConf.to_container(config.robustness.corruptions)  # type: ignore  # noqa: F841

        # results = run_robustness_evaluation(model, test_loader, _corruptions, _device)

        console.print("[yellow]Robustness evaluation placeholder[/yellow]")


@app.command()
def full(
    ssl_run_id: str = typer.Option(
        ...,
        "--ssl-run-id",
        "-s",
        help="MLflow run ID of pretrained SSL model",
    ),
    config_path: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config YAML",
    ),
) -> None:
    """Run full evaluation suite (k-NN, linear probe, fine-tune)."""
    setup_logging()
    console.print("[blue]Running full evaluation suite...[/blue]")

    # Run k-NN (no training)
    console.print("\n[bold]1. k-NN Evaluation[/bold]")
    knn(ssl_run_id=ssl_run_id, config_path=config_path)

    # TODO: Add linear probe and fine-tune evaluations

    console.print("\n[green]Full evaluation complete![/green]")


if __name__ == "__main__":
    app()
