#!/usr/bin/env python
"""Train model with MLflow tracking."""

from pathlib import Path

import mlflow
import typer
from omegaconf import OmegaConf
from rich.console import Console

from ml_portfolio.common.logging import get_logger, setup_logging
from ml_portfolio.common.paths import get_project_paths
from ml_portfolio.tracking.mlflow_utils import (
    log_config,
    log_reproducibility_info,
    setup_mlflow,
)

# from projects._template.project.data import create_data_splits
# from projects._template.project.model import create_model
# from projects._template.project.train import train, TrainConfig

app = typer.Typer()
console = Console()
logger = get_logger(__name__)

PROJECT_NAME = "_template"


@app.command()
def main(
    config_path: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config YAML. Defaults to configs/default.yaml",
    ),
    run_name: str = typer.Option(
        None,
        "--run-name",
        "-n",
        help="MLflow run name",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config without training",
    ),
) -> None:
    """Train the model."""
    setup_logging()

    paths = get_project_paths(PROJECT_NAME)
    config_path = config_path or paths.default_config()

    # Load config
    logger.info(f"Loading config from {config_path}")
    config = OmegaConf.load(config_path)

    if dry_run:
        console.print("[green]Config validation successful[/green]")
        console.print(OmegaConf.to_yaml(config))
        return

    # Setup MLflow
    experiment_name = config.get("mlflow", {}).get("experiment_name", PROJECT_NAME)
    setup_mlflow(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Log configuration and reproducibility info
        log_config(config)
        log_reproducibility_info(config)

        mlflow.set_tags(
            {
                "project": PROJECT_NAME,
                "stage": "development",
                "run_type": "training",
            }
        )

        # Implement training logic
        # 1. Create datasets
        # train_ds, val_ds, test_ds = create_data_splits(paths.data_dir, config.data)

        # 2. Create model
        # model = create_model(config.model, device="cuda")

        # 3. Train
        # result = train(model, train_loader, val_loader, config.training, paths.artifacts_dir)

        # 4. Log metrics
        # mlflow.log_metrics({"val_loss": result.best_val_loss})

        # 5. Log model
        # mlflow.pytorch.log_model(model, "model")

        console.print("[red]Not implemented: Add training logic[/red]")

        logger.info("Training complete")


if __name__ == "__main__":
    app()
