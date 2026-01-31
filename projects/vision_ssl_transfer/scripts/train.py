#!/usr/bin/env python
"""Train SSL model with MLflow tracking."""

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
def ssl(
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
    algorithm: str = typer.Option(
        None,
        "--algorithm",
        "-a",
        help="SSL algorithm: simclr, mae (overrides config)",
    ),
    epochs: int = typer.Option(
        None,
        "--epochs",
        "-e",
        help="Number of epochs (overrides config)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config without training",
    ),
) -> None:
    """Train SSL model (SimCLR, MAE, etc.)."""
    setup_logging()

    paths = get_project_paths(PROJECT_NAME)
    config_path = config_path or paths.default_config()

    # Load config
    logger.info(f"Loading config from {config_path}")
    config = OmegaConf.load(config_path)

    # Override from CLI
    if algorithm:
        config.ssl.algorithm = algorithm
    if epochs:
        config.ssl.epochs = epochs

    if dry_run:
        console.print("[green]Config validation successful[/green]")
        console.print(OmegaConf.to_yaml(config))
        return

    # Setup MLflow
    experiment_name = config.get("mlflow", {}).get("experiment_name", PROJECT_NAME)
    setup_mlflow(experiment_name)

    device = get_device()
    logger.info(f"Using device: {device}")

    with mlflow.start_run(run_name=run_name or f"ssl_{config.ssl.algorithm}"):
        log_config(config)
        log_reproducibility_info(config)

        mlflow.set_tags(
            {
                "project": PROJECT_NAME,
                "stage": "pretraining",
                "algorithm": config.ssl.algorithm,
            }
        )

        # Import here to avoid slow startup
        from projects.vision_ssl_transfer.project.data import (
            DataConfig,
            create_ssl_dataloaders,
        )
        from projects.vision_ssl_transfer.project.model import (
            BackboneConfig,
            create_backbone,
        )
        from projects.vision_ssl_transfer.project.ssl import create_ssl_model
        from projects.vision_ssl_transfer.project.train import (
            SSLTrainConfig,
            train_ssl,
        )

        # Create data loaders
        data_config = DataConfig(
            data_dir=paths.data_dir,
            dataset=config.data.dataset,
            image_size=config.data.image_size,
            augmentation=OmegaConf.to_container(config.data.augmentation),  # type: ignore
        )

        train_loader, val_loader = create_ssl_dataloaders(
            paths.data_dir,
            data_config,
            batch_size=config.ssl.batch_size,
        )

        logger.info(f"Train samples: {len(train_loader.dataset)}")  # type: ignore

        # Create model
        backbone_config = BackboneConfig(
            name=config.ssl.backbone,
            pretrained=False,
        )
        backbone = create_backbone(backbone_config, device=device)

        ssl_model = create_ssl_model(
            backbone,
            config.ssl.algorithm,
            OmegaConf.to_container(config.ssl),  # type: ignore
        )

        # Train
        train_config = SSLTrainConfig(
            batch_size=config.ssl.batch_size,
            learning_rate=config.ssl.learning_rate,
            weight_decay=config.ssl.weight_decay,
            epochs=config.ssl.epochs,
            warmup_epochs=config.ssl.warmup_epochs,
        )

        result = train_ssl(
            ssl_model,
            train_loader,
            train_config,
            paths.artifacts_dir,
            device=device,
            val_dataloader=val_loader,
        )

        # Log metrics
        mlflow.log_metrics(
            {
                "best_epoch": result.best_epoch,
                "best_loss": result.best_val_metric,
            }
        )

        # Log model
        mlflow.log_artifact(str(result.model_path))

        console.print("[green]SSL training complete![/green]")
        console.print(f"Best epoch: {result.best_epoch}, Best loss: {result.best_val_metric:.4f}")


@app.command()
def linear_probe(
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
    run_name: str = typer.Option(
        None,
        "--run-name",
        "-n",
        help="MLflow run name",
    ),
) -> None:
    """Train linear probe on frozen SSL features."""
    setup_logging()

    paths = get_project_paths(PROJECT_NAME)
    config_path = config_path or paths.default_config()
    config = OmegaConf.load(config_path)

    experiment_name = config.get("mlflow", {}).get("experiment_name", PROJECT_NAME)
    setup_mlflow(experiment_name)

    device = get_device()
    logger.info(f"Using device: {device}")

    with mlflow.start_run(run_name=run_name or "linear_probe"):
        log_config(config)

        mlflow.set_tags(
            {
                "project": PROJECT_NAME,
                "stage": "evaluation",
                "method": "linear_probe",
                "ssl_run_id": ssl_run_id,
            }
        )

        # Load SSL model
        logger.info(f"Loading SSL model from run {ssl_run_id}")

        # Import here to avoid slow startup
        from projects.vision_ssl_transfer.project.data import (
            DataConfig,
            create_supervised_dataloaders,
        )
        from projects.vision_ssl_transfer.project.model import (
            LinearProbeClassifier,
            SSLBackbone,
        )
        from projects.vision_ssl_transfer.project.train import (
            TransferTrainConfig,
            train_linear_probe,
        )

        # Load pretrained backbone
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=ssl_run_id, artifact_path="best_ssl_model.pt"
        )
        backbone = SSLBackbone.from_pretrained(Path(artifact_path))

        # Create classifier
        num_classes = 2  # Binary for Oxford Pets (cat vs dog)
        model = LinearProbeClassifier(backbone, num_classes, freeze_backbone=True)

        # Create data loaders
        data_config = DataConfig(
            data_dir=paths.data_dir,
            dataset=config.data.dataset,
            image_size=config.data.image_size,
        )

        train_loader, val_loader, test_loader = create_supervised_dataloaders(
            paths.data_dir,
            data_config,
            batch_size=config.transfer.linear_probe.batch_size,
        )

        # Train
        train_config = TransferTrainConfig(
            batch_size=config.transfer.linear_probe.batch_size,
            learning_rate=config.transfer.linear_probe.learning_rate,
            epochs=config.transfer.linear_probe.epochs,
        )

        result = train_linear_probe(
            model,
            train_loader,
            val_loader,
            train_config,
            paths.artifacts_dir / "linear_probe",
            device=device,
        )

        # Log metrics
        mlflow.log_metrics(
            {
                "best_epoch": result.best_epoch,
                "best_val_accuracy": result.best_val_metric,
            }
        )

        console.print("[green]Linear probe training complete![/green]")
        console.print(f"Best val accuracy: {result.best_val_metric:.4f}")


if __name__ == "__main__":
    app()
