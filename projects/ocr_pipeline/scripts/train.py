#!/usr/bin/env python
"""Train TrOCR model on SROIE dataset."""

from pathlib import Path

import mlflow
import torch
import typer
from omegaconf import OmegaConf
from rich.console import Console
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ml_portfolio.common.logging import get_logger, setup_logging
from ml_portfolio.common.paths import get_project_paths
from ml_portfolio.tracking.mlflow_utils import (
    log_config,
    log_reproducibility_info,
    setup_mlflow,
)

from projects.ocr_pipeline.project.data import SROIEDataset, DataConfig
from projects.ocr_pipeline.project.train import train, TrainConfig

app = typer.Typer()
console = Console()
logger = get_logger(__name__)

PROJECT_NAME = "ocr_pipeline"


def get_device() -> str:
    """Get best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    device: str = typer.Option(
        None,
        "--device",
        "-d",
        help="Device to use (cuda, mps, cpu). Auto-detected if not set.",
    ),
) -> None:
    """Train TrOCR on SROIE dataset."""
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

    # Determine device
    device = device or get_device()
    logger.info(f"Using device: {device}")

    # Load processor and model
    logger.info(f"Loading model: {config.model.name}")
    processor = TrOCRProcessor.from_pretrained(config.model.name)
    model = VisionEncoderDecoderModel.from_pretrained(config.model.name)

    # Set decoder config for generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Create datasets
    logger.info("Creating datasets")
    train_dataset = SROIEDataset(
        paths.data_dir,
        split="train",
        processor=processor,
        max_samples=config.data.get("max_samples"),
    )

    val_dataset = SROIEDataset(
        paths.data_dir,
        split="val",
        processor=processor,
        max_samples=config.data.get("max_samples"),
    )

    if len(train_dataset) == 0:
        console.print("[red]No training data found. Run download_data.py first.[/red]")
        raise typer.Exit(1)

    # Create dataloaders
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Setup MLflow
    experiment_name = config.mlflow.get("experiment_name", PROJECT_NAME)
    setup_mlflow(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Log config and reproducibility info
        log_config(config)
        log_reproducibility_info(config)

        mlflow.set_tags({
            "project": PROJECT_NAME,
            "stage": "development",
            "run_type": "training",
            "model_name": config.model.name,
            "device": device,
        })

        # Log dataset info
        mlflow.log_params({
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        })

        # Create training config
        train_config = TrainConfig(
            batch_size=config.training.batch_size,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.get("weight_decay", 0.01),
            epochs=config.training.epochs,
            warmup_ratio=config.training.get("warmup_ratio", 0.1),
            early_stopping_patience=config.training.early_stopping_patience,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            fp16=config.training.get("fp16", True) and device == "cuda",
            max_grad_norm=config.training.get("max_grad_norm", 1.0),
        )

        # Train
        logger.info("Starting training")
        result = train(
            model=model,
            processor=processor,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            config=train_config,
            output_dir=paths.artifacts_dir,
            device=device,
        )

        # Log final metrics
        mlflow.log_metrics({
            "best_val_cer": result.best_val_cer,
            "best_epoch": result.best_epoch,
        })

        # Log training curves
        for epoch, (loss, cer) in enumerate(zip(result.train_losses, result.val_cers)):
            mlflow.log_metrics({"train_loss": loss, "val_cer": cer}, step=epoch)

        # Log model
        mlflow.log_artifact(str(result.model_path), "model")

        console.print(f"\n[green]Training complete![/green]")
        console.print(f"Best validation CER: {result.best_val_cer:.4f} at epoch {result.best_epoch + 1}")
        console.print(f"Model saved to: {result.model_path}")
        console.print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    app()
