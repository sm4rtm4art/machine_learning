"""Training logic for SSL pretraining and fine-tuning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml_portfolio.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SSLTrainConfig:
    """SSL pretraining configuration."""

    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    epochs: int = 100
    warmup_epochs: int = 10
    max_grad_norm: float = 1.0


@dataclass
class TransferTrainConfig:
    """Transfer learning / fine-tuning configuration."""

    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50
    early_stopping_patience: int = 5
    max_grad_norm: float = 1.0
    freeze_backbone_epochs: int = 0


@dataclass
class TrainResult:
    """Training result."""

    best_epoch: int
    best_val_metric: float
    train_losses: list[float]
    val_metrics: list[float]
    model_path: Path


def train_ssl_epoch(
    model: nn.Module,
    dataloader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    device: str,
    config: SSLTrainConfig,
) -> float:
    """Train SSL model for one epoch.

    Args:
        model: SSL model (SimCLR, MAE, etc.).
        dataloader: Training dataloader (returns view1, view2).
        optimizer: Optimizer.
        device: Device to use.
        config: Training configuration.

    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc="SSL Training")
    optimizer.zero_grad()

    for batch in progress:
        view1 = batch["view1"].to(device)
        view2 = batch["view2"].to(device)

        # Forward pass
        outputs = model(view1, view2)
        loss = outputs["loss"]

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

        progress.set_postfix({"loss": total_loss / num_batches})

    return total_loss / max(num_batches, 1)


def train_ssl(
    model: nn.Module,
    train_dataloader: DataLoader[Any],
    config: SSLTrainConfig,
    output_dir: Path,
    device: str = "cpu",
    val_dataloader: DataLoader[Any] | None = None,
) -> TrainResult:
    """Full SSL pretraining loop.

    Args:
        model: SSL model.
        train_dataloader: Training dataloader.
        config: Training configuration.
        output_dir: Directory to save checkpoints.
        device: Device to use.
        val_dataloader: Optional validation dataloader.

    Returns:
        Training result.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Learning rate scheduler with warmup
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=config.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs - config.warmup_epochs,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_epochs],
    )

    train_losses: list[float] = []
    val_metrics: list[float] = []
    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch + 1}/{config.epochs}")

        # Train
        train_loss = train_ssl_epoch(model, train_dataloader, optimizer, device, config)
        train_losses.append(train_loss)

        # Validation (optional)
        if val_dataloader is not None:
            val_loss = validate_ssl(model, val_dataloader, device)
            val_metrics.append(val_loss)
            logger.info(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
            metric = val_loss
        else:
            logger.info(f"Train loss: {train_loss:.4f}")
            metric = train_loss

        # Step scheduler
        scheduler.step()

        # Save best model
        if metric < best_loss:
            best_loss = metric
            best_epoch = epoch

            model_path = output_dir / "best_ssl_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": metric,
                },
                model_path,
            )
            logger.info(f"Saved best model to {model_path}")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": metric,
                },
                checkpoint_path,
            )

    return TrainResult(
        best_epoch=best_epoch,
        best_val_metric=best_loss,
        train_losses=train_losses,
        val_metrics=val_metrics,
        model_path=output_dir / "best_ssl_model.pt",
    )


def validate_ssl(
    model: nn.Module,
    dataloader: DataLoader[Any],
    device: str,
) -> float:
    """Validate SSL model.

    Args:
        model: SSL model.
        dataloader: Validation dataloader.
        device: Device to use.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            view1 = batch["view1"].to(device)
            view2 = batch["view2"].to(device)

            outputs = model(view1, view2)
            loss = outputs["loss"]

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def train_classifier_epoch(
    model: nn.Module,
    dataloader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    config: TransferTrainConfig,
) -> tuple[float, float]:
    """Train classifier for one epoch.

    Args:
        model: Classifier model.
        dataloader: Training dataloader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to use.
        config: Training configuration.

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(dataloader, desc="Training")

    for batch in progress:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix({"loss": total_loss / (total / labels.size(0))})

    accuracy = correct / max(total, 1)
    batch_size = dataloader.batch_size or 1
    avg_loss = total_loss / max(total / batch_size, 1)

    return avg_loss, accuracy


def validate_classifier(
    model: nn.Module,
    dataloader: DataLoader[Any],
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Validate classifier.

    Args:
        model: Classifier model.
        dataloader: Validation dataloader.
        criterion: Loss function.
        device: Device to use.

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / max(total, 1)
    batch_size = dataloader.batch_size or 1
    avg_loss = total_loss / max(total / batch_size, 1)

    return avg_loss, accuracy


def train_linear_probe(
    model: nn.Module,
    train_dataloader: DataLoader[Any],
    val_dataloader: DataLoader[Any],
    config: TransferTrainConfig,
    output_dir: Path,
    device: str = "cpu",
) -> TrainResult:
    """Train linear probe classifier.

    Args:
        model: Linear probe model with frozen backbone.
        train_dataloader: Training dataloader.
        val_dataloader: Validation dataloader.
        config: Training configuration.
        output_dir: Directory to save checkpoints.
        device: Device to use.

    Returns:
        Training result.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
    )
    criterion = nn.CrossEntropyLoss()

    train_losses: list[float] = []
    val_metrics: list[float] = []
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch + 1}/{config.epochs}")

        train_loss, train_acc = train_classifier_epoch(
            model, train_dataloader, optimizer, criterion, device, config
        )
        train_losses.append(train_loss)

        val_loss, val_acc = validate_classifier(model, val_dataloader, criterion, device)
        val_metrics.append(val_acc)

        logger.info(
            f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, "
            f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            model_path = output_dir / "best_linear_probe.pt"
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    return TrainResult(
        best_epoch=best_epoch,
        best_val_metric=best_acc,
        train_losses=train_losses,
        val_metrics=val_metrics,
        model_path=output_dir / "best_linear_probe.pt",
    )
