"""Training logic template."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml_portfolio.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    """Training configuration."""

    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 10
    early_stopping_patience: int = 3
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0


@dataclass
class TrainResult:
    """Training result."""

    best_epoch: int
    best_val_loss: float
    train_losses: list[float]
    val_losses: list[float]
    model_path: Path


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    device: str,
    config: TrainConfig,
) -> float:
    """Train for one epoch.

    Args:
        model: Model to train.
        dataloader: Training dataloader.
        optimizer: Optimizer.
        device: Device to use.
        config: Training configuration.

    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc="Training")
    optimizer.zero_grad()

    for batch_idx, _batch in enumerate(progress):
        # Move batch to device
        # batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        # outputs = model(**batch)
        # loss = outputs.loss

        # Placeholder - implement your training logic
        loss = torch.tensor(0.0, device=device)

        # Gradient accumulation
        loss = loss / config.gradient_accumulation_steps
        loss.backward()

        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * config.gradient_accumulation_steps
        num_batches += 1

        progress.set_postfix({"loss": total_loss / num_batches})

    return total_loss / max(num_batches, 1)


def validate(
    model: nn.Module,
    dataloader: DataLoader[Any],
    device: str,
) -> float:
    """Validate model.

    Args:
        model: Model to validate.
        dataloader: Validation dataloader.
        device: Device to use.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for _batch in tqdm(dataloader, desc="Validation"):
            # Move batch to device
            # batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            # outputs = model(**batch)
            # loss = outputs.loss

            # Placeholder - implement your validation logic
            loss = torch.tensor(0.0, device=device)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def train(
    model: nn.Module,
    train_dataloader: DataLoader[Any],
    val_dataloader: DataLoader[Any],
    config: TrainConfig,
    output_dir: Path,
    device: str = "cpu",
) -> TrainResult:
    """Full training loop.

    Args:
        model: Model to train.
        train_dataloader: Training dataloader.
        val_dataloader: Validation dataloader.
        config: Training configuration.
        output_dir: Directory to save checkpoints.
        device: Device to use.

    Returns:
        Training result with best model info.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch + 1}/{config.epochs}")

        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, device, config)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_dataloader, device)
        val_losses.append(val_loss)

        logger.info(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            model_path = output_dir / "best_model.pt"
            # model.save(model_path)
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    return TrainResult(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        train_losses=train_losses,
        val_losses=val_losses,
        model_path=output_dir / "best_model.pt",
    )
