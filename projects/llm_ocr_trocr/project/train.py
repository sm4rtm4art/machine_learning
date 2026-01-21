"""Training logic for TrOCR fine-tuning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from ml_portfolio.common.logging import get_logger
from ml_portfolio.metrics.ocr import compute_cer

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    """Training configuration."""

    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    epochs: int = 10
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 3
    gradient_accumulation_steps: int = 2
    fp16: bool = True
    max_grad_norm: float = 1.0


@dataclass
class TrainResult:
    """Training result."""

    best_epoch: int
    best_val_cer: float
    train_losses: list[float]
    val_cers: list[float]
    model_path: Path


def train_epoch(
    model: Any,
    dataloader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: str,
    config: TrainConfig,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Train for one epoch.

    Args:
        model: TrOCR model.
        dataloader: Training dataloader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device to use.
        config: Training configuration.
        scaler: Gradient scaler for mixed precision.

    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    num_steps = 0

    progress = tqdm(dataloader, desc="Training")
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # Mixed precision forward pass
        if config.fp16 and scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()
        else:
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            if config.fp16 and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * config.gradient_accumulation_steps
        num_steps += 1

        progress.set_postfix({
            "loss": total_loss / num_steps,
            "lr": scheduler.get_last_lr()[0],
        })

    return total_loss / max(num_steps, 1)


def validate(
    model: Any,
    dataloader: DataLoader[Any],
    processor: Any,
    device: str,
    config: TrainConfig,
) -> tuple[float, float]:
    """Validate model.

    Args:
        model: TrOCR model.
        dataloader: Validation dataloader.
        processor: TrOCR processor for decoding.
        device: Device to use.
        config: Training configuration.

    Returns:
        Tuple of (average loss, average CER).
    """
    model.eval()
    total_loss = 0.0
    all_predictions: list[str] = []
    all_references: list[str] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            texts = batch["text"]

            # Compute loss
            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()

            # Generate predictions
            generated_ids = model.generate(pixel_values, max_length=128)
            predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)

            all_predictions.extend(predictions)
            all_references.extend(texts)

    avg_loss = total_loss / len(dataloader)

    # Compute CER
    cers = [compute_cer(pred, ref) for pred, ref in zip(all_predictions, all_references)]
    avg_cer = sum(cers) / len(cers)

    return avg_loss, avg_cer


def train(
    model: Any,
    processor: Any,
    train_dataloader: DataLoader[Any],
    val_dataloader: DataLoader[Any],
    config: TrainConfig,
    output_dir: Path,
    device: str = "cuda",
) -> TrainResult:
    """Full training loop.

    Args:
        model: TrOCR model.
        processor: TrOCR processor.
        train_dataloader: Training dataloader.
        val_dataloader: Validation dataloader.
        config: Training configuration.
        output_dir: Directory to save checkpoints.
        device: Device to use.

    Returns:
        Training result with best model info.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    num_training_steps = len(train_dataloader) * config.epochs // config.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda") if config.fp16 and device == "cuda" else None

    train_losses: list[float] = []
    val_cers: list[float] = []
    best_val_cer = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch + 1}/{config.epochs}")

        # Train
        train_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler, device, config, scaler
        )
        train_losses.append(train_loss)

        # Validate
        val_loss, val_cer = validate(model, val_dataloader, processor, device, config)
        val_cers.append(val_cer)

        logger.info(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val CER: {val_cer:.4f}")

        # Early stopping check
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            model_path = output_dir / "best_model"
            model.save_pretrained(model_path)
            processor.save_pretrained(model_path)
            logger.info(f"Saved best model to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    return TrainResult(
        best_epoch=best_epoch,
        best_val_cer=best_val_cer,
        train_losses=train_losses,
        val_cers=val_cers,
        model_path=output_dir / "best_model",
    )
