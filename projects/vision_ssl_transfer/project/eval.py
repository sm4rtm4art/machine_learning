"""Evaluation logic for SSL representations and downstream classifiers."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml_portfolio.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    batch_size: int = 64
    metrics: list[str] = field(default_factory=lambda: ["accuracy", "f1", "roc_auc"])
    knn_k: int = 20
    knn_temperature: float = 0.07


@dataclass
class EvalResult:
    """Evaluation result."""

    metrics: dict[str, float]
    predictions: list[Any]
    targets: list[Any]
    embeddings: np.ndarray[Any, Any] | None = None


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader[Any],
    device: str = "cpu",
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Extract embeddings from encoder.

    Args:
        model: Encoder model (backbone or SSL model with .encode method).
        dataloader: Dataloader with images.
        device: Device to use.

    Returns:
        Tuple of (embeddings, labels) as numpy arrays.
    """
    model.eval()
    model = model.to(device)

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            images = batch["image"].to(device)

            # Handle different model interfaces
            embeddings = model.encode(images) if hasattr(model, "encode") else model(images)

            all_embeddings.append(embeddings.cpu().numpy())

            if "label" in batch:
                all_labels.append(batch["label"].numpy())
            elif "is_cat" in batch:
                # Binary classification for Oxford Pets
                all_labels.append(batch["is_cat"].numpy().astype(int))

    embeddings_array = np.concatenate(all_embeddings)
    labels_array = np.concatenate(all_labels) if all_labels else np.array([])

    return embeddings_array, labels_array


def knn_evaluate(
    train_embeddings: np.ndarray[Any, Any],
    train_labels: np.ndarray[Any, Any],
    test_embeddings: np.ndarray[Any, Any],
    test_labels: np.ndarray[Any, Any],
    k: int = 20,
    temperature: float = 0.07,
) -> dict[str, float]:
    """Evaluate using k-Nearest Neighbors (no training required).

    Args:
        train_embeddings: Training set embeddings.
        train_labels: Training set labels.
        test_embeddings: Test set embeddings.
        test_labels: Test set labels.
        k: Number of neighbors.
        temperature: Softmax temperature.

    Returns:
        Dictionary of metrics.
    """
    from sklearn.metrics import accuracy_score, f1_score

    # Normalize embeddings
    train_embeddings = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)

    # Compute similarity matrix
    similarity = np.dot(test_embeddings, train_embeddings.T)

    # Get top-k neighbors
    top_k_indices = np.argsort(similarity, axis=1)[:, -k:]
    top_k_similarities = np.take_along_axis(similarity, top_k_indices, axis=1)

    # Weighted voting with temperature
    weights = np.exp(top_k_similarities / temperature)
    weights = weights / weights.sum(axis=1, keepdims=True)

    # Get neighbor labels and vote
    neighbor_labels = train_labels[top_k_indices]

    # Weighted voting for each class
    unique_classes = np.unique(train_labels)
    class_scores = np.zeros((len(test_embeddings), len(unique_classes)))

    for i, cls in enumerate(unique_classes):
        class_mask = neighbor_labels == cls
        class_scores[:, i] = (weights * class_mask).sum(axis=1)

    predictions = unique_classes[class_scores.argmax(axis=1)]

    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average="weighted")

    return {
        "knn_accuracy": float(accuracy),
        "knn_f1": float(f1),
    }


def linear_probe_evaluate(
    model: nn.Module,
    test_dataloader: DataLoader[Any],
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate linear probe classifier.

    Args:
        model: Linear probe classifier.
        test_dataloader: Test dataloader.
        device: Device to use.

    Returns:
        Dictionary of metrics.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        roc_auc_score,
    )

    model.eval()
    model = model.to(device)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["label"]

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    # ROC-AUC (handle binary vs multiclass)
    if probs.shape[1] == 2:
        roc_auc = roc_auc_score(labels, probs[:, 1])
    else:
        roc_auc = roc_auc_score(labels, probs, multi_class="ovr")

    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }


def evaluate_representation_quality(
    encoder: nn.Module,
    train_dataloader: DataLoader[Any],
    test_dataloader: DataLoader[Any],
    config: EvalConfig,
    device: str = "cpu",
) -> dict[str, Any]:
    """Comprehensive evaluation of representation quality.

    Runs:
    - k-NN evaluation (no training)
    - Embedding statistics

    Args:
        encoder: Encoder model.
        train_dataloader: Training dataloader.
        test_dataloader: Test dataloader.
        config: Evaluation configuration.
        device: Device to use.

    Returns:
        Dictionary with all evaluation results.
    """
    logger.info("Extracting embeddings...")

    train_embeddings, train_labels = extract_embeddings(encoder, train_dataloader, device)
    test_embeddings, test_labels = extract_embeddings(encoder, test_dataloader, device)

    results: dict[str, Any] = {}

    # k-NN evaluation
    logger.info("Running k-NN evaluation...")
    knn_results = knn_evaluate(
        train_embeddings,
        train_labels,
        test_embeddings,
        test_labels,
        k=config.knn_k,
        temperature=config.knn_temperature,
    )
    results.update(knn_results)

    # Embedding statistics
    results["embedding_stats"] = {
        "mean_norm": float(np.linalg.norm(test_embeddings, axis=1).mean()),
        "std_norm": float(np.linalg.norm(test_embeddings, axis=1).std()),
        "embedding_dim": test_embeddings.shape[1],
    }

    return results


def run_robustness_evaluation(
    model: nn.Module,
    dataloader: DataLoader[Any],
    corruptions: list[dict[str, Any]],
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Evaluate model robustness under various corruptions.

    Args:
        model: Classifier model.
        dataloader: Test dataloader.
        corruptions: List of corruption configurations.
        device: Device to use.

    Returns:
        List of results per corruption.
    """
    from sklearn.metrics import accuracy_score

    results = []

    for corruption in corruptions:
        corruption_name = corruption["name"]
        severities = corruption.get("severities", [1])

        for severity in severities:
            logger.info(f"Evaluating {corruption_name} at severity {severity}")

            all_preds = []
            all_labels = []

            model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    images = batch["image"]

                    # Apply corruption
                    corrupted = apply_corruption(images, corruption_name, severity)
                    corrupted = corrupted.to(device)

                    outputs = model(corrupted)
                    preds = outputs.argmax(dim=1)

                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(batch["label"].numpy())

            preds = np.concatenate(all_preds)
            labels = np.concatenate(all_labels)

            accuracy = accuracy_score(labels, preds)

            results.append(
                {
                    "corruption": corruption_name,
                    "severity": severity,
                    "accuracy": float(accuracy),
                }
            )

    return results


def apply_corruption(
    images: torch.Tensor,
    corruption_name: str,
    severity: float,
) -> torch.Tensor:
    """Apply corruption to images.

    Args:
        images: Input images (B, C, H, W).
        corruption_name: Name of corruption.
        severity: Severity level.

    Returns:
        Corrupted images.
    """
    if corruption_name == "gaussian_noise":
        noise = torch.randn_like(images) * (0.1 * severity)
        return torch.clamp(images + noise, 0, 1)

    elif corruption_name == "blur":
        # Simplified blur using average pooling
        kernel_size = int(3 + 2 * severity)
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2
        return torch.nn.functional.avg_pool2d(images, kernel_size, stride=1, padding=padding)

    elif corruption_name == "contrast":
        # Reduce contrast
        factor = 1.0 - (0.2 * severity)
        mean = images.mean(dim=(2, 3), keepdim=True)
        return factor * (images - mean) + mean

    elif corruption_name == "occlusion":
        # Random occlusion
        occluded = images.clone()
        h, w = images.shape[2:]
        occlusion_size = int(h * severity)

        for i in range(images.shape[0]):
            x = torch.randint(0, w - occlusion_size, (1,)).item()
            y = torch.randint(0, h - occlusion_size, (1,)).item()
            occluded[i, :, y : y + occlusion_size, x : x + occlusion_size] = 0

        return occluded

    else:
        logger.warning(f"Unknown corruption: {corruption_name}")
        return images


def save_evaluation_results(
    results: dict[str, Any],
    output_dir: Path,
    run_id: str,
) -> None:
    """Save evaluation results to disk.

    Args:
        results: Evaluation results dictionary.
        output_dir: Output directory.
        run_id: Run identifier.
    """
    report_dir = output_dir / run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics as JSON
    metrics_path = report_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved evaluation results to {metrics_path}")


def compute_calibration_error(
    probs: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute Expected Calibration Error (ECE).

    Args:
        probs: Predicted probabilities (N, num_classes).
        labels: True labels (N,).
        n_bins: Number of confidence bins.

    Returns:
        Dictionary with ECE and other calibration metrics.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == labels

    ece = 0.0
    bin_counts = []

    for i in range(n_bins):
        low = i / n_bins
        high = (i + 1) / n_bins

        mask = (confidences >= low) & (confidences < high)
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_count = mask.sum()

            ece += (bin_count / len(labels)) * abs(bin_acc - bin_conf)
            bin_counts.append(
                {
                    "bin": i,
                    "accuracy": float(bin_acc),
                    "confidence": float(bin_conf),
                    "count": int(bin_count),
                }
            )

    return {
        "ece": float(ece),
        "mean_confidence": float(confidences.mean()),
        "mean_accuracy": float(accuracies.mean()),
        "bins": bin_counts,
    }
