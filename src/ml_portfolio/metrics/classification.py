"""Classification metrics including calibration."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    brier_score: float | None
    ece: float | None  # Expected Calibration Error
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "brier_score": self.brier_score,
            "ece": self.ece,
            "threshold": self.threshold,
        }


def compute_classification_metrics(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    y_prob: NDArray[np.float64] | None = None,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels (0 or 1 for binary).
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for positive class (optional).
        threshold: Classification threshold used.

    Returns:
        ClassificationMetrics with all computed values.
    """
    metrics = ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=None,
        pr_auc=None,
        brier_score=None,
        ece=None,
        threshold=threshold,
    )

    if y_prob is not None:
        # Only compute if we have both classes
        if len(np.unique(y_true)) > 1:
            metrics.roc_auc = float(roc_auc_score(y_true, y_prob))
            metrics.pr_auc = float(average_precision_score(y_true, y_prob))

        metrics.brier_score = float(brier_score_loss(y_true, y_prob))

        # Compute calibration
        calibration = compute_calibration_metrics(y_true, y_prob)
        metrics.ece = calibration["ece"]

    return metrics


def compute_calibration_metrics(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration metrics.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_prob: Predicted probabilities.
        n_bins: Number of bins for calibration.

    Returns:
        Dictionary with ECE, MCE, and per-bin statistics.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    ece = 0.0
    mce = 0.0
    bins_data = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue

        bin_prob = y_prob[mask]
        bin_true = y_true[mask]

        avg_confidence = float(bin_prob.mean())
        avg_accuracy = float(bin_true.mean())
        bin_size = int(mask.sum())

        calibration_error = abs(avg_accuracy - avg_confidence)
        ece += (bin_size / len(y_true)) * calibration_error
        mce = max(mce, calibration_error)

        bins_data.append(
            {
                "bin": i,
                "bin_start": float(bin_edges[i]),
                "bin_end": float(bin_edges[i + 1]),
                "avg_confidence": avg_confidence,
                "avg_accuracy": avg_accuracy,
                "count": bin_size,
                "calibration_error": calibration_error,
            }
        )

    return {
        "ece": float(ece),
        "mce": float(mce),
        "n_bins": n_bins,
        "bins": bins_data,
    }


def find_optimal_threshold(
    y_true: NDArray[np.int_],
    y_prob: NDArray[np.float64],
    metric: str = "f1",
) -> float:
    """Find optimal classification threshold.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        metric: Metric to optimize ('f1', 'accuracy', 'balanced_accuracy').

    Returns:
        Optimal threshold value.
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_score = 0.0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "accuracy":
            score = accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return float(best_threshold)
