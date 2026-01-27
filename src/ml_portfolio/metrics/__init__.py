"""Evaluation metrics for different ML problem types."""

from ml_portfolio.metrics.classification import (
    compute_calibration_metrics,
    compute_classification_metrics,
)
from ml_portfolio.metrics.ocr import compute_cer, compute_ocr_metrics, compute_wer
from ml_portfolio.metrics.regression import compute_regression_metrics

__all__ = [
    "compute_classification_metrics",
    "compute_calibration_metrics",
    "compute_regression_metrics",
    "compute_cer",
    "compute_wer",
    "compute_ocr_metrics",
]
