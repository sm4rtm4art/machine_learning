"""Regression metrics including probabilistic metrics."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class RegressionMetrics:
    """Container for regression metrics."""

    rmse: float
    mae: float
    r2: float
    mape: float | None  # Mean Absolute Percentage Error

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "mape": self.mape,
        }


def compute_regression_metrics(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    compute_mape: bool = True,
) -> RegressionMetrics:
    """Compute comprehensive regression metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        compute_mape: Whether to compute MAPE (skip if y_true has zeros).

    Returns:
        RegressionMetrics with all computed values.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    mape = None
    if compute_mape and not np.any(y_true == 0):
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    return RegressionMetrics(rmse=rmse, mae=mae, r2=r2, mape=mape)


def compute_pinball_loss(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    quantile: float,
) -> float:
    """Compute pinball loss for quantile regression.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted quantile values.
        quantile: The quantile being predicted (0 to 1).

    Returns:
        Pinball loss value.
    """
    errors = y_true - y_pred
    loss = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
    return float(np.mean(loss))


def compute_interval_coverage(
    y_true: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> float:
    """Compute prediction interval coverage.

    Args:
        y_true: Ground truth values.
        lower: Lower bound predictions.
        upper: Upper bound predictions.

    Returns:
        Fraction of true values within the interval.
    """
    in_interval = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(in_interval))


def compute_interval_width(
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> float:
    """Compute average prediction interval width.

    Args:
        lower: Lower bound predictions.
        upper: Upper bound predictions.

    Returns:
        Mean interval width.
    """
    return float(np.mean(upper - lower))
