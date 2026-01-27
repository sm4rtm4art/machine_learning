"""Slice-based evaluation for subgroup analysis."""

import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SliceResult:
    """Result for a single slice."""

    slice_name: str
    slice_value: str
    metrics: dict[str, float]
    sample_count: int


@dataclass
class SliceEvaluator:
    """Evaluate metrics across data slices.

    Example:
        evaluator = SliceEvaluator()
        evaluator.add_slice("image_quality", quality_labels)
        evaluator.add_slice("text_length", length_buckets)

        results = evaluator.evaluate(
            compute_fn=lambda mask: compute_cer(preds[mask], refs[mask]),
            metric_name="cer"
        )
    """

    slices: dict[str, NDArray[Any]] = field(default_factory=dict)
    results: list[SliceResult] = field(default_factory=list)

    def add_slice(
        self,
        name: str,
        values: NDArray[Any] | list[Any],
    ) -> None:
        """Add a slice dimension.

        Args:
            name: Name of the slice dimension (e.g., "image_quality").
            values: Array of slice values per sample (e.g., ["high", "low", ...]).
        """
        self.slices[name] = np.array(values)

    def evaluate(
        self,
        compute_fn: Callable[[NDArray[np.bool_]], dict[str, float]],
        n_samples: int | None = None,
    ) -> list[SliceResult]:
        """Evaluate metrics across all slices.

        Args:
            compute_fn: Function that takes a boolean mask and returns metrics dict.
            n_samples: Total number of samples (inferred from slices if not provided).

        Returns:
            List of SliceResult for each slice value.
        """
        if not self.slices:
            raise ValueError("No slices defined. Call add_slice first.")

        if n_samples is None:
            n_samples = len(next(iter(self.slices.values())))

        self.results = []

        for slice_name, slice_values in self.slices.items():
            unique_values = np.unique(slice_values)

            for value in unique_values:
                mask = slice_values == value
                sample_count = int(mask.sum())

                if sample_count == 0:
                    continue

                metrics = compute_fn(mask)

                result = SliceResult(
                    slice_name=slice_name,
                    slice_value=str(value),
                    metrics=metrics,
                    sample_count=sample_count,
                )
                self.results.append(result)

        return self.results

    def to_csv(self, path: Path) -> None:
        """Save slice results to CSV.

        Args:
            path: Output CSV path.
        """
        if not self.results:
            raise ValueError("No results to save. Call evaluate first.")

        # Get all metric names
        metric_names: set[str] = set()
        for result in self.results:
            metric_names.update(result.metrics.keys())
        metric_names_sorted = sorted(metric_names)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            header = ["slice_name", "slice_value", "sample_count", *metric_names_sorted]
            writer.writerow(header)

            # Data
            for result in self.results:
                row = [
                    result.slice_name,
                    result.slice_value,
                    result.sample_count,
                ]
                for metric in metric_names_sorted:
                    row.append(result.metrics.get(metric, ""))
                writer.writerow(row)

    def summary(self) -> dict[str, dict[str, Any]]:
        """Get summary statistics for each slice dimension.

        Returns:
            Dict mapping slice names to their summary stats.
        """
        summary = {}

        for slice_name in self.slices:
            slice_results = [r for r in self.results if r.slice_name == slice_name]

            if not slice_results:
                continue

            # Get first metric name for summary
            if slice_results[0].metrics:
                first_metric = next(iter(slice_results[0].metrics.keys()))
                values = [r.metrics[first_metric] for r in slice_results]

                summary[slice_name] = {
                    "n_values": len(slice_results),
                    "min": min(values),
                    "max": max(values),
                    "range": max(values) - min(values),
                    "worst_slice": max(
                        slice_results, key=lambda r: r.metrics[first_metric]
                    ).slice_value,
                }

        return summary


def create_length_buckets(
    lengths: NDArray[np.int_] | list[int],
    n_buckets: int = 3,
    labels: list[str] | None = None,
) -> NDArray[np.str_]:
    """Create bucket labels from length values.

    Args:
        lengths: Array of lengths.
        n_buckets: Number of buckets.
        labels: Optional custom labels (e.g., ["short", "medium", "long"]).

    Returns:
        Array of bucket labels.
    """
    lengths = np.array(lengths)

    if labels is None:
        labels = [f"bucket_{i}" for i in range(n_buckets)]

    percentiles = np.linspace(0, 100, n_buckets + 1)
    edges = np.percentile(lengths, percentiles)

    buckets = np.digitize(lengths, edges[1:-1])
    return np.array([labels[b] for b in buckets])
