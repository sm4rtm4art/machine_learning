"""Evaluation logic for TrOCR OCR pipeline."""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from projects.ocr_pipeline.project.preprocess import (
    apply_perturbation,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml_portfolio.common.logging import get_logger
from ml_portfolio.eval.slicing import SliceEvaluator, create_length_buckets
from ml_portfolio.metrics.ocr import (
    compute_cer,
    compute_confidence_correlation,
    compute_ocr_metrics,
    compute_wer,
)

logger = get_logger(__name__)


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    batch_size: int = 16
    metrics: list[str] = field(default_factory=lambda: ["cer", "wer", "exact_match"])
    slices: list[str] = field(default_factory=lambda: ["field_type", "text_length"])
    robustness: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Evaluation result."""

    metrics: dict[str, float]
    slice_results: list[dict[str, Any]]
    robustness_results: list[dict[str, Any]]
    predictions: list[str]
    references: list[str]
    confidences: list[float]


def evaluate(
    model: Any,
    processor: Any,
    dataloader: DataLoader[Any],
    config: EvalConfig,
    device: str = "cpu",
) -> EvalResult:
    """Evaluate TrOCR model on dataset.

    Args:
        model: TrOCR model.
        processor: TrOCR processor.
        dataloader: Evaluation dataloader.
        config: Evaluation configuration.
        device: Device to use.

    Returns:
        Evaluation results.
    """
    model.eval()
    model = model.to(device)

    all_predictions: list[str] = []
    all_references: list[str] = []
    all_confidences: list[float] = []
    all_text_lengths: list[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            texts = batch["text"]

            # Generate predictions
            outputs = model.generate(
                pixel_values,
                max_length=128,
                num_beams=4,
                return_dict_in_generate=True,
                output_scores=True,
            )

            # Decode predictions
            predictions = processor.batch_decode(outputs.sequences, skip_special_tokens=True)

            # Compute confidence scores (average token probability)
            if hasattr(outputs, "scores") and outputs.scores:
                scores = torch.stack(outputs.scores, dim=1)
                probs = torch.softmax(scores, dim=-1)
                for i, seq in enumerate(outputs.sequences):
                    token_probs = probs[i].gather(1, seq[1:].unsqueeze(-1)).squeeze(-1)
                    confidence = float(token_probs.mean())
                    all_confidences.append(confidence)
            else:
                all_confidences.extend([1.0] * len(predictions))

            all_predictions.extend(predictions)
            all_references.extend(texts)
            all_text_lengths.extend([len(t) for t in texts])

    # Compute primary metrics
    ocr_metrics = compute_ocr_metrics(all_predictions, all_references, all_confidences)
    metrics = ocr_metrics.to_dict()

    # Add confidence correlation
    if all_confidences:
        metrics["confidence_correlation"] = compute_confidence_correlation(
            all_predictions, all_references, all_confidences
        )

    # Compute slice metrics
    slice_results = compute_slice_metrics(
        all_predictions, all_references, all_text_lengths, config.slices
    )

    # Robustness results would be computed separately with perturbed images
    robustness_results: list[dict[str, Any]] = []

    return EvalResult(
        metrics=metrics,
        slice_results=slice_results,
        robustness_results=robustness_results,
        predictions=all_predictions,
        references=all_references,
        confidences=all_confidences,
    )


def compute_slice_metrics(
    predictions: list[str],
    references: list[str],
    text_lengths: list[int],
    slice_names: list[str],
) -> list[dict[str, Any]]:
    """Compute metrics for each slice.

    Args:
        predictions: Predicted texts.
        references: Ground truth texts.
        text_lengths: Length of each text.
        slice_names: Names of slices to compute.

    Returns:
        List of slice results.
    """
    import numpy as np

    evaluator = SliceEvaluator()

    # Add text length slice
    if "text_length" in slice_names:
        length_buckets = create_length_buckets(
            text_lengths, n_buckets=3, labels=["short", "medium", "long"]
        )
        evaluator.add_slice("text_length", length_buckets)

    # Compute slice metrics
    def compute_fn(mask: np.ndarray) -> dict[str, float]:
        preds = [predictions[i] for i in range(len(predictions)) if mask[i]]
        refs = [references[i] for i in range(len(references)) if mask[i]]

        if not preds:
            return {"cer": 0.0, "wer": 0.0}

        cers = [compute_cer(p, r) for p, r in zip(preds, refs)]
        wers = [compute_wer(p, r) for p, r in zip(preds, refs)]

        return {
            "cer": float(np.mean(cers)),
            "wer": float(np.mean(wers)),
        }

    results = evaluator.evaluate(compute_fn, n_samples=len(predictions))

    return [
        {
            "slice_name": r.slice_name,
            "slice_value": r.slice_value,
            "sample_count": r.sample_count,
            **r.metrics,
        }
        for r in results
    ]


def evaluate_robustness(
    model: Any,
    processor: Any,
    dataset: Any,
    config: EvalConfig,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Evaluate model robustness under perturbations.

    Args:
        model: TrOCR model.
        processor: TrOCR processor.
        dataset: Dataset with images.
        config: Evaluation configuration.
        device: Device to use.

    Returns:
        List of robustness results.
    """
    model.eval()
    model = model.to(device)

    results = []

    # First, compute baseline metrics
    baseline_predictions = []
    baseline_references = []

    with torch.no_grad():
        for sample in tqdm(dataset, desc="Baseline evaluation"):
            image = Image.open(sample["image_path"]).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

            generated_ids = model.generate(pixel_values, max_length=128)
            prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            baseline_predictions.append(prediction)
            baseline_references.append(sample["text"])

    baseline_cer = sum(
        compute_cer(p, r) for p, r in zip(baseline_predictions, baseline_references)
    ) / len(baseline_predictions)

    # Evaluate each perturbation
    for perturbation_name, intensities in config.robustness.items():
        for intensity in intensities:
            perturbed_predictions = []

            with torch.no_grad():
                for sample in tqdm(dataset, desc=f"{perturbation_name}={intensity}"):
                    image = Image.open(sample["image_path"]).convert("RGB")

                    # Apply perturbation
                    perturbed_image = apply_perturbation(image, perturbation_name, intensity)

                    pixel_values = processor(
                        images=perturbed_image, return_tensors="pt"
                    ).pixel_values.to(device)

                    generated_ids = model.generate(pixel_values, max_length=128)
                    prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                    perturbed_predictions.append(prediction)

            perturbed_cer = sum(
                compute_cer(p, r) for p, r in zip(perturbed_predictions, baseline_references)
            ) / len(perturbed_predictions)

            degradation = (
                ((perturbed_cer - baseline_cer) / baseline_cer * 100) if baseline_cer > 0 else 0
            )

            results.append(
                {
                    "perturbation": perturbation_name,
                    "intensity": intensity,
                    "metric_name": "cer",
                    "baseline_value": baseline_cer,
                    "perturbed_value": perturbed_cer,
                    "degradation_pct": degradation,
                }
            )

    return results


def save_results(
    result: EvalResult,
    output_dir: Path,
    run_id: str,
) -> None:
    """Save evaluation results to files.

    Args:
        result: Evaluation results.
        output_dir: Output directory.
        run_id: MLflow run ID.
    """
    report_dir = output_dir / run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    # Save primary metrics
    metrics_path = report_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "metrics": result.metrics,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved metrics to {metrics_path}")

    # Save slice results
    if result.slice_results:
        slices_path = report_dir / "slices.csv"
        with open(slices_path, "w", newline="") as f:
            if result.slice_results:
                fieldnames = result.slice_results[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(result.slice_results)
        logger.info(f"Saved slices to {slices_path}")

    # Save robustness results
    if result.robustness_results:
        robustness_path = report_dir / "robustness.csv"
        with open(robustness_path, "w", newline="") as f:
            fieldnames = result.robustness_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(result.robustness_results)
        logger.info(f"Saved robustness to {robustness_path}")

    # Create plots directory
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    logger.info(f"Results saved to {report_dir}")
