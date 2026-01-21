"""OCR-specific metrics: CER, WER, and field-level accuracy."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class OCRMetrics:
    """Container for OCR metrics."""

    cer: float  # Character Error Rate
    wer: float  # Word Error Rate
    exact_match: float  # Fraction of exact matches
    mean_confidence: float | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cer": self.cer,
            "wer": self.wer,
            "exact_match": self.exact_match,
            "mean_confidence": self.mean_confidence,
        }


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Minimum number of edits to transform s1 to s2.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_cer(prediction: str, reference: str) -> float:
    """Compute Character Error Rate.

    CER = (insertions + deletions + substitutions) / reference_length

    Args:
        prediction: Predicted text.
        reference: Ground truth text.

    Returns:
        Character error rate (0.0 = perfect, higher = worse).
    """
    if len(reference) == 0:
        return 0.0 if len(prediction) == 0 else 1.0

    distance = levenshtein_distance(prediction, reference)
    return distance / len(reference)


def compute_wer(prediction: str, reference: str) -> float:
    """Compute Word Error Rate.

    WER = (insertions + deletions + substitutions) / reference_word_count

    Args:
        prediction: Predicted text.
        reference: Ground truth text.

    Returns:
        Word error rate (0.0 = perfect, higher = worse).
    """
    pred_words = prediction.split()
    ref_words = reference.split()

    if len(ref_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0

    # Use same Levenshtein but on word lists
    distance = levenshtein_distance_words(pred_words, ref_words)
    return distance / len(ref_words)


def levenshtein_distance_words(s1: list[str], s2: list[str]) -> int:
    """Compute Levenshtein distance on word lists."""
    if len(s1) < len(s2):
        return levenshtein_distance_words(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))

    for i, w1 in enumerate(s1):
        current_row = [i + 1]
        for j, w2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (w1 != w2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_ocr_metrics(
    predictions: list[str],
    references: list[str],
    confidences: list[float] | None = None,
) -> OCRMetrics:
    """Compute comprehensive OCR metrics.

    Args:
        predictions: List of predicted texts.
        references: List of ground truth texts.
        confidences: Optional list of confidence scores.

    Returns:
        OCRMetrics with CER, WER, exact match, and confidence.
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    if len(predictions) == 0:
        raise ValueError("Cannot compute metrics on empty lists")

    # Compute per-sample metrics
    cers = [compute_cer(p, r) for p, r in zip(predictions, references)]
    wers = [compute_wer(p, r) for p, r in zip(predictions, references)]
    exact_matches = [p == r for p, r in zip(predictions, references)]

    return OCRMetrics(
        cer=float(np.mean(cers)),
        wer=float(np.mean(wers)),
        exact_match=float(np.mean(exact_matches)),
        mean_confidence=float(np.mean(confidences)) if confidences else None,
    )


def compute_field_metrics(
    predictions: dict[str, list[str]],
    references: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    """Compute per-field OCR metrics for structured extraction.

    Args:
        predictions: Dict mapping field names to predicted values.
        references: Dict mapping field names to ground truth values.

    Returns:
        Dict mapping field names to their metrics.
    """
    results = {}

    for field_name in references.keys():
        if field_name not in predictions:
            continue

        pred_values = predictions[field_name]
        ref_values = references[field_name]

        if len(pred_values) != len(ref_values):
            continue

        metrics = compute_ocr_metrics(pred_values, ref_values)
        results[field_name] = metrics.to_dict()

    return results


def compute_confidence_correlation(
    predictions: list[str],
    references: list[str],
    confidences: list[float],
) -> float:
    """Compute correlation between confidence and correctness.

    Higher correlation means confidence scores are meaningful.

    Args:
        predictions: Predicted texts.
        references: Ground truth texts.
        confidences: Confidence scores.

    Returns:
        Pearson correlation between confidence and correctness.
    """
    correctness = [1.0 if p == r else 0.0 for p, r in zip(predictions, references)]

    conf_array = np.array(confidences)
    correct_array = np.array(correctness)

    # Handle edge case where all predictions are same correctness
    if np.std(correct_array) == 0:
        return 0.0

    correlation = np.corrcoef(conf_array, correct_array)[0, 1]
    return float(correlation) if not np.isnan(correlation) else 0.0
