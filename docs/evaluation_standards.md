# Evaluation Standards

This document defines the evaluation requirements for all projects in the ML Portfolio.

## Philosophy

> A model without proper evaluation is just a random number generator with good PR.

Every model must be evaluated beyond aggregate metrics. We measure:

1. **Primary metrics**: The headline numbers
2. **Slice performance**: How does it work for different subgroups?
3. **Calibration**: Are confidence scores meaningful?
4. **Robustness**: How does it degrade under stress?

## Required Metrics by Problem Type

### Classification (Binary/Multiclass)

| Metric | Required | Purpose |
|--------|----------|---------|
| ROC-AUC | ✓ | Threshold-independent discrimination |
| PR-AUC | ✓ | Performance under class imbalance |
| F1 (tuned threshold) | ✓ | Balanced precision/recall |
| Accuracy | Optional | Only if classes are balanced |
| ECE (Expected Calibration Error) | ✓ | Calibration quality |
| Brier Score | ✓ | Probabilistic accuracy |

**Additional for binary classification:**
- Decision curve analysis
- Threshold selection rationale

### Regression

| Metric | Required | Purpose |
|--------|----------|---------|
| RMSE | ✓ | Standard error measure |
| MAE | ✓ | Robust to outliers |
| R² | Optional | Explained variance |
| Pinball Loss | If probabilistic | Quantile accuracy |
| Interval Coverage | If probabilistic | Calibration of intervals |

### OCR / Text Recognition

| Metric | Required | Purpose |
|--------|----------|---------|
| CER (Character Error Rate) | ✓ | Character-level accuracy |
| WER (Word Error Rate) | ✓ | Word-level accuracy |
| Field Exact Match | If structured | Extraction accuracy |
| Confidence Correlation | ✓ | Is confidence meaningful? |

### Time Series Forecasting

| Metric | Required | Purpose |
|--------|----------|---------|
| RMSE/MAE | ✓ | Point forecast accuracy |
| MASE | ✓ | Scale-independent accuracy |
| Pinball Loss | ✓ | Probabilistic calibration |
| Coverage (50%, 90%) | ✓ | Interval calibration |
| CRPS | Optional | Full distribution accuracy |

### Ranking / Recommendation

| Metric | Required | Purpose |
|--------|----------|---------|
| NDCG@k | ✓ | Ranking quality |
| MAP@k | ✓ | Precision at ranks |
| MRR | ✓ | First relevant position |
| Hit Rate@k | ✓ | Retrieval success |

## Slice-Based Evaluation

### What is Slicing?

Aggregate metrics hide failures. Slicing breaks down performance by meaningful subgroups:

```python
# Example slices for OCR
slices = {
    "image_quality": ["high", "medium", "low"],
    "text_length": ["short", "medium", "long"],
    "field_type": ["company", "date", "total", "address"],
    "has_noise": [True, False],
}
```

### Required Slices

Implemented projects should define slices relevant to their domain:

| Domain | Required Slice Dimensions |
|--------|--------------------------|
| Tabular | Feature missingness, category frequency, outlier bands |
| Vision | Image quality, object size, lighting conditions |
| NLP | Text length, language complexity, domain |
| Time Series | Seasonality, trend strength, volatility |

### Slice Output Format

```csv
slice_name,slice_value,metric_name,metric_value,sample_count
image_quality,high,cer,0.023,1500
image_quality,medium,cer,0.045,800
image_quality,low,cer,0.112,200
field_type,company,exact_match,0.92,500
field_type,date,exact_match,0.98,500
```

## Robustness Testing

### Perturbation Types

| Domain | Perturbations |
|--------|---------------|
| Vision | Blur, noise, rotation, compression, occlusion |
| Tabular | Missing values, noise injection, outliers |
| NLP | Typos, truncation, paraphrasing |
| Time Series | Missing timestamps, noise, trend breaks |

### Robustness Output Format

```csv
perturbation,intensity,metric_name,baseline_value,perturbed_value,degradation_pct
gaussian_blur,sigma=1.0,cer,0.023,0.031,34.8
gaussian_blur,sigma=2.0,cer,0.023,0.058,152.2
rotation,degrees=5,cer,0.023,0.029,26.1
jpeg_compression,quality=50,cer,0.023,0.041,78.3
```

## Calibration

### Why Calibration Matters

A model that outputs 90% confidence should be correct 90% of the time. Poor calibration means:
- Users can't trust confidence scores
- Threshold selection is unreliable
- Downstream systems make wrong decisions

### Calibration Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| ECE | Mean absolute calibration error | Lower is better, <0.05 is good |
| MCE | Max calibration error | Worst-case bucket error |
| Brier Score | Mean squared error of probabilities | Lower is better |

### Required Calibration Artifacts

1. **Reliability diagram**: Predicted vs actual probability by bin
2. **Confidence histogram**: Distribution of confidence scores
3. **Calibration table**: Per-bin accuracy and count

## Artifact Specifications

### metrics.json

```json
{
  "run_id": "abc123",
  "timestamp": "2024-01-15T10:30:00Z",
  "dataset": "sroie_test",
  "metrics": {
    "cer": 0.023,
    "wer": 0.089,
    "field_exact_match": 0.91,
    "ece": 0.032
  },
  "thresholds": {
    "confidence_threshold": 0.85,
    "selection_method": "f1_optimization"
  }
}
```

### slices.csv

See format above. Required columns:
- `slice_name`: Dimension name
- `slice_value`: Value within dimension
- `metric_name`: Which metric
- `metric_value`: The value
- `sample_count`: Number of samples in slice

### robustness.csv

See format above. Required columns:
- `perturbation`: Type of perturbation
- `intensity`: Strength/parameter of perturbation
- `metric_name`: Which metric
- `baseline_value`: Without perturbation
- `perturbed_value`: With perturbation
- `degradation_pct`: Percentage change

### plots/

Required plots vary by problem type:

| Plot | Classification | Regression | OCR |
|------|---------------|------------|-----|
| Calibration | ✓ | | ✓ |
| Confusion Matrix | ✓ | | |
| ROC Curve | ✓ | | |
| PR Curve | ✓ | | |
| Residuals | | ✓ | |
| Error Distribution | | ✓ | ✓ |
| Robustness Curves | ✓ | ✓ | ✓ |

## CI/CD Integration

### Regression Gates

Projects with CI regression gates should define thresholds in `configs/eval_gates.yaml`:

```yaml
gates:
  cer:
    max: 0.05
    comparison: less_than
  field_exact_match:
    min: 0.85
    comparison: greater_than
  ece:
    max: 0.10
    comparison: less_than
```

CI fails if:
- Any metric violates its gate
- Required artifacts are missing
- Slice coverage is below minimum

### Nightly Evaluation

Nightly runs re-evaluate models on:
- Current test set (detect data drift)
- New edge cases (expanding test coverage)
- Robustness sweeps (track stability)
