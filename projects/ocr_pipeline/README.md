# OCR Pipeline

**Status**: 🚧 Active Development

A production-ready OCR pipeline using Microsoft's TrOCR model, evaluated on the SROIE receipts dataset.

## Overview

This project demonstrates:
- End-to-end OCR pipeline (preprocessing → recognition → postprocessing)
- Rigorous evaluation with CER, WER, and field-level metrics
- Robustness testing (blur, rotation, compression)
- ONNX export and INT8 quantization for deployment
- Speed/accuracy tradeoff analysis

## Dataset: SROIE

The [SROIE dataset](https://rrc.cvc.uab.es/?ch=13) (Scanned Receipts OCR and Information Extraction) contains:
- 626 training receipts, 347 test receipts
- Four structured fields: company, date, address, total
- Real-world receipt images with varying quality

## Pipeline Architecture

```
Raw Image
    ↓
┌─────────────────────────────────────┐
│         Preprocessing               │
│  • Resize & normalize               │
│  • Deskew (optional)                │
│  • Text region detection            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│         TrOCR Recognition           │
│  • Vision encoder (ViT/DeiT)        │
│  • Text decoder (RoBERTa)           │
│  • Beam search decoding             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│         Postprocessing              │
│  • Text normalization               │
│  • Field extraction rules           │
│  • Confidence scoring               │
└─────────────────────────────────────┘
    ↓
Structured Output
```

## Quick Start

```bash
# Download SROIE dataset
uv run python projects/ocr_pipeline/scripts/download_data.py

# Train (fine-tune TrOCR on SROIE)
uv run python projects/ocr_pipeline/scripts/train.py

# Evaluate
uv run python projects/ocr_pipeline/scripts/evaluate.py --run-id <run_id>

# Export to ONNX
uv run python projects/ocr_pipeline/scripts/export.py --run-id <run_id>
```

## Evaluation Metrics

### Primary Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| CER | Character Error Rate | < 5% |
| WER | Word Error Rate | < 10% |
| Field Exact Match | Per-field accuracy | > 85% |

### Slice Dimensions

- **Field type**: company, date, address, total
- **Image quality**: high, medium, low (based on blur/noise)
- **Text length**: short, medium, long

### Robustness Tests

| Perturbation | Intensities |
|--------------|-------------|
| Gaussian blur | σ = 1.0, 2.0, 3.0 |
| Rotation | ±5°, ±10°, ±15° |
| JPEG compression | quality = 75, 50, 25 |
| Brightness | ±20%, ±40% |

## Model Variants

| Variant | Size | Speed | CER |
|---------|------|-------|-----|
| trocr-base-printed | 334M | 1.0x | Baseline |
| trocr-base-printed (ONNX) | 334M | 1.5x | Same |
| trocr-base-printed (INT8) | 85M | 2.5x | +0.5% |

## Getting Started (Notebooks)

Start with the interactive notebook to understand TrOCR:

```bash
# Open Jupyter
uv run jupyter lab projects/ocr_pipeline/notebooks/01_trocr_experiments.ipynb
```

The notebook includes:
- Model selection dropdown (Databricks-style widgets!)
- Synthetic test image creation (golden set approach)
- Interactive robustness testing (blur, rotation, JPEG)
- MLflow experiment tracking

## Project Structure

```
projects/ocr_pipeline/
├── configs/
│   └── default.yaml          # Training configuration
├── notebooks/
│   └── 01_trocr_experiments.ipynb  # ⭐ Start here!
├── scripts/
│   ├── download_data.py      # Download SROIE dataset
│   ├── train.py              # Fine-tune TrOCR
│   ├── evaluate.py           # Full evaluation
│   ├── export.py             # ONNX export
│   └── serve.py              # FastAPI server
├── project/
│   ├── data.py               # SROIE dataset wrapper
│   ├── model.py              # TrOCR wrapper
│   ├── preprocess.py         # Image preprocessing
│   ├── postprocess.py        # Text postprocessing
│   ├── train.py              # Training loop
│   └── eval.py               # Evaluation logic
└── tests/
    └── test_trocr_smoke.py   # Basic tests
```

## Key Design Decisions

<details>
<summary><strong>Why TrOCR?</strong></summary>

TrOCR combines a vision encoder (ViT) with a text decoder (RoBERTa) in an encoder-decoder architecture. This is state-of-the-art for OCR tasks and:

- Works well on printed and handwritten text
- Provides confidence scores out of the box
- Can be fine-tuned on domain-specific data
- Has good ONNX export support

</details>

<details>
<summary><strong>Why SROIE?</strong></summary>

SROIE is ideal for demonstrating production OCR because:

- It's a real business use case (receipt processing)
- It has structured fields enabling field-level evaluation
- Images have realistic quality variations
- It's publicly available and well-documented

</details>

<details>
<summary><strong>Why evaluate robustness?</strong></summary>

Production images differ from training data. By testing degradation under blur, rotation, and compression, we:

- Understand failure modes before deployment
- Set realistic expectations for different input quality
- Identify preprocessing steps that could help

</details>

## Results

*Results will be added after training runs are complete.*

## References

- [TrOCR Paper](https://arxiv.org/abs/2109.10282)
- [SROIE Dataset](https://rrc.cvc.uab.es/?ch=13)
- [Hugging Face TrOCR](https://huggingface.co/microsoft/trocr-base-printed)
