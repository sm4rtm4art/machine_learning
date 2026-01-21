# ML Portfolio Documentation

Welcome to the ML Portfolio documentation. This guide covers the conventions, standards, and practices used throughout the repository.

## Contents

- [Repository Conventions](repo_conventions.md) — Code organization, notebook policy, CLI contracts
- [Evaluation Standards](evaluation_standards.md) — Required metrics, artifact specs, slicing
- [MLflow Conventions](mlflow_conventions.md) — Experiment naming, tags, artifact organization

## Quick Reference

### Project Structure

Every project follows this structure:

```
projects/<name>/
├── README.md           # Project overview and results
├── configs/            # Hydra/YAML configurations
│   └── default.yaml
├── notebooks/          # Report notebooks only
│   └── 00_report.ipynb
├── scripts/            # CLI entry points
│   ├── download_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export.py
│   └── serve.py
├── project/            # Project-specific code
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── eval.py
└── tests/
    ├── test_smoke.py
    └── test_metrics.py
```

### Standard Commands

```bash
# Any project
uv run python projects/<name>/scripts/download_data.py
uv run python projects/<name>/scripts/train.py --config configs/default.yaml
uv run python projects/<name>/scripts/evaluate.py --run-id <mlflow_run_id>
uv run python projects/<name>/scripts/export.py --format onnx
uv run python projects/<name>/scripts/serve.py --port 8000
```

### Evaluation Outputs

Every `evaluate.py` produces:

```
reports/<project>/<run_id>/
├── metrics.json        # Primary metrics (CI gates)
├── slices.csv          # Per-subgroup performance
├── robustness.csv      # Degradation analysis
└── plots/
    ├── calibration.png
    ├── confusion.png
    └── ...
```

## Getting Help

- Check the project-specific README for dataset and model details
- Review evaluation standards before adding new metrics
- Follow MLflow conventions for experiment tracking
