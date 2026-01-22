# Repository Conventions

This document defines the code organization and conventions for the ML Portfolio.

## Code Location Rules

### Where Code Lives

| Code Type | Location | Example |
|-----------|----------|---------|
| Shared utilities | `src/ml_portfolio/` | Metrics, tracking, data utils |
| Project logic | `projects/<name>/project/` | Model definitions, training loops |
| CLI scripts | `projects/<name>/scripts/` | Entry points only |
| Configuration | `projects/<name>/configs/` | Hydra YAML files |
| Tests | `projects/<name>/tests/` | pytest files |

### Principles

1. **Shared code is importable**: Everything in `src/ml_portfolio/` can be imported by any project
2. **Project code is self-contained**: A project's `project/` module contains all project-specific logic
3. **Scripts are thin**: Scripts parse arguments and call functions from `project/`
4. **No logic in notebooks**: Notebooks import from `project/` and visualize results

## Notebook Policy

### Notebooks Are Reports, Not Code

```python
# ✓ Good: Notebook imports and visualizes
from projects.ocr_pipeline.project.eval import load_results
results = load_results(run_id)
plot_calibration(results)

# ✗ Bad: Notebook contains training logic
model = TrOCRModel()
for epoch in range(100):
    ...
```

### Notebook Requirements

1. **One report notebook per project**: `notebooks/00_report.ipynb`
2. **No large outputs committed**: Pre-commit strips outputs automatically
3. **Parameterized execution**: Notebooks should accept run_id as a parameter
4. **Reproducible**: Re-running the notebook with the same run_id produces the same report

### CI Enforcement

The CI pipeline fails if:
- Notebooks contain outputs larger than 100KB
- Logic exists in notebooks that isn't also in `project/` modules
- Notebooks import from locations other than `project/` or `ml_portfolio`

## CLI Contracts

### Standard Entry Points

Every project exposes these scripts:

| Script | Purpose | Required Arguments |
|--------|---------|-------------------|
| `download_data.py` | Fetch and prepare dataset | `--output-dir` |
| `train.py` | Train model, log to MLflow | `--config` |
| `evaluate.py` | Evaluate model, produce reports | `--run-id` or `--model-path` |
| `export.py` | Export model to deployment format | `--run-id`, `--format` |
| `serve.py` | Start inference server | `--model-path`, `--port` |

### Argument Conventions

```python
# All scripts use Typer for CLI
import typer
from pathlib import Path

app = typer.Typer()

@app.command()
def main(
    config: Path = typer.Option(..., help="Path to config YAML"),
    run_name: str = typer.Option(None, help="MLflow run name"),
    dry_run: bool = typer.Option(False, help="Validate without training"),
):
    ...
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Configuration error |
| 3 | Data error |
| 4 | Model error |

## Naming Conventions

### Files and Directories

- **Directories**: `snake_case` (e.g., `ocr_pipeline`)
- **Python files**: `snake_case` (e.g., `download_data.py`)
- **Config files**: `snake_case.yaml` (e.g., `default.yaml`)
- **Classes**: `PascalCase` (e.g., `TrOCRModel`)
- **Functions**: `snake_case` (e.g., `compute_cer`)

### MLflow

- **Experiments**: `<project_name>` (e.g., `ocr_pipeline`)
- **Run names**: `<descriptor>_<date>` (e.g., `baseline_20240115`)
- **Artifact paths**: See [MLflow Conventions](mlflow_conventions.md)

## Import Structure

### Absolute Imports Only

```python
# ✓ Good
from ml_portfolio.metrics.ocr import compute_cer
from projects.ocr_pipeline.project.model import TrOCRWrapper

# ✗ Bad
from ..metrics.ocr import compute_cer
from .model import TrOCRWrapper
```

### Import Order

1. Standard library
2. Third-party packages
3. `ml_portfolio` (shared library)
4. `projects.<name>.project` (project-specific)

Ruff enforces this automatically.

## Configuration

### Hydra/OmegaConf

All projects use Hydra for configuration:

```yaml
# configs/default.yaml
model:
  name: microsoft/trocr-base-printed
  max_length: 128

training:
  batch_size: 16
  learning_rate: 5e-5
  epochs: 10

data:
  train_split: 0.8
  seed: 42
```

### Environment Variables

- Use `.env` files for secrets (never commit)
- Use `pydantic-settings` for loading
- Prefix with `ML_PORTFOLIO_` for custom vars

## Testing

### Test Location

- **Unit tests**: `projects/<name>/tests/test_*.py`
- **Integration tests**: `projects/<name>/tests/test_integration_*.py`
- **Shared library tests**: `src/ml_portfolio/**/test_*.py`

### Required Tests

Every project must have:

1. **Smoke test**: Model loads and runs inference on dummy input
2. **Metrics test**: Metric functions produce expected outputs on known inputs
3. **Config test**: Default config loads without errors

```python
# test_smoke.py
def test_model_loads():
    model = TrOCRWrapper.from_config(default_config)
    assert model is not None

def test_model_inference():
    model = TrOCRWrapper.from_config(default_config)
    output = model.predict(dummy_image)
    assert isinstance(output, str)
```
