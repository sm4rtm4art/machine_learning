# MLflow Conventions

This document defines how we use MLflow for experiment tracking across all projects.

## Setup

### Local Development

```bash
# Start MLflow server
make mlflow

# Access UI at http://localhost:5000
```

### Configuration

Set in `.env`:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=default
```

## Experiment Organization

### Naming Convention

```
Experiment: <project_name>
├── Run: <descriptor>_<YYYYMMDD>
├── Run: <descriptor>_<YYYYMMDD>
└── ...
```

**Examples:**
- Experiment: `llm_ocr_trocr`
  - Run: `baseline_trocr_base_20240115`
  - Run: `finetuned_sroie_20240116`
  - Run: `quantized_int8_20240117`

### Experiment per Project

Each project has exactly one MLflow experiment:

```python
import mlflow

mlflow.set_experiment("llm_ocr_trocr")
```

## Run Structure

### Required Parameters

Every `train.py` must log:

| Parameter | Example | Purpose |
|-----------|---------|---------|
| `model_name` | `microsoft/trocr-base-printed` | Model identifier |
| `model_version` | `v1.0` | Version if applicable |
| `dataset` | `sroie` | Dataset name |
| `dataset_version` | `2019` | Dataset version |
| `batch_size` | `16` | Training batch size |
| `learning_rate` | `5e-5` | Learning rate |
| `epochs` | `10` | Training epochs |
| `seed` | `42` | Random seed |

```python
mlflow.log_params({
    "model_name": config.model.name,
    "dataset": config.data.name,
    "batch_size": config.training.batch_size,
    "learning_rate": config.training.learning_rate,
    "epochs": config.training.epochs,
    "seed": config.seed,
})
```

### Required Metrics

Log metrics at appropriate intervals:

```python
# Per-epoch metrics
for epoch in range(epochs):
    mlflow.log_metrics({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_cer": val_cer,
    }, step=epoch)

# Final metrics
mlflow.log_metrics({
    "test_cer": test_cer,
    "test_wer": test_wer,
    "inference_latency_ms": latency,
})
```

### Required Tags

| Tag | Example | Purpose |
|-----|---------|---------|
| `project` | `llm_ocr_trocr` | Project identifier |
| `stage` | `development` / `staging` / `production` | Lifecycle stage |
| `run_type` | `training` / `evaluation` / `export` | What this run does |
| `environment_hash` | `abc123` | Reproducibility |

```python
mlflow.set_tags({
    "project": "llm_ocr_trocr",
    "stage": "development",
    "run_type": "training",
    "environment_hash": get_env_hash(),
})
```

## Artifact Organization

### Directory Structure

```
artifacts/
├── model/
│   ├── model.pt              # PyTorch weights
│   ├── config.json           # Model config
│   └── tokenizer/            # Tokenizer files
├── onnx/
│   ├── model.onnx            # ONNX export
│   └── model_quantized.onnx  # Quantized version
├── evaluation/
│   ├── metrics.json          # Primary metrics
│   ├── slices.csv            # Slice performance
│   ├── robustness.csv        # Robustness results
│   └── plots/
│       ├── calibration.png
│       ├── confusion.png
│       └── errors.png
├── data/
│   ├── train_manifest.json   # Training data info
│   └── test_manifest.json    # Test data info
└── config/
    └── config.yaml           # Full config used
```

### Logging Artifacts

```python
# Log model
mlflow.pytorch.log_model(model, "model")

# Log ONNX
mlflow.log_artifact("exports/model.onnx", "onnx")

# Log evaluation results
mlflow.log_artifact("reports/metrics.json", "evaluation")
mlflow.log_artifacts("reports/plots/", "evaluation/plots")

# Log config
mlflow.log_artifact("configs/default.yaml", "config")
```

## Model Registry

### Registration

Register models that pass evaluation gates:

```python
# After successful evaluation
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "trocr-sroie")
```

### Versioning

- **None**: Development, not ready for use
- **Staging**: Passed eval gates, ready for testing
- **Production**: Validated, ready for deployment
- **Archived**: Superseded, kept for reference

```python
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="trocr-sroie",
    version=1,
    stage="Staging",
)
```

## Best Practices

### Reproducibility

Always log:

1. **Full config**: The complete configuration used
2. **Git commit**: `mlflow.set_tag("git_commit", get_git_commit())`
3. **Environment hash**: Hash of requirements/dependencies
4. **Data manifest**: What data was used (paths, checksums)

```python
from ml_portfolio.tracking.mlflow_utils import log_reproducibility_info

with mlflow.start_run():
    log_reproducibility_info(config)
    # ... training code
```

### Nested Runs

Use nested runs for hyperparameter searches:

```python
with mlflow.start_run(run_name="hyperparam_search"):
    for params in param_grid:
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            # ... train and evaluate
```

### Comparing Runs

Use tags for easy filtering:

```python
# Find all production-ready models
runs = mlflow.search_runs(
    experiment_names=["llm_ocr_trocr"],
    filter_string="tags.stage = 'production'",
)

# Find best model by metric
runs = mlflow.search_runs(
    experiment_names=["llm_ocr_trocr"],
    order_by=["metrics.test_cer ASC"],
    max_results=1,
)
```

## Integration with Evaluation

### evaluate.py Integration

```python
def evaluate(run_id: str):
    # Load model from MLflow
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")

    # Run evaluation
    results = run_evaluation(model, test_data)

    # Log results back to the same run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(results.metrics)
        mlflow.log_artifact(results.metrics_path, "evaluation")
        mlflow.log_artifacts(results.plots_dir, "evaluation/plots")
```

### CI Integration

```yaml
# .github/workflows/ci.yml
- name: Evaluate Model
  run: |
    uv run python projects/llm_ocr_trocr/scripts/evaluate.py \
      --run-id ${{ env.MLFLOW_RUN_ID }}

- name: Check Gates
  run: |
    uv run python -c "
    import mlflow
    run = mlflow.get_run('${{ env.MLFLOW_RUN_ID }}')
    assert run.data.metrics['test_cer'] < 0.05, 'CER gate failed'
    "
```
