# ML Portfolio

[![CI](https://github.com/sm4rtm4art/machine_learning/actions/workflows/ci.yml/badge.svg)](https://github.com/sm4rtm4art/machine_learning/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tracked with MLflow](https://img.shields.io/badge/tracked%20with-MLflow-blue.svg)](https://mlflow.org/)

A collection of production-quality machine learning projects demonstrating rigorous evaluation practices, reproducible experiments, and clean engineering.

---

## Philosophy

This portfolio is about **learning by building** — making steady progress through small, runnable experiments and tight feedback loops.

Principles I try to follow:
- **Start small, then scale**: get an end-to-end baseline working before adding complexity
- **Understand before hacking**: prefer reading docs, inspecting failures, and writing minimal repros over reverse‑engineering libraries in the dark
- **Make progress legible**: scripts/configs/tests over notebooks, with decisions and results recorded

> The goal isn't to look productive. It's to understand *when* a model works, *why* it fails, and *how* to fix it — without getting stuck in debugging hell.

---

## Projects

**Status key**: 🚧 Active = notebooks/code in progress | 📋 Planned = design docs only

### Core ML (Start Here)

| Project | Status | Key Technologies | Planned Interconnections |
|---------|--------|------------------|--------------------------|
| [OCR Pipeline](projects/ocr_pipeline/) | 🚧 Active | Tesseract, TrOCR, SVM routing | *Future*: LLM post-processing |
| [Tabular Boosting Suite](projects/tabular_boosting/) | 📋 Planned | LightGBM, XGBoost, CatBoost, SHAP | → AutoML |
| [Time Series Forecasting](projects/timeseries_forecasting_covariates/) | 📋 Planned | Darts, NeuralProphet, conformal | Standalone |

### Advanced Architectures

| Project | Status | Key Technologies | Planned Interconnections |
|---------|--------|------------------|--------------------------|
| [Vision SSL Transfer](projects/vision_ssl_transfer/) | 🚧 Active | SSL (SimCLR, MAE), SHAP, timm | Shares encoder patterns with OCR |
| [Graph Neural Networks](projects/graph_neural_networks/) | 📋 Planned | PyG, DGL, node/graph classification | → Materials Discovery |
| [LLM Evaluation Harness](projects/llm_eval_harness/) | 📋 Planned | lm-eval-harness, custom metrics | Benchmarks domain experts |

### Optimization & Meta-Learning

| Project | Status | Key Technologies | Interconnections |
|---------|--------|------------------|------------------|
| [Bayesian Optimization](projects/bayesian_optimization/) | 📋 Planned | Optuna, BoTorch, Ax | → AutoML, RL |
| [AutoML Comparison](projects/automl_comparison/) | 📋 Planned | Auto-sklearn, FLAML, H2O | Uses Boosting, Bayesian |

### Scientific & Applied

| Project | Status | Key Technologies | Interconnections |
|---------|--------|------------------|------------------|
| [Scientific ML - Materials](projects/scientific_ml_materials/) | 📋 Planned | JAX, equinox, crystal graphs | Uses GNN |
| [RL Operations Simulator](projects/rl_operations_simulator/) | 📋 Planned | Gymnasium, Stable-Baselines3 | Uses Bayesian for tuning |
| [Synthetic Data Generation](projects/synthetic_data_tabular/) | 📋 Planned | CTGAN, SDV, privacy metrics | Supports all tabular |

### Infrastructure

| Project | Status | Key Technologies | Notes |
|---------|--------|------------------|-------|
| [Framework Comparison](projects/framework_comparison/) | 📋 Planned | PyTorch, TensorFlow, JAX | Cross-cutting analysis |
| [ONNX Export Hub](projects/onnx_export_hub/) | 📋 Planned | ONNX, ONNX Runtime, TensorRT | Deployment optimization |

---

## Quick Start

### Prerequisites

- Python 3.13+ (deliberate choice for latest typing features; some ML libraries may lag - tested combinations documented per project)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/sm4rtm4art/machine_learning.git
cd machine_learning

# Install with uv (recommended)
uv sync --all-extras

# Or with pip
pip install -e ".[all]"

# Set up pre-commit hooks
make pre-commit-install
```

### Running a Project

Active projects follow a consistent CLI interface (planned projects have READMEs only):

```bash
# Example: OCR Pipeline (🚧 Active)
uv run python projects/ocr_pipeline/scripts/download_data.py
uv run python projects/ocr_pipeline/scripts/train.py
uv run python projects/ocr_pipeline/scripts/evaluate.py
uv run python projects/ocr_pipeline/scripts/export.py
```

**Note**: Only projects marked "🚧 Active" have implemented scripts. Projects marked "📋 Planned" contain design documentation only.

### Start MLflow

```bash
make mlflow
# Open http://localhost:5000
```

---

## Repository Structure

```
machine_learning/
├── src/ml_portfolio/        # Shared library code
│   ├── common/              # Config, logging, paths
│   ├── metrics/             # Evaluation metrics by domain
│   ├── eval/                # Slicing, robustness, drift
│   └── tracking/            # MLflow utilities
│
├── projects/                # Individual ML projects
│   ├── _template/           # Copyable project template
│   └── <project>/
│       ├── configs/         # Hydra/YAML configs
│       ├── notebooks/       # Report notebooks only
│       ├── scripts/         # CLI entry points
│       ├── project/         # Project-specific code
│       └── tests/           # Project tests
│
├── infra/                   # Infrastructure (Docker)
│   ├── mlflow/              # MLflow server
│   └── monitoring/          # Evidently for drift
│
├── docs/                    # Documentation
├── data/                    # Data directory (gitignored)
├── artifacts/               # Model artifacts (gitignored)
└── reports/                 # Generated evaluation reports
```

<details>
<summary><strong>Why this structure?</strong></summary>

**Shared library in `src/`**: Common utilities (metrics, tracking, data loading) are reusable across projects. This avoids copy-paste and ensures consistency.

**Projects as self-contained units**: Each project has its own configs, scripts, and tests. You can understand a project without reading the entire repo.

**Notebooks as reports only**: Notebooks are for visualization and communication, not for logic. All code lives in importable modules. This makes testing possible and diffs readable.

**Consistent CLI per project**: Active projects follow a standard interface (`download_data.py`, `train.py`, `evaluate.py`, `export.py`, `serve.py`). This reduces cognitive load and enables automation. Planned projects will adopt this structure as they're implemented.

</details>

---

## Evaluation Philosophy

Every project in this portfolio adheres to rigorous evaluation standards:

### Beyond Aggregate Metrics

```
✗ "Model achieves 95% accuracy"
✓ "Model achieves 95% accuracy overall, but 72% on edge cases involving X"
```

<details>
<summary><strong>What this means in practice</strong></summary>

**Slice-based evaluation**: Break down performance by meaningful subgroups (data quality, category frequency, edge cases).

**Calibration**: A model that says "90% confident" should be right 90% of the time. We measure this with ECE and reliability diagrams.

**Robustness**: How does performance degrade with noise, missing data, or distribution shift?

**Decision curves**: For classification, accuracy isn't enough. We analyze the tradeoff between false positives and false negatives at different thresholds.

</details>

### Standard Artifacts

Every `evaluate.py` produces:

| Artifact | Purpose |
|----------|---------|
| `metrics.json` | Primary metrics for CI gates |
| `slices.csv` | Performance by subgroup |
| `robustness.csv` | Degradation under perturbations |
| `plots/` | Visualizations (calibration, confusion, etc.) |

---

## Development

### Code Quality

```bash
make lint        # Run ruff linter
make format      # Auto-format code
make typecheck   # Run mypy
make check       # All of the above
```

### Testing

```bash
make test        # Run all tests
make test-cov    # With coverage report
```

### Pre-commit Hooks

Pre-commit hooks enforce:
- Code formatting (ruff)
- Linting (ruff)
- Type checking (mypy) on src/ and project/ modules
- Notebook output stripping
- Large file prevention

---

## Conventions

See [docs/repo_conventions.md](docs/repo_conventions.md) for detailed guidelines on:
- Where code should live
- Notebook policy
- CLI contracts
- Naming conventions

See [docs/evaluation_standards.md](docs/evaluation_standards.md) for:
- Required metrics by problem type
- Artifact specifications
- Slice definitions

See [docs/mlflow_conventions.md](docs/mlflow_conventions.md) for:
- Experiment naming
- Tag schema
- Artifact organization

---

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This portfolio was developed with AI assistance (Claude, Gemini, GPT, etc.) while maintaining human oversight on architecture decisions, evaluation methodology, and code quality. The goal is to demonstrate not just ML skills, but the judgment to know when AI suggestions are appropriate and when they need refinement.
