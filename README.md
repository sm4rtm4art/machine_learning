# ML Portfolio

[![CI](https://github.com/sm4rtm4art/machine_learning/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sm4rtm4art/machine_learning/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![MLflow convention](https://img.shields.io/badge/MLflow-convention-blue.svg)](https://mlflow.org/)

A work-in-progress ML portfolio focused on Python, machine learning, evaluation discipline, reproducible project structure, and clean engineering habits.

## Early Public Preview / Work in Progress

> This repository is public early so review and collaboration can happen while cleanup and stabilization are still in progress. Implemented work is concentrated in the OCR pipeline and Vision SSL prototype. Other tracks are clearly marked as planned roadmap/design work.

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

**Status key**: 🚧 Active = notebooks/code/prototypes in progress | 📋 Planned phase = roadmap/design work, no completed implementation implied

For a quick review, start with [OCR Pipeline](projects/ocr_pipeline/) and then [Vision SSL Transfer](projects/vision_ssl_transfer/). Synthetic Data and Time Series are priority tracks, but currently remain in planned phase.

### Core ML

| Project | Status | Key Technologies | Planned Interconnections |
|---------|--------|------------------|--------------------------|
| [OCR Pipeline](projects/ocr_pipeline/) | 🚧 Active | Tesseract, TrOCR, SVM routing | *Future*: LLM post-processing |
| [Tabular Boosting Suite](projects/tabular_boosting/) | 📋 Planned phase | LightGBM, XGBoost, CatBoost, SHAP, [TabPFN](https://github.com/PriorLabs/TabPFN) | Future AutoML input |
| [Time Series Forecasting](projects/timeseries_forecasting_covariates/) | 📋 Planned phase | Darts, NeuralProphet, conformal | Standalone priority track |

### Advanced Architectures

| Project | Status | Key Technologies | Planned Interconnections |
|---------|--------|------------------|--------------------------|
| [Vision SSL Transfer](projects/vision_ssl_transfer/) | 🚧 Active | SSL (SimCLR, MAE), SHAP, timm | Shares encoder patterns with OCR |
| [Graph Neural Networks](projects/graph_neural_networks/) | 📋 Planned phase | PyG, DGL, node/graph classification | Future Materials work |
| [LLM Evaluation Harness](projects/llm_eval_harness/) | 📋 Planned phase | lm-eval-harness, custom metrics | Future OCR post-processing evaluation |
| [Quantum Machine Learning](projects/quantum_ml/) | 📋 Planned phase | Qiskit, PennyLane, TFQ, VQC, quantum kernels | Future optimization/materials track |

### Optimization & Meta-Learning

| Project | Status | Key Technologies | Interconnections |
|---------|--------|------------------|------------------|
| [Bayesian Optimization](projects/bayesian_optimization/) | 📋 Planned phase | Optuna, BoTorch, Ax | Future AutoML/RL support |
| AutoML Comparison | 📋 Planned phase | Auto-sklearn, FLAML, H2O | Roadmap item; no project directory yet |

### Scientific & Applied

| Project | Status | Key Technologies | Interconnections |
|---------|--------|------------------|------------------|
| [Scientific ML - Materials](projects/scientific_ml_materials/) | 📋 Planned phase | JAX, equinox, crystal graphs | Future GNN application |
| [RL Operations Simulator](projects/rl_operations_simulator/) | 📋 Planned phase | Gymnasium, Stable-Baselines3 | Future Bayesian tuning use case |
| [Synthetic Data Generation](projects/synthetic_data_tabular/) | 📋 Planned phase | CTGAN, SDV, privacy metrics | Priority planned track |

### Infrastructure

| Project | Status | Key Technologies | Notes |
|---------|--------|------------------|-------|
| Framework Comparison | 📋 Planned phase | PyTorch, TensorFlow, JAX | Roadmap item; no project directory yet |
| [ONNX Export Hub](projects/onnx_export_hub/) | 📋 Planned phase | ONNX, ONNX Runtime, TensorRT | Future deployment optimization |

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

**Note**: Only projects marked "🚧 Active" have implemented scripts or prototypes. Projects marked "📋 Planned phase" contain design documentation or roadmap notes only.

### Start MLflow

```bash
make mlflow
# Open http://localhost:5000
```

### Start Evidently

```bash
make evidently
# Open http://localhost:8000
```

### Prototype Tracking and Monitoring Examples

```bash
# MLflow example (linear+FFT baseline vs Conv2D)
uv run python projects/vision_ssl_transfer/prototypes/mlflow_quickstart_example.py

# Same MLflow example but using ssl_2d_minimal generated samples
USE_SSL2D_SAMPLES=1 uv run python projects/vision_ssl_transfer/prototypes/mlflow_quickstart_example.py

# Optuna + MLflow nested trial runs (requires optuna from tabular extra)
USE_OPTUNA=1 OPTUNA_TRIALS=20 uv run --extra tabular python projects/vision_ssl_transfer/prototypes/mlflow_quickstart_example.py

# Evidently drift + quality report from ssl_2d_minimal generated samples
uv run --extra monitoring python projects/vision_ssl_transfer/prototypes/evidently_quickstart_example.py
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

The target evaluation standard for this portfolio is to go beyond aggregate metrics:

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

Implemented `evaluate.py` scripts should move toward this artifact convention:

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

See [infra/monitoring/evidently/README.md](infra/monitoring/evidently/README.md) for:
- Evidently service usage
- Monitoring use cases

---

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This portfolio was developed with AI assistance (Claude, Gemini, GPT, etc.) while maintaining human oversight on architecture decisions, evaluation methodology, and code quality. The goal is to demonstrate not just ML skills, but the judgment to know when AI suggestions are appropriate and when they need refinement.
