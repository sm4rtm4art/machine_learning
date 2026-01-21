# ML Portfolio

[![CI](https://github.com/martin/ml-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/martin/ml-portfolio/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tracked with MLflow](https://img.shields.io/badge/tracked%20with-MLflow-blue.svg)](https://mlflow.org/)

A collection of production-quality machine learning projects demonstrating rigorous evaluation practices, reproducible experiments, and clean engineering.

---

## Philosophy

This portfolio represents my approach to applied machine learning: **Human in the Loop + AI Assistance**.

I leverage modern AI tools to accelerate development while maintaining critical oversight on:
- **Evaluation rigor** — Every model is measured against meaningful metrics, not just accuracy
- **Reproducibility** — Every experiment is tracked, versioned, and reproducible
- **Production readiness** — Code is structured for deployment, not just notebooks

> The goal isn't to show I can train models. It's to demonstrate I understand *when* a model works, *why* it fails, and *how* to ship it.

---

## Projects

| Project | Domain | Status | Highlights |
|---------|--------|--------|------------|
| [TrOCR OCR Pipeline](projects/llm_ocr_trocr/) | Vision + NLP | 🚧 Active | SROIE receipts, ONNX export, CER/WER metrics |
| [Tabular Boosting Suite](projects/tabular_boosting/) | Tabular | 📋 Planned | LightGBM/XGBoost/CatBoost comparison, SHAP |
| [LLM Evaluation Harness](projects/llm_eval_harness/) | NLP | 📋 Planned | Reusable eval framework for RAG/agents |
| [Synthetic Data Generation](projects/synthetic_data_tabular/) | Tabular | 📋 Planned | CTGAN, privacy/utility tradeoffs |
| [Vision SSL Transfer](projects/vision_ssl_transfer/) | Vision | 📋 Planned | Self-supervised pretraining, robustness |
| [Time Series Forecasting](projects/timeseries_forecasting_covariates/) | Time Series | 📋 Planned | Covariates, backtesting, probabilistic |
| [RL Operations Simulator](projects/rl_operations_simulator/) | RL | 📋 Planned | Discrete-event sim, DQN/PPO |
| [Multimodal Fusion](projects/multimodal_fusion/) | Multimodal | 📋 Planned | Text + tabular + time fusion |

---

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/martin/ml-portfolio.git
cd ml-portfolio

# Install with uv (recommended)
uv sync --all-extras

# Or with pip
pip install -e ".[all]"

# Set up pre-commit hooks
make pre-commit-install
```

### Running a Project

Each project follows a consistent CLI interface:

```bash
# Download data
uv run python projects/llm_ocr_trocr/scripts/download_data.py

# Train a model
uv run python projects/llm_ocr_trocr/scripts/train.py

# Evaluate
uv run python projects/llm_ocr_trocr/scripts/evaluate.py

# Export to ONNX
uv run python projects/llm_ocr_trocr/scripts/export.py
```

### Start MLflow

```bash
make mlflow
# Open http://localhost:5000
```

---

## Repository Structure

```
ml-portfolio/
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

**Consistent CLI per project**: Every project exposes the same entry points (`download_data.py`, `train.py`, `evaluate.py`, `export.py`, `serve.py`). This reduces cognitive load and enables automation.

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

This portfolio was developed with AI assistance (Claude) while maintaining human oversight on architecture decisions, evaluation methodology, and code quality. The goal is to demonstrate not just ML skills, but the judgment to know when AI suggestions are appropriate and when they need refinement.
