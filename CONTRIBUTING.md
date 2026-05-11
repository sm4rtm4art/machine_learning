# Contributing

This repository is primarily a personal ML portfolio, but issues and suggestions are welcome. The current priority is stabilization: keep changes small, reviewable, and honest about implementation status.

## Local Setup

```bash
uv sync --all-extras
make pre-commit-install
```

If a full dependency sync is heavy for your machine, install only the extra needed for the project you are touching.

## Development Workflow

1. Create a focused branch for one change.
2. Prefer reusable code in `src/ml_portfolio/` and project-specific logic in `projects/<name>/project/`.
3. Keep scripts in `projects/<name>/scripts/` thin; they should call importable project code.
4. Do not commit generated data, model artifacts, MLflow state, local SQLite databases, notebook outputs, or secrets.
5. Update public docs only when the files, scripts, notebooks, or artifacts they describe actually exist.

## Checks

Run the narrowest useful check first:

```bash
make lint
make test
```

Before publishing or opening a pull request, prefer:

```bash
make check
uv run pre-commit run --all-files
```

## Notebooks And Prototypes

Notebooks are used for reports, exploration, and communication. Durable training, evaluation, and preprocessing logic should live in importable Python modules.

Prototype code is allowed when clearly labeled, but it should not be presented as production-ready. If a prototype becomes part of an active project, move reusable logic into that project's `project/` package and add focused tests.
