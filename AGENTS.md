# AGENTS.md

This file is guidance for AI assistants working in this repository.

## Repository Posture

This is an ML portfolio with mixed project maturity. Treat it as a public-facing, work-in-progress engineering portfolio, not as a uniformly production-ready platform.

Keep the repository ambitious but truthful. Do not invent implementation status, benchmark numbers, notebook state, MLflow coverage, production readiness, CI enforcement, deployments, or model quality.

## Current Project Priority

Use this priority order when planning or changing work:

1. `projects/ocr_pipeline/`
2. `projects/synthetic_data_tabular/`
3. `projects/vision_ssl_transfer/`
4. `projects/timeseries_forecasting_covariates/`

Lower-priority projects should only be touched when they create public contradictions, broken links, hygiene risks, or small consistency fixes.

## Status Language

Use public maturity language carefully:

- `Active`: real code, notebooks, prototypes, or implementation work exists.
- `Planned`: roadmap/design work exists, but implementation is not yet present.
- `Prototype`: exploratory work exists and may not meet the main project standards yet.
- `Target standard`: desired convention that is not fully implemented across the repo.

If a file, notebook, metric, experiment, artifact, or integration does not exist, describe it as planned or target state.

## Documentation Rules

- Verify public claims against actual files before updating READMEs or docs.
- Prefer "production-grade direction" or "engineering discipline" over "production-quality" unless end-to-end behavior is proven.
- Avoid broad "Every project does..." claims unless the repository actually enforces them.
- Keep README tables aligned with real directories and implemented artifacts.
- Do not create decorative notebooks, placeholder metrics, or fake results to make a project look mature.

## Code And Notebook Rules

- Keep changes small, reviewable, and testable.
- Put reusable shared code in `src/ml_portfolio/`.
- Put project-specific logic in `projects/<name>/project/`.
- Keep `projects/<name>/scripts/` as thin CLI entry points.
- Keep core logic out of notebooks; notebooks should report, explore, or demonstrate.
- Prefer existing helpers for paths, logging, metrics, evaluation, slicing, and tracking before adding new abstractions.

## Artifact Hygiene

Do not commit local/generated state:

- local datasets
- model weights
- generated reports
- MLflow tracking state
- SQLite databases such as `mlflow.db`
- caches
- secrets or `.env` files

When generated files are useful, document the command that creates them instead of committing local machine output.

## MLflow, Optuna, And Evidently

Treat MLflow as the primary experiment-tracking convention. Do not imply full repository-wide tracking unless the implementation proves it.

Treat Optuna and Evidently as target conventions, prototype examples, or monitoring/data-quality direction unless a specific project has real integrated usage.

## Working Style

- Search existing code and docs before proposing new structure.
- Prefer stabilization before expansion.
- Make one focused change at a time.
- Run the narrowest useful check first; broaden only when risk requires it.
- Never overwrite user or collaborator work without explicit instruction.
