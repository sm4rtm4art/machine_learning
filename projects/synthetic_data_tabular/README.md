# Synthetic Tabular Data

**Status**: 📋 Planned priority track

## Current State

This project is currently in the planning and design phase. No runnable scripts, notebooks, trained generators, synthetic datasets, MLflow runs, privacy reports, or benchmark results are implemented yet.

The goal is to build this project around an evaluation-first question:

> When does synthetic tabular data preserve useful downstream signal without creating unacceptable privacy risk?

## Purpose

This project will evaluate synthetic data generation for tabular datasets through a utility, fidelity, privacy, and validity lens. The emphasis is not only on trying modern generators, but on building an evaluation ladder that makes it clear when a synthetic dataset is useful, when it is misleading, and when it may create privacy risk.

## Planned Evaluation Ladder

### Phase 0: Design and Contracts

- Define dataset contracts for tabular inputs and generator outputs.
- Define train/validation/test split ownership.
- Define utility, fidelity, privacy, and validity metrics.
- Document generator isolation strategy for dependency-sensitive tooling.

### Phase 1: Simple Baselines

Planned simple baselines:

- Bootstrap / row resampling
- Independent marginal sampling, for example with [DataSynthesizer](https://github.com/DataResponsibly/DataSynthesizer) or [SDV](https://github.com/sdv-dev/SDV) classical models
- Gaussian copula or other classical statistical synthesizers

These baselines will establish whether more complex generators provide value beyond simple distributional approximations.

### Phase 2: Neural Baselines

Planned neural baselines:

- Autoencoder or variational autoencoder style generator
- [CTGAN](https://github.com/sdv-dev/CTGAN)
- TVAE or similar tabular VAE baseline, likely through SDV tooling

These models are planned as practical synthetic-data baselines, likely through isolated generator environments when dependency compatibility requires it.

### Phase 3: Modern Generators

Planned modern approaches:

- Tabular diffusion models such as TabDDPM-style methods
- [Synthcity](https://github.com/vanderschaarlab/synthcity)-style generator comparisons, if dependency support is practical

These approaches are planned only after the evaluation protocol and simple baselines are stable.

## Planned Data Contract Architecture

The main repository targets Python 3.13. Some synthetic data libraries may lag behind current Python versions or require incompatible dependency sets.

To avoid forcing the main repository environment around one generator library, this project plans to use an isolated generator architecture:

- The main evaluation pipeline stays in the repository's Python 3.13 environment.
- Complex generators may run in isolated `uv` environments, Docker images, or other reproducible execution contexts.
- Generators communicate with the evaluation core through a strict tabular data contract.
- The data contract uses Parquet files plus a metadata schema file, not ad hoc CSV exchange, to strictly preserve data types such as dates and nullable integers and prevent silent drift.

Planned contract files:

- `train.parquet`: real training data made available to the generator.
- `dataset_contract.yaml`: schema, roles, constraints, target definition, split policy, and privacy-relevant metadata.
- `synthetic.parquet`: generated synthetic rows returned by the generator.
- `run_manifest.yaml`: generator name, version, seed, input row count, output row count, and environment metadata.

The generator must not receive validation or test rows unless a future experiment explicitly documents that choice.

## Planned Evaluation Protocol

### Utility

Target utility checks:

- Train-on-synthetic-test-on-real (TSTR)
- Train-on-real-test-on-synthetic (TRTS)
- Real-train-real-test reference baseline
- Downstream classification or regression performance
- Slice-level performance for missingness, rare categories, outliers, and class imbalance

### Fidelity

Target fidelity checks:

- Marginal distributions
- Category frequencies
- Pairwise correlations or dependency structure
- Missing-value patterns
- Class balance preservation
- Constraint preservation

### Privacy

Target privacy checks:

- Distance to Closest Record (DCR)
- Duplicate and near-duplicate detection
- Membership inference attack simulation, targeting tools like Statice's [Anonymeter](https://github.com/statice/anonymeter)
- Singling-out and outlier-risk analysis using rare real-data equivalence classes

These checks are risk indicators, not formal privacy guarantees. Formal differential privacy is out of scope unless a future generator explicitly implements and documents it.

### Validity

Target validity checks:

- Schema conformity
- Type preservation
- Allowed category validation
- Range and business-rule constraints
- Missing-value semantics
- Target leakage checks

## Planned Artifacts

Future implemented versions of this project may produce:

- Evaluation metrics in `metrics.json`
- Fidelity and privacy summary reports
- Slice-level utility outputs
- Validity reports for generated datasets
- MLflow experiment runs
- A report notebook summarizing results

These artifacts are planned and are not currently present.

## Planned Tooling References

Candidate tools and libraries to evaluate:

- [DataSynthesizer](https://github.com/DataResponsibly/DataSynthesizer) for classical statistical synthetic-data baselines.
- [SDV](https://github.com/sdv-dev/SDV) for Gaussian copula, CTGAN, and TVAE-style tabular generation.
- [CTGAN](https://github.com/sdv-dev/CTGAN) as a recognizable neural tabular generation baseline.
- [Synthcity](https://github.com/vanderschaarlab/synthcity) for broader synthetic-data benchmarking and possible diffusion-style comparisons.
- [Anonymeter](https://github.com/statice/anonymeter) for privacy attack and risk evaluation, if dependency compatibility is practical.

These references identify candidate tooling for the planned project. They are not currently integrated into this repository.

## Non-Goals

The initial project scope excludes:

- Fine-tuning LLMs for tabular generation
- Commercial synthetic-data platforms
- Claims of formal anonymization
- Claims of differential privacy without an implemented DP mechanism
- Broad benchmark claims before reproducible experiments exist

## Roadmap

1. Finalize README, ADR, and privacy evaluation design.
2. Define the Parquet + metadata dataset contract.
3. Implement one small tabular dataset and one simple baseline.
4. Add utility and fidelity evaluation.
5. Add privacy and validity checks.
6. Add CTGAN or VAE-style generator through an isolated environment.
7. Evaluate whether diffusion-based generators add measurable value.
