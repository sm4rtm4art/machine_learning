# ADR 0001: Isolated Generator Environments

**Status**: Planned
**Project**: Synthetic Tabular Data

## Context

The main repository targets Python 3.13 as a deliberate modern Python baseline. Some synthetic data libraries, especially legacy or research-oriented tabular generation tools, may lag behind current Python versions or require dependency versions that conflict with the main repository environment.

The synthetic data project should be able to evaluate generators such as CTGAN, VAE-style models, and tabular diffusion models without forcing the entire repository to adopt each generator's dependency constraints.

## Decision

Complex synthetic data generators will run in isolated environments when needed. These environments may use separate `uv` environments, Docker images, or another reproducible execution boundary.

The main evaluation core will remain in the repository's Python 3.13 environment.

Generator environments will communicate with the evaluation core through a strict data contract:

- Input data: `train.parquet`
- Metadata contract: `dataset_contract.yaml`
- Generated output: `synthetic.parquet`
- Optional run metadata: `run_manifest.yaml`

Generators must not receive validation or test data unless a future experiment explicitly documents and justifies that choice.

## Rationale

Parquet is preferred over CSV because tabular synthetic data evaluation depends on preserving types and missing-value semantics. CSV can silently change dates, nullable integers, booleans, categorical identifiers, leading zeros, and null representations. These changes can create false evaluation results or hide generator failures.

Parquet provides stronger native support for typed tabular exchange and reduces accidental schema drift between the evaluation core and isolated generator environments.

The metadata sidecar is still required because file types alone do not describe the full dataset contract. `dataset_contract.yaml` will define information such as:

- Column roles: numerical, categorical, ordinal, datetime, ID, target, sensitive, quasi-identifier
- Allowed categories
- Missing-value semantics
- Target column and prediction task
- Train/validation/test split policy
- Business rules and range constraints
- Privacy-relevant quasi-identifiers and sensitive attributes
- Whether labels may be used by the generator

## Consequences

This architecture keeps the main repository environment stable while allowing dependency-sensitive generator experiments.

It also creates extra validation work. Every generator output must be checked against the dataset contract before utility, fidelity, or privacy metrics are trusted.

This decision does not imply that any generator environment, adapter, or evaluation pipeline has been implemented yet.
