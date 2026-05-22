# Design 0001: Privacy Evaluation Protocol

**Status**: Planned
**Project**: Synthetic Tabular Data

## Context

Synthetic data does not automatically provide privacy. A generator can memorize training records, recreate rare individuals, leak sensitive attributes, or produce rows that are dangerously close to real records.

This project needs a defense-in-depth privacy evaluation protocol before synthetic datasets can be treated as useful portfolio artifacts.

## Decision

The planned privacy evaluation will combine multiple risk indicators rather than relying on a single metric.

### 1. Distance to Closest Record

Distance to Closest Record (DCR) will be used to identify synthetic rows that are exact or near copies of real training rows.

The planned evaluation will compare synthetic-to-real distances against real-to-real distance distributions where appropriate, so that unusually close synthetic records can be flagged for review.

### 2. Membership Inference Attacks

Membership inference attack (MIA) simulations will be used to estimate whether an attacker can infer that a specific real record was included in the generator's training data.

Tools such as Anonymeter may be evaluated for this purpose if they are compatible with the project architecture and dependency constraints.

### 3. Singling-Out and Outlier-Risk Analysis

Classic k-anonymity, l-diversity, and t-closeness are not treated here as privacy guarantees for synthetic data.

Instead, this project plans to use rare equivalence classes as a singling-out and outlier-risk signal:

1. Identify quasi-identifiers in the original real training dataset.
2. Find highly unique or low-k equivalence classes in that real dataset.
3. Cross-reference generated synthetic rows against those rare real groups.
4. Flag cases where the generator appears to recreate rare, isolated, or sensitive real-data patterns.

This analysis is intended to detect whether the generator copied or over-preserved unusual real individuals or groups.

## Limitations

These checks are risk indicators, not formal privacy proofs.

The protocol does not claim differential privacy unless a future generator explicitly implements a documented DP mechanism with a stated privacy budget.

Privacy results must be interpreted alongside utility, fidelity, and validity metrics. A dataset with high utility but high memorization risk should not be considered successful.
