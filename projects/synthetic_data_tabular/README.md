# Synthetic Tabular Data

**Status**: 📋 Planned

Experimental evaluation of synthetic data generation for tabular datasets, focusing on utility vs privacy tradeoffs.

## Planned Features

- Generator comparison:
  - CTGAN baseline
  - Modern alternatives (TabDDPM, etc.)
- Utility evaluation:
  - Train-on-synthetic-test-on-real (TSTR)
  - Train-on-real-test-on-synthetic (TRTS)
  - Distributional metrics (marginals, pairwise correlations)
- Privacy risk assessment:
  - Nearest neighbor distance leakage
  - Membership inference attack simulation
- Decision framework for when to use synthetic data

## Coming Soon

This project will demonstrate:
- Practical synthetic data evaluation
- Privacy-utility tradeoff analysis
- Guidance on when synthetic data helps vs harms
- Reproducible experimental framework
