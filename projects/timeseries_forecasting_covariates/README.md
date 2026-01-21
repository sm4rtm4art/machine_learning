# Time Series Forecasting with Covariates

**Status**: 📋 Planned

Rigorous time series forecasting with exogenous covariates, proper backtesting, and probabilistic evaluation.

## Planned Features

- Model types:
  - Global models (train on multiple series)
  - Local models (one per series)
  - Hybrid approaches
- Covariate handling:
  - Past covariates (known history)
  - Future covariates (known future)
  - Static covariates
- Evaluation methodology:
  - Rolling-origin backtesting
  - Proper time splits (no leakage)
  - Covariate alignment validation
- Probabilistic metrics:
  - Pinball loss
  - Interval coverage (50%, 90%)
  - CRPS

## Coming Soon

This project will demonstrate:
- Leakage-free evaluation setup
- Backtesting implementation
- Residual diagnostics
- Drift detection inputs for monitoring
