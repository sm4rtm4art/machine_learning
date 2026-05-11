# Time Series Forecasting with Covariates

**Status**: 📋 Planned

Rigorous time series forecasting with exogenous covariates, proper backtesting, and probabilistic evaluation.

## Focus on different workflow

Time series data are conditional, so the training and the crossvalidation must be conditional.
Randomness destois the features we are interested in.

## Libraries of interest

Here are some libraries for time series analysis.

### Time Series
  - [statsmodels](https://www.statsmodels.org/stable/index.html)
  - [Darts](https://unit8co.github.io/darts/#)
  - [sktime](https://www.sktime.net/en/stable/index.html)
  - [Kats](https://facebookresearch.github.io/Kats/)
### Annomalie detection
  - [Python Outlier Detection](https://github.com/yzhao062/pyod)
  - [TODS](https://github.com/datamllab/tods)
### Time Series Foundation Models
  - [Chronos](https://github.com/amazon-science/chronos-forecasting)
  - [TimeFM](https://github.com/google-research/timesfm?tab=readme-ov-file)
  - [uni2ts](https://github.com/SalesforceAIResearch/uni2ts)

### Base model

Arima, Exponential smoothing.


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
