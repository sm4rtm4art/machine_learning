# Tabular Boosting Suite

**Status**: 📋 Planned

A comprehensive benchmark comparing gradient boosting frameworks (LightGBM, XGBoost, CatBoost) with rigorous evaluation and calibration analysis.
Futher the complementary tabular transfromer models are applied like [TabPFN](https://github.com/PriorLabs/TabPFN)

## Planned Features

- Binary classification with imbalance handling
- Regression with uncertainty quantification
- Calibration analysis (ECE, Brier score, reliability diagrams)
- Feature importance (gain + permutation)
- SHAP explanations
- Slice-based evaluation (missingness, category frequency, outliers)
- Inference latency benchmarks
- TabPFN as comparisson
- 
## Datasets

- Classification: Credit default prediction or similar
- Regression: House price prediction with heteroscedastic noise

## Coming Soon

This project will demonstrate:
- Hyperparameter tuning with Optuna
- MLflow experiment tracking
- Production-ready model evaluation
- Decision support for model selection
