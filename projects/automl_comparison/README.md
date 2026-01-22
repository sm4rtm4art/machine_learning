# AutoML Comparison

**Status**: 📋 Planned

## Purpose

Systematic comparison of AutoML frameworks to understand when automation helps and when manual tuning is necessary. Focus on tabular data and practical deployment considerations.

## Key Technologies

- **Auto-sklearn**: Automated scikit-learn pipeline search
- **FLAML**: Fast and lightweight AutoML (Microsoft)
- **H2O AutoML**: Enterprise-grade AutoML
- **AutoGluon**: Deep learning-based AutoML (Amazon)
- **TPOT**: Genetic programming for pipeline optimization

## Planned Experiments

1. **Framework Comparison**
   - Accuracy vs time budget tradeoffs
   - Interpretability of generated pipelines
   - Robustness across datasets

2. **When AutoML Helps**
   - Dataset characteristics (size, features, class balance)
   - Time constraints
   - Domain expertise availability

3. **When Manual Tuning Wins**
   - Domain-specific feature engineering
   - Custom loss functions
   - Production constraints (latency, memory)

4. **Hybrid Approaches**
   - AutoML for initial baseline
   - Manual refinement of top pipelines
   - Warm-starting with domain knowledge

## Interconnections

- **Uses**: [Tabular Boosting Suite](../tabular_boosting/) (baseline models)
- **Leverages**: [Bayesian Optimization](../bayesian_optimization/) (HPO component)
- **Compares with**: Manual pipelines from other projects

## Benchmark Datasets

- OpenML-CC18 (curated classification tasks)
- Kaggle competition datasets
- UCI ML Repository
- Custom business datasets (anonymized)

## Evaluation Criteria

| Criterion | Metric |
|-----------|--------|
| **Accuracy** | Test set performance |
| **Efficiency** | Time to best model |
| **Robustness** | Performance variance across datasets |
| **Interpretability** | Pipeline complexity, feature importance |
| **Production** | Model size, inference latency |

## References

- [AutoML Survey](https://arxiv.org/abs/1908.00709)
- [Auto-sklearn](https://automl.github.io/auto-sklearn/)
- [FLAML Paper](https://arxiv.org/abs/1911.04706)
- [AutoGluon](https://auto.gluon.ai/)

## Next Steps

1. Set up benchmark suite (OpenML-CC18)
2. Run Auto-sklearn, FLAML, H2O on same tasks
3. Analyze pipeline patterns
4. Document when manual beats auto
