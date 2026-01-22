# Scientific ML - Materials Discovery

**Status**: 📋 Planned

## Purpose

Apply machine learning to materials science, focusing on crystal structure property prediction and materials discovery. Combines physics-informed neural networks with graph-based representations.

## Key Technologies

- **JAX**: High-performance numerical computing with automatic differentiation
- **Equinox**: Neural networks in JAX
- **PyMatGen**: Materials analysis library
- **Crystal graphs**: Graph representations of crystal structures
- **DFT data**: Density Functional Theory computational results

## Planned Experiments

1. **Property Prediction**
   - Band gap prediction from crystal structure
   - Formation energy estimation
   - Stability prediction

2. **Physics-Informed Learning**
   - Incorporate symmetry constraints
   - Conservation laws in loss functions
   - Equivariant neural networks

3. **Materials Discovery**
   - Generative models for new materials
   - Active learning for efficient exploration
   - Multi-objective optimization

4. **Interpretability**
   - Feature importance for material properties
   - Attention on atomic sites
   - Uncertainty quantification

## Interconnections

- **Uses**: [Graph Neural Networks](../graph_neural_networks/) (crystal structure as graph)
- **Optimization**: [Bayesian Optimization](../bayesian_optimization/) (materials search)
- **Data**: Materials Project, OQMD, AFLOW databases

## Datasets

- [Materials Project](https://materialsproject.org/)
- [OQMD](http://oqmd.org/)
- [JARVIS-DFT](https://jarvis.nist.gov/)
- [Matbench](https://matbench.materialsproject.org/)

## References

- [CGCNN Paper](https://arxiv.org/abs/1710.10324)
- [MEGNet](https://arxiv.org/abs/1812.05055)
- [SchNet](https://arxiv.org/abs/1706.08566)
- [Equinox Documentation](https://docs.kidger.site/equinox/)

## Next Steps

1. Set up JAX environment
2. Load Materials Project dataset
3. Implement CGCNN baseline
4. Add physics-informed constraints
