# Graph Neural Networks

**Status**: 📋 Planned

## Purpose

Explore Graph Neural Networks (GNNs) for learning on non-Euclidean data structures. Focus on node classification, graph classification, and link prediction tasks.

## Key Technologies

- **PyTorch Geometric (PyG)**: Primary GNN library
- **DGL (Deep Graph Library)**: Alternative framework comparison
- **NetworkX**: Graph manipulation and visualization
- **Graph datasets**: Cora, CiteSeer, PubMed, OGB datasets

## Planned Experiments

1. **Node Classification**
   - GCN, GAT, GraphSAGE comparison
   - Inductive vs transductive learning
   - Heterogeneous graphs

2. **Graph Classification**
   - Molecular property prediction
   - Social network analysis
   - Pooling strategies (global, hierarchical)

3. **Link Prediction**
   - Knowledge graph completion
   - Recommendation systems
   - Temporal graphs

4. **Explainability**
   - GNNExplainer
   - Attention visualization
   - Subgraph importance

## Interconnections

- **Feeds into**: [Scientific ML - Materials Discovery](../scientific_ml_materials/) (crystal structure graphs)
- **Techniques**: Message passing, graph pooling, attention mechanisms
- **Evaluation**: Node/graph-level metrics, robustness to graph perturbations

## References

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [DGL Documentation](https://docs.dgl.ai/)
- [Open Graph Benchmark](https://ogb.stanford.edu/)
- [GNN Papers](https://github.com/thunlp/GNNPapers)

## Next Steps

1. Set up PyG environment
2. Implement baseline GCN on Cora dataset
3. Compare architectures (GCN, GAT, GraphSAGE)
4. Apply to molecular property prediction
