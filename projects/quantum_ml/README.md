# Quantum Machine Learning

**Status**: 📋 Planned

## Purpose

Explore the intersection of quantum computing and machine learning. Focus on understanding where quantum approaches provide genuine advantages, the current limitations of NISQ (Noisy Intermediate-Scale Quantum) hardware, and building practical hybrid quantum-classical pipelines.

This is not about hype—it's about hands-on experimentation to understand what works, what doesn't, and why.

## Key Technologies

- **Qiskit**: IBM's quantum computing framework
- **PennyLane**: Quantum ML library with autodiff support
- **Cirq**: Google's quantum computing framework (comparison)
- **Qiskit Machine Learning**: High-level QML algorithms
- **Classical simulators**: Aer, default.qubit for development

## Core Concepts to Explore

### Quantum Data Encoding
How classical data enters the quantum world:
- **Angle Encoding**: Map features to rotation angles (RY, RZ gates)
- **Amplitude Encoding**: Encode vectors in quantum amplitudes (exponentially compact)
- **Basis Encoding**: Binary features as qubit states
- **Feature Maps**: Non-linear quantum embeddings for kernel methods

### Variational Quantum Circuits (VQC)
The "neural networks" of quantum ML:
- Parameterized gates (trainable rotations)
- Entanglement layers (CNOT, CZ patterns)
- Ansatz design (hardware-efficient vs problem-specific)
- Hybrid training loops (quantum forward pass, classical optimizer)

### Quantum Kernels
Leverage quantum computers for kernel methods:
- Quantum feature maps create high-dimensional embeddings
- Kernel trick: compute inner products in quantum Hilbert space
- Compare with classical RBF/polynomial kernels

## Planned Experiments

1. **Variational Quantum Classifier (VQC)**
   - Binary classification on toy datasets (moons, circles, iris)
   - Compare: VQC vs classical MLP vs SVM
   - Analyze: circuit depth, number of parameters, convergence
   - Measure: accuracy, training time, simulator overhead

2. **Quantum Kernel Methods**
   - Implement quantum kernel estimation
   - QSVM on small datasets
   - When does quantum kernel beat RBF kernel?
   - Feature map design impact

3. **Barren Plateau Analysis**
   - Gradient vanishing in deep random circuits
   - Mitigation strategies (local cost functions, structured ansätze)
   - Practical depth limits for trainable circuits

4. **Noise Impact Study**
   - Compare ideal simulator vs noisy simulator vs hardware
   - Error mitigation techniques (zero-noise extrapolation, probabilistic error cancellation)
   - How much noise kills quantum advantage?

5. **Hybrid Classical-Quantum Models**
   - CNN feature extractor + quantum classifier head
   - When does the quantum layer help?
   - Practical integration patterns

## NISQ Era Constraints

Understanding current hardware limitations:

| Constraint | Typical Value | Impact |
|------------|---------------|--------|
| **Qubit count** | 50-1000 | Limits problem size |
| **Gate fidelity** | ~99.5% | Errors compound with depth |
| **Coherence time** | ~100 µs | Limits circuit execution time |
| **Connectivity** | 2D grid | Requires SWAP gates, increases depth |
| **Readout error** | ~1% | Noisy measurements |

## Evaluation Approach

Not just "does it work?" but "does it make sense?":

1. **Fair Comparisons**: Same data, same compute budget, same tuning effort
2. **Resource Accounting**: Include classical simulation cost, circuit compilation overhead
3. **Scaling Analysis**: How does performance change with qubits/depth?
4. **Practical Viability**: Could this run on real hardware today?

## Interconnections

- **Bayesian Optimization**: Tuning VQC hyperparameters (ansatz structure, learning rate)
- **Scientific ML - Materials**: Quantum chemistry simulations (VQE for molecular energies)
- **Tabular Boosting**: Baseline comparisons for classification tasks

## Tools & Environment

```bash
# Core quantum libraries
pip install qiskit qiskit-machine-learning pennylane

# Visualization and analysis
pip install qiskit-aer matplotlib

# Optional: IBM Quantum access
# pip install qiskit-ibm-runtime
```

## Project Structure (Planned)

```
quantum_ml/
├── configs/
│   └── default.yaml
├── notebooks/
│   ├── 01_quantum_basics.ipynb      # Gates, circuits, measurement
│   ├── 02_data_encoding.ipynb       # Encoding strategies comparison
│   ├── 03_vqc_classifier.ipynb      # Variational classifier experiments
│   ├── 04_quantum_kernels.ipynb     # QSVM and kernel methods
│   └── 05_noise_analysis.ipynb      # NISQ limitations study
├── project/
│   ├── __init__.py
│   ├── circuits.py                  # Ansatz and feature map builders
│   ├── data.py                      # Data loading and encoding
│   ├── train.py                     # Hybrid training loops
│   └── eval.py                      # Evaluation utilities
├── scripts/
│   └── train.py
└── tests/
    └── test_qml_smoke.py
```

## References

### Frameworks
- [Qiskit Textbook](https://learning.quantum.ibm.com/)
- [PennyLane Tutorials](https://pennylane.ai/qml/)
- [Cirq Documentation](https://quantumai.google/cirq)

### Papers
- [Variational Quantum Eigensolver (VQE)](https://arxiv.org/abs/1304.3061) - Peruzzo et al.
- [Quantum Machine Learning](https://arxiv.org/abs/1611.09347) - Schuld & Killoran
- [Barren Plateaus in QML](https://arxiv.org/abs/1803.11173) - McClean et al.
- [Power of Data in Quantum ML](https://arxiv.org/abs/2011.01938) - Huang et al.

### Courses
- [MIT xQuantum](https://xpro.mit.edu/programs/program-v1:xPRO+QCF/)
- [Qiskit Global Summer School](https://qiskit.org/events/summer-school/)

## Next Steps

1. Set up Qiskit + PennyLane environment
2. Work through quantum basics notebook (gates, measurement)
3. Implement angle encoding and amplitude encoding
4. Build first VQC on moons dataset
5. Compare with classical baselines (fair evaluation)
6. Document: "What did quantum actually buy us here?"

## Honest Assessment

QML in 2026 is still largely research territory. The goal of this project is not to prove quantum supremacy, but to:
- Build intuition for quantum computation
- Understand hybrid quantum-classical workflows
- Identify problem classes where quantum might help (eventually)
- Be ready when hardware improves

The most valuable outcome may be learning what *doesn't* work and why.
