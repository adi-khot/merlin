:github_url: https://github.com/merlinquantum/merlin

=====================================
SLOS: Strong Linear Optical Simulator
=====================================

Summary
=======

SLOS (Strong Linear Optical Simulator) is a computational framework designed to solve the **BosonSampling problem** by directly computing the output probabilities of all possible quantum states (Fock states) in a photonic circuit. It relies on a **strong simulation** approach and is more efficient than traditional permanent-based methods in terms of time complexity, at the cost of increased memory usage.

SLOS is particularly useful for designing, testing, and verifying photonic quantum circuits, as well as for applications in quantum machine learning and quantum algorithm development.

Background
==========

Photonic quantum computing relies on **linear optical (LO) circuits**, which are usually composed of passive linear components such as beamsplitters and phaseshifters, as well as photon sources and detectors. These circuits manipulate indistinguishable photons to perform quantum computations.

The **BosonSampling problem** involves calculating the probability distribution of output Fock states when indistinguishable photons propagate through an LO circuit. This problem is known to be **#P-hard**, making it intractable for classical computers as the number of photons and modes increases.

Two main simulation approaches exist:

- **Weak simulation**: Simulates individual photon propagation using Monte Carlo methods (e.g., Clifford & Clifford, 2017). This approach is computationally expensive and only provides estimations of the respective probabilities.
- **Strong simulation**: Directly computes the exact probabilities of all output Fock states. SLOS is a strong simulation method that improves upon the state-of-the-art by reducing time complexity, though it requires significant memory.

Key Features of SLOS
====================

1. **Iterative Probability Calculation**
   SLOS computes the output probabilities iteratively, incrementally increasing the number of photons and calculating the probabilities of Fock states at each step. This avoids the exponential cost of computing permanents for each output state independently.

2. **Complexity Improvement**
   - **Previous state-of-the-art**: Permanent-based methods (e.g., Ryser, 1963; Glynn, 2010) and the previous state-of-the-art method (Shchesnovich, 2019) have a time complexity of :math:`O(n 2^n)` for computing the permanent of an :math:`n \times n` matrix.
   - **SLOS**: Achieves a time complexity of :math:`O(n \binom{n+m-1}{m-1})`, where :math:`n` is the number of photons and :math:`m` is the number of modes. This is **linear in the number of output states** and provides an exponential speedup over permanent-based methods for many practical cases.

3. **Memory Trade-off**
   SLOS requires :math:`O(\binom{n+m-1}{m-1})` memory to store intermediate results, which can be prohibitive for large :math:`n` and :math:`m`. However, this trade-off enables faster computation for moderate-sized problems.

Implementation
==============

SLOS is implemented in the **Perceval** library and integrated into **MerLin** for seamless use in quantum machine learning workflows. When constructing a ``QuantumLayer`` and calling ``forward()``, SLOS is used transparently for strong simulation.

Practical Performance
=====================

- **Benchmarking**: SLOS outperforms permanent-based methods (e.g., Glynn’s algorithm) for small to moderate :math:`n` and :math:`m`. For example, simulating a 10-photon, 10-mode circuit takes hours with SLOS but days with permanent-based methods.
- **Memory Limits**: Strong simulation becomes impractical for very large :math:`n` and :math:`m` due to memory constraints (e.g., 24 photons in 24 modes require ~1.5 Petabytes of memory).

For detailed memory and time performance evaluations, refer to :doc:`../performance/performance`.

Use Cases
=========

- **Quantum Machine Learning**: Training models that require full output distributions, such as differential equation solvers.
- **Circuit Design**: Verifying and optimizing LO circuits for specific tasks (e.g., entangled state generation, logic gates).
- **Noise Modeling**: Validating simulations against experimental hardware.

Conclusion
==========

SLOS is a powerful tool for strong simulation of linear optical circuits, offering significant speedups over permanent-based methods while trading off memory. It is ideal for applications requiring exact output probabilities, such as quantum algorithm development and circuit verification.

References
==========
- Heurtel, N., Mansfield, S., Senellart, J., & Valiron, B. (2023). *Strong Simulation of Linear Optical Processes*. `arXiv:2206.10549 <https://arxiv.org/abs/2206.10549>`_.
- Heurtel, N., Fyrillas, A., de Gliniasty, G., Bihan, R.L., et al. (2023). *Perceval: a software platform for discrete variable photonic quantum computing*. `arXiv:2204.00602 <https://arxiv.org/abs/2204.00602>`_.
- Shchesnovich, V. (2019). *On the classical complexity of sampling from quantum interference of indistinguishable bosons*. `arXiv:1904.02013 <https://arxiv.org/abs/1904.02013>`_.
- Clifford, P., & Clifford, R. (2017). *The Classical Complexity of Boson Sampling*. `arXiv:1706.01260 <https://arxiv.org/abs/1706.01260>`_.
- Glynn, D.G. (2010). *The permanent of a square matrix*. European Journal of Combinatorics, 31, 1887-1891.
- Ryser, H.J. (1963). *Combinatorial Mathematics*. American Mathematical Society.

