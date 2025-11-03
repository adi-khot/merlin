merlin.algorithms.kernels module
================================

.. automodule:: merlin.algorithms.kernels
   :members: FeatureMap, FidelityKernel, KernelCircuitBuilder
   :undoc-members:
   :show-inheritance:

Examples
--------

Quickstart: Fidelity kernel in a few lines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    from merlin.algorithms.kernels import FidelityKernel

    # Build a kernel where inputs of size 2 are encoded in a 4-mode circuit
    kernel = FidelityKernel.simple(
        input_size=2,
        n_modes=4,
        shots=0,                 # exact probabilities (no sampling)
        no_bunching=False,       # allow bunched outcomes if needed
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    # X_train: (N, 2), X_test: (M, 2)
    X_train = torch.rand(10, 2)
    X_test = torch.rand(5, 2)

    K_train = kernel(X_train)               # (N, N)
    K_test = kernel(X_test, X_train)        # (M, N)

Custom circuit with FeatureMap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    import perceval as pcvl
    from merlin.algorithms.kernels import FeatureMap, FidelityKernel
    from merlin.core.generators import CircuitType, StatePattern
    from merlin.core.photonicbackend import PhotonicBackend

    # Backend describes modes/photons/topology
    backend = PhotonicBackend(
        circuit_type=CircuitType.SERIES,
        n_modes=6,
        n_photons=2,
        state_pattern=StatePattern.PERIODIC,
    )

    # Use the backend to create a FeatureMap automatically
    feature_map = FeatureMap.from_photonic_backend(
        input_size=3,
        photonic_backend=backend,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    # Build the kernel; you can pass a custom input Fock state if desired
    kernel = FidelityKernel(
        feature_map=feature_map,
        input_state=[1, 1, 0, 0, 0, 0],
        shots=0,
        no_bunching=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    X = torch.rand(8, 3)
    K = kernel(X)  # (8, 8)

Use with scikit-learn (precomputed kernel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    from sklearn.svm import SVC
    from merlin.algorithms.kernels import FidelityKernel

    # Build kernel and compute Gram matrices
    kernel = FidelityKernel.simple(input_size=4, n_modes=6)
    K_train = kernel(X_train)
    K_test = kernel(X_test, X_train)

    # Train a precomputed-kernel SVM
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
