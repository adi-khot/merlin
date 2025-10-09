merlin.core.cloud_processor module
===================================

.. automodule:: merlin.core.cloud_processor
   :members:
   :undoc-members:
   :show-inheritance:

--------------------------------------------------------------------------------
Cloud Execution for Quantum Layers
--------------------------------------------------------------------------------

The `cloud_processor` module provides seamless integration with Quandela Cloud platforms
for executing QuantumLayer computations on remote quantum processors and simulators.

Key Components
--------------

1. **CloudProcessor**:
   Main class handling cloud execution, batch job management, and result processing.

   .. code-block:: python

      from merlin.core.cloud_processor import CloudProcessor

      # Initialize with platform and authentication
      cloud_proc = CloudProcessor(
          platform="sim:ascella",
          token="your_token",
          max_batch_size=32,
          max_shots_per_call=100000
      )

2. **deploy_to_cloud**:
   Convenience function for deploying QuantumLayers to cloud platforms.

   .. code-block:: python

      from merlin.core.cloud_processor import deploy_to_cloud

      # Deploy trained quantum layer
      cloud_proc = deploy_to_cloud(
          quantum_layer,
          platform="qpu:ascella",
          token="your_token",
          wait_timeout=120
      )

Supported Platforms
-------------------

- **Simulators**: ``sim:ascella``, ``sim:clifford``, ``sim:slos``
- **QPUs**: ``qpu:ascella``, ``qpu:andromeda``

Authentication
--------------

Three methods for providing authentication:

1. **Direct token**:

   .. code-block:: python

      cloud_proc = CloudProcessor(platform="sim:ascella", token="your_token")

2. **Environment variable**:

   .. code-block:: bash

      export PCVL_CLOUD_TOKEN="your_token"

3. **Perceval configuration**:

   .. code-block:: python

      from perceval.runtime import RemoteConfig
      RemoteConfig.set_token("your_token")
      RemoteConfig.save()

Usage Examples
--------------

**Basic Cloud Execution**:

.. code-block:: python

   import torch
   from merlin.algorithms import QuantumLayer
   from merlin.core.cloud_processor import deploy_to_cloud

   # Create and train quantum layer locally
   quantum_layer = QuantumLayer(
       input_size=4,
       output_size=3,
       circuit=my_circuit,
       trainable_parameters=["theta"],
       input_parameters=["input"],
       input_state=[1,0,1,0,1,0,0]
   )

   # ... training code ...

   # Deploy to cloud for inference
   cloud_proc = deploy_to_cloud(quantum_layer, platform="sim:ascella")

   # Switch to eval mode (required for cloud)
   quantum_layer.eval()

   # Execute on cloud
   X_test = torch.randn(10, 4)
   cloud_output = quantum_layer(X_test)  # Returns probabilities

**Batch Processing**:

.. code-block:: python

   # CloudProcessor automatically handles large batches
   large_batch = torch.randn(100, 4)  # Exceeds cloud limit of 32

   # Automatically split into sub-batches
   output = quantum_layer(large_batch)  # Processed in chunks

**No-Bunching Support**:

.. code-block:: python

   # Cloud processor respects no_bunching constraint
   quantum_layer = QuantumLayer(
       input_size=3,
       circuit=circuit,
       input_state=[1,1,0,0],
       no_bunching=True  # Only non-bunched states
   )

   cloud_proc = deploy_to_cloud(quantum_layer, platform="sim:slos")

Advanced Configuration
----------------------

**Custom Batch Settings**:

.. code-block:: python

   cloud_proc = CloudProcessor(
       platform="sim:ascella",
       token="your_token",
       max_batch_size=16,  # Smaller batches
       max_shots_per_call=50000,  # Fewer shots per call
       wait_timeout=180  # Longer timeout
   )

**Job Group Management**:

.. code-block:: python

   # Track multiple jobs
   cloud_proc = CloudProcessor(
       platform="qpu:ascella",
       use_job_group=True,
       job_group_name="my_experiment"
   )

   # Get job history
   jobs = cloud_proc.get_job_history()

   # Clear history
   cloud_proc.clear_job_history()

Important Notes
---------------

- **Inference Only**: Cloud execution only supports inference mode. Use ``quantum_layer.eval()``
- **Probability Output**: Cloud always returns probability distributions, never raw amplitudes
- **Batch Limits**: Maximum batch size is 32 samples per job
- **Shot Limits**: Maximum shots per call depends on platform (typically 100,000)

Error Handling
--------------

The module provides clear error messages for common issues:

- Missing authentication token
- Training mode detection (must use eval mode)
- Job timeouts
- Platform connection failures
- Batch size violations

Performance Tips
----------------

1. Use high shot counts for better statistical accuracy
2. Batch multiple samples together for efficiency
3. Consider platform-specific constraints when choosing backends
4. Use ``sim:slos`` for exact probability computation when available