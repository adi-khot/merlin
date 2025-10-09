"""
Test suite for Merlin cloud deployment via Perceval.
Tests using direct circuit construction instead of .simple() or ansatz.
"""

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================
CLOUD_TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6Mzk1LCJleHAiOjE3NjEyMjgyNzUuMjY4MDkzfQ.vPEHupHJhtXAFVMqyhav7s97cfp_CtJFxA9eH7328JSehdxKry192BKZ8i99KarjlMBkKoIyEJEmU45O3aDjSw"  # Replace with your Quandela Cloud token
"""
Test suite for Merlin cloud deployment via Perceval.
Tests using direct circuit construction instead of .simple() or ansatz.
"""

"""
Test suite for Merlin cloud deployment via Perceval.
Tests using direct circuit construction instead of .simple() or ansatz.
"""

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================

REMOTE_PLATFORM = "sim:ascella"  # Options: "sim:clifford", "sim:slos", "qpu:ascella", etc.
# ============================================================================

import torch
import torch.nn as nn
import numpy as np
import time
import perceval as pcvl

from merlin.algorithms import QuantumLayer
from merlin.core.cloud_processor import CloudProcessor, deploy_to_cloud
from merlin.sampling.strategies import OutputMappingStrategy


def create_quantum_circuit(m):
    """Create quantum circuit with specified number of modes.

    Args:
        m: Number of quantum modes in the circuit

    Returns:
        pcvl.Circuit: Complete quantum circuit with trainable parameters
    """
    # Left interferometer with trainable parameters
    wl = pcvl.GenericInterferometer(
        m,
        lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_li{i}")) //
                  pcvl.BS() // pcvl.PS(pcvl.P(f"theta_lo{i}")),
        shape=pcvl.InterferometerShape.RECTANGLE
    )

    # Variable phase shifters for input encoding
    c_var = pcvl.Circuit(m)
    for i in range(min(4, m)):  # Support up to 4 input features
        px = pcvl.P(f"px{i + 1}")
        c_var.add(i + (m - min(4, m)) // 2, pcvl.PS(px))

    # Right interferometer with trainable parameters
    wr = pcvl.GenericInterferometer(
        m,
        lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_ri{i}")) //
                  pcvl.BS() // pcvl.PS(pcvl.P(f"theta_ro{i}")),
        shape=pcvl.InterferometerShape.RECTANGLE
    )

    # Combine all components
    c = pcvl.Circuit(m)
    c.add(0, wl, merge=True)
    c.add(0, c_var, merge=True)
    c.add(0, wr, merge=True)

    return c


def test_basic_cloud_execution():
    """Test basic cloud execution with a custom quantum circuit."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Cloud Execution with Custom Circuit")
    print("=" * 60)

    # Create custom quantum circuit
    n_modes = 4
    circuit = create_quantum_circuit(n_modes)

    # Create quantum layer with direct circuit specification
    quantum_layer = QuantumLayer(
        input_size=2,
        output_size=4,
        circuit=circuit,
        trainable_parameters=["theta"],
        input_parameters=["px"],
        input_state=[1, 0, 1, 0],  # 2 photons in 4 modes
        output_mapping_strategy=OutputMappingStrategy.LINEAR,
        shots=1000
    )

    # Deploy to cloud
    print(f"Deploying custom circuit to {REMOTE_PLATFORM}...")
    cloud_proc = deploy_to_cloud(
        quantum_layer,
        platform=REMOTE_PLATFORM,
        token=CLOUD_TOKEN,
        wait_timeout=120  # Increase timeout for cloud execution
    )

    # Test with small batch
    batch_size = 4
    test_input = torch.randn(batch_size, 2)

    print(f"Executing batch of size {batch_size}...")
    quantum_layer.eval()  # Must be in eval mode for cloud

    start_time = time.time()
    output = quantum_layer(test_input)
    execution_time = time.time() - start_time

    print(f"Output shape: {output.shape}")
    print(f"Output (probabilities):\n{output}")
    print(f"Execution time: {execution_time:.2f} seconds")

    # Verify output
    assert output.shape == (batch_size, 4), f"Expected shape {(batch_size, 4)}, got {output.shape}"

    print("✓ Test passed")
    # Don't return anything!


def test_batch_splitting():
    """Test automatic batch splitting with custom circuit."""
    print("\n" + "=" * 60)
    print("TEST 2: Batch Splitting with Custom Circuit")
    print("=" * 60)

    # Create larger circuit
    n_modes = 6
    circuit = create_quantum_circuit(n_modes)

    # Create quantum layer
    quantum_layer = QuantumLayer(
        input_size=3,
        output_size=8,
        circuit=circuit,
        trainable_parameters=["theta"],
        input_parameters=["px"],
        input_state=[1, 1, 1, 0, 0, 0],  # 3 photons in 6 modes
        output_mapping_strategy=OutputMappingStrategy.LINEAR,
        shots=500
    )

    # Deploy with explicit batch size limit
    cloud_proc = CloudProcessor(
        platform=REMOTE_PLATFORM,
        token=CLOUD_TOKEN,
        max_batch_size=16,  # Set lower limit for testing
        wait_timeout=120
    )
    cloud_proc.attach_layer(quantum_layer)

    # Test with batch exceeding limit
    batch_size = 40  # Exceeds limit of 16
    test_input = torch.randn(batch_size, 3)

    print(f"Testing batch size {batch_size} with limit {cloud_proc.max_batch_size}")
    is_valid, message = cloud_proc.validate_batch_size(batch_size)
    print(f"Validation: {message}")

    quantum_layer.eval()

    start_time = time.time()
    output = quantum_layer(test_input)
    execution_time = time.time() - start_time

    print(f"Output shape: {output.shape}")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"First 5 samples:\n{output[:5]}")

    assert output.shape == (batch_size, 8), f"Expected shape {(batch_size, 8)}, got {output.shape}"
    print("✓ Test passed - batch splitting works correctly")
    # Don't return anything!


def test_hybrid_model():
    """Test a hybrid classical-quantum model with custom circuits."""
    print("\n" + "=" * 60)
    print("TEST 3: Hybrid Model with Custom Quantum Circuit")
    print("=" * 60)

    # Create hybrid model
    class HybridModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.classical_1 = nn.Linear(10, 4)

            # Create custom quantum circuit
            circuit = create_quantum_circuit(5)
            self.quantum = QuantumLayer(
                input_size=4,
                output_size=6,
                circuit=circuit,
                trainable_parameters=["theta"],
                input_parameters=["px"],
                input_state=[1, 1, 1, 1, 0],  # 4 photons in 5 modes
                output_mapping_strategy=OutputMappingStrategy.LINEAR,
                shots=1000
            )

            self.classical_2 = nn.Linear(6, 3)

        def forward(self, x):
            x = torch.relu(self.classical_1(x))
            x = self.quantum(x)
            x = self.classical_2(x)
            return x

    model = HybridModel()

    # Deploy quantum layer to cloud
    print(f"Deploying quantum layer to {REMOTE_PLATFORM}...")
    cloud_proc = deploy_to_cloud(
        model.quantum,
        platform=REMOTE_PLATFORM,
        token=CLOUD_TOKEN,
        wait_timeout=120
    )

    # Test forward pass
    batch_size = 8
    test_input = torch.randn(batch_size, 10)

    model.eval()  # Required for cloud execution

    print(f"Running hybrid model with batch size {batch_size}...")
    start_time = time.time()
    output = model(test_input)
    execution_time = time.time() - start_time

    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")
    print(f"Execution time: {execution_time:.2f} seconds")

    assert output.shape == (batch_size, 3), f"Expected shape {(batch_size, 3)}, got {output.shape}"
    print("✓ Test passed")
    # Don't return anything!


def test_no_bunching_constraint():
    """Test quantum layer with no_bunching constraint."""
    print("\n" + "=" * 60)
    print("TEST 4: No-Bunching Constraint")
    print("=" * 60)

    # Create circuit
    n_modes = 7
    circuit = create_quantum_circuit(n_modes)

    # Create layer with no_bunching=True
    quantum_layer = QuantumLayer(
        input_size=4,
        output_size=3,
        circuit=circuit,
        trainable_parameters=["theta"],
        input_parameters=["px"],
        input_state=[1, 0] * 3 + [1],  # 4 photons spread across 7 modes
        no_bunching=True,  # Exclude states with multiple photons per mode
        output_mapping_strategy=OutputMappingStrategy.LINEAR,
        shots=1000
    )

    # Deploy to cloud
    print(f"Deploying no-bunching circuit to {REMOTE_PLATFORM}...")
    cloud_proc = deploy_to_cloud(
        quantum_layer,
        platform=REMOTE_PLATFORM,
        token=CLOUD_TOKEN,
        wait_timeout=120
    )

    quantum_layer.eval()

    # Test execution
    batch_size = 5
    test_input = torch.randn(batch_size, 4)

    print(f"Executing with no_bunching=True...")
    output = quantum_layer(test_input)

    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")

    assert output.shape == (batch_size, 3), f"Expected shape {(batch_size, 3)}, got {output.shape}"
    print("✓ Test passed")
    # Don't return anything!


def test_different_output_strategies():
    """Test different output mapping strategies."""
    print("\n" + "=" * 60)
    print("TEST 5: Different Output Mapping Strategies")
    print("=" * 60)

    strategies = [
        OutputMappingStrategy.LINEAR,
        # Removed RANDOM - it doesn't exist
        OutputMappingStrategy.LEXGROUPING,
        OutputMappingStrategy.MODGROUPING
    ]

    n_modes = 6
    circuit = create_quantum_circuit(n_modes)
    test_input = torch.randn(2, 3)

    for strategy in strategies:
        print(f"\nTesting {strategy.value} strategy...")

        # Create layer with specific strategy
        quantum_layer = QuantumLayer(
            input_size=3,
            output_size=4,
            circuit=circuit,
            trainable_parameters=["theta"],
            input_parameters=["px"],
            input_state=[1, 1, 1, 0, 0, 0],
            output_mapping_strategy=strategy,
            shots=500
        )

        # Deploy to cloud
        cloud_proc = deploy_to_cloud(
            quantum_layer,
            platform=REMOTE_PLATFORM,
            token=CLOUD_TOKEN,
            wait_timeout=120
        )

        quantum_layer.eval()
        output = quantum_layer(test_input)

        print(f"  Output shape: {output.shape}")
        print(f"  Sample output: {output[0]}")

        assert output.shape == (2, 4), f"Expected shape (2, 4), got {output.shape}"

    print("\n✓ All strategies tested successfully")
    # Don't return anything!


def test_platform_info():
    """Test platform information retrieval."""
    print("\n" + "=" * 60)
    print("TEST 6: Platform Information")
    print("=" * 60)

    # Create simple circuit
    circuit = create_quantum_circuit(4)
    quantum_layer = QuantumLayer(
        input_size=2,
        output_size=4,
        circuit=circuit,
        trainable_parameters=["theta"],
        input_parameters=["px"],
        input_state=[1, 0, 1, 0],
        output_mapping_strategy=OutputMappingStrategy.LINEAR
    )

    # Deploy to cloud
    cloud_proc = deploy_to_cloud(
        quantum_layer,
        platform=REMOTE_PLATFORM,
        token=CLOUD_TOKEN
    )

    # Get platform info
    info = cloud_proc.platform_info

    print("Platform Information:")
    print(f"  Name: {info.get('name', 'N/A')}")
    print(f"  Status: {info.get('status', 'N/A')}")
    print(f"  Max Batch Size: {info.get('max_batch_size', 'N/A')}")

    if 'constraints' in info:
        print(f"  Constraints: {info['constraints']}")

    if 'performance' in info:
        print(f"  Performance: {info['performance']}")

    assert 'name' in info, "Platform info should contain name"
    assert 'status' in info, "Platform info should contain status"

    print("✓ Test passed")
    # Don't return anything!


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MERLIN CLOUD DEPLOYMENT TEST SUITE")
    print("Using Direct Circuit Construction")
    print("=" * 60)
    print(f"Platform: {REMOTE_PLATFORM}")
    print(f"Token: {'*' * 10}...")

    if CLOUD_TOKEN == "YOUR_TOKEN_HERE":
        print("\n⚠️  WARNING: Please set your cloud token in the file!")
        print("Edit CLOUD_TOKEN at the top of this file")
        return

    try:
        # Run tests
        test_basic_cloud_execution()
        test_batch_splitting()
        test_hybrid_model()
        test_no_bunching_constraint()
        test_different_output_strategies()
        test_platform_info()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()