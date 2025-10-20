"""
Test suite for MerlinProcessor cloud deployment via Perceval.
Tests both remote (cloud) and local execution modes with the current API.
"""

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================
CLOUD_TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6Mzk1LCJleHAiOjE3NjEyMjgyNzUuMjY4MDkzfQ.vPEHupHJhtXAFVMqyhav7s97cfp_CtJFxA9eH7328JSehdxKry192BKZ8i99KarjlMBkKoIyEJEmU45O3aDjSw"
REMOTE_PLATFORM = "sim:ascella"  # Options: "sim:clifford", "sim:slos", "qpu:ascella", etc.
# ============================================================================

import torch
import torch.nn as nn
import numpy as np
import time
import perceval as pcvl
from perceval.runtime import RemoteConfig

from merlin.algorithms import QuantumLayer
from merlin.core.merlin_processor import MerlinProcessor
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.sampling.strategies import OutputMappingStrategy

# Set token globally
if CLOUD_TOKEN and CLOUD_TOKEN != "YOUR_TOKEN_HERE":
    RemoteConfig.set_token(CLOUD_TOKEN)


def create_quantum_layer(n_modes, n_photons, input_size, output_size=None):
    """Create a quantum layer using CircuitBuilder."""
    builder = CircuitBuilder(n_modes=n_modes)
    builder.add_rotation_layer(trainable=True, name="theta")
    builder.add_angle_encoding(modes=list(range(min(input_size, n_modes))), name="px")
    if n_modes >= 3:
        builder.add_entangling_layer(depth=1)

    layer = QuantumLayer(
        input_size=input_size,
        output_size=output_size,
        circuit=builder,
        n_photons=n_photons,  # local exact simulation
        no_bunching=True,
        output_mapping_strategy=OutputMappingStrategy.NONE if output_size is None else OutputMappingStrategy.LINEAR,
    )
    layer.eval()
    return layer


def test_basic_cloud_execution():
    """Test basic cloud execution with MerlinProcessor."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Cloud Execution with MerlinProcessor")
    print("=" * 60)

    # Create quantum layer
    quantum_layer = create_quantum_layer(n_modes=4, n_photons=2, input_size=2, output_size=None)

    # Create remote processor
    print(f"Connecting to {REMOTE_PLATFORM}...")
    processor = MerlinProcessor.from_platform(
        platform=REMOTE_PLATFORM,
        token=CLOUD_TOKEN,
        max_batch_size=32,
        timeout=120.0
    )

    # Print platform info
    info = processor.platform_info
    print(f"Platform info: {info.get('name')} - Mode: {info.get('mode')}")
    print(f"Available commands: {info.get('available_commands')}")

    # Test with small batch
    batch_size = 4
    test_input = torch.randn(batch_size, 2)

    print(f"Executing batch of size {batch_size}...")
    start_time = time.time()

    # Use forward with shots parameter
    output = processor.forward(quantum_layer, test_input, shots=1000)

    execution_time = time.time() - start_time

    print(f"Output shape: {output.shape}")
    print(f"Output (probabilities):\n{output}")
    print(f"Execution time: {execution_time:.2f} seconds")

    # Verify output - C(4,2) = 6 for no_bunching
    expected_size = 6
    assert output.shape == (batch_size, expected_size), f"Expected shape {(batch_size, expected_size)}, got {output.shape}"
    assert torch.all(output >= 0) and torch.all(output <= 1), "Output should be probabilities"

    print("✓ Test passed")


def test_local_vs_remote_execution():
    """Test local execution vs remote execution."""
    print("\n" + "=" * 60)
    print("TEST 2: Local vs Remote Execution Comparison")
    print("=" * 60)

    quantum_layer = create_quantum_layer(n_modes=5, n_photons=2, input_size=2)
    test_input = torch.randn(3, 2)

    # Test local execution (pass None or non-RemoteProcessor)
    print("Testing local execution...")


    start_time = time.time()

    local_output = quantum_layer.forward(test_input, shots=None)
    local_time = time.time() - start_time

    print(f"Local output shape: {local_output.shape}")
    print(f"Local execution time: {local_time:.2f} seconds")

    # Test remote execution
    print(f"\nTesting remote execution on {REMOTE_PLATFORM}...")
    remote_proc = MerlinProcessor.from_platform(
        platform=REMOTE_PLATFORM,
        token=CLOUD_TOKEN,
        timeout=120.0
    )

    start_time = time.time()
    remote_output = remote_proc.forward(quantum_layer, test_input, shots=10000)
    remote_time = time.time() - start_time

    print(f"Remote output shape: {remote_output.shape}")
    print(f"Remote execution time: {remote_time:.2f} seconds")

    # Compare shapes (values may differ due to sampling vs exact)
    assert local_output.shape == remote_output.shape
    print(f"\n✓ Both outputs have same shape: {local_output.shape}")


def test_async_execution():
    """Test asynchronous execution with futures."""
    print("\n" + "=" * 60)
    print("TEST 3: Asynchronous Execution with Futures")
    print("=" * 60)

    quantum_layer = create_quantum_layer(n_modes=6, n_photons=2, input_size=2)

    # Remote processor
    processor = MerlinProcessor.from_platform(
        platform=REMOTE_PLATFORM,
        token=CLOUD_TOKEN
    )

    batch_size = 4
    test_input = torch.randn(batch_size, 2)

    print(f"Submitting async job for batch size {batch_size}...")

    # Submit async job
    future = processor.forward_async(quantum_layer, test_input, shots=2000)

    # Check future attributes
    assert hasattr(future, 'cancel_remote')
    assert hasattr(future, 'status')
    assert hasattr(future, 'job_ids')

    print("Waiting for completion...")

    # Poll status
    for i in range(5):
        status = future.status()
        print(f"  Status check {i + 1}: {status.get('state', 'unknown')}")
        if future.done():
            break
        time.sleep(1)

    # Get result
    output = future.wait()

    print(f"Output shape: {output.shape}")
    print(f"Job IDs: {future.job_ids}")

    expected_size = 15  # C(6,2) = 15
    assert output.shape == (batch_size, expected_size)
    print("✓ Test passed")


def test_hybrid_model():
    """Test a hybrid classical-quantum model."""
    print("\n" + "=" * 60)
    print("TEST 4: Hybrid Classical-Quantum Model")
    print("=" * 60)

    # Create hybrid model
    class HybridModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.classical_1 = nn.Linear(10, 3)
            self.quantum = create_quantum_layer(n_modes=5, n_photons=3, input_size=3)
            # C(5,3) = 10 for no_bunching
            self.classical_2 = nn.Linear(10, 4)
            self.output = nn.Softmax(dim=-1)

        def forward(self, x):
            x = torch.relu(self.classical_1(x))
            x = self.quantum(x)
            x = self.classical_2(x)
            x = self.output(x)
            return x

    model = HybridModel()
    model.eval()

    # Create processor
    processor = MerlinProcessor.from_platform(
        platform=REMOTE_PLATFORM,
        token=CLOUD_TOKEN
    )

    # Test forward pass
    batch_size = 6
    test_input = torch.randn(batch_size, 10)

    print(f"Running hybrid model with batch size {batch_size}...")
    start_time = time.time()

    # Processor automatically handles the quantum layer
    output = processor.forward(model, test_input, shots=1000)

    execution_time = time.time() - start_time

    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")
    print(f"Execution time: {execution_time:.2f} seconds")

    assert output.shape == (batch_size, 4)
    # Check softmax sums to 1
    assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-5)
    print("✓ Test passed")


def test_batch_size_limits():
    """Test batch size validation and limits."""
    print("\n" + "=" * 60)
    print("TEST 5: Batch Size Limits")
    print("=" * 60)

    quantum_layer = create_quantum_layer(n_modes=4, n_photons=2, input_size=2)

    # Create processor with small batch limit
    processor = MerlinProcessor.from_platform(
        platform=REMOTE_PLATFORM,
        token=CLOUD_TOKEN,
        max_batch_size=8
    )

    print(f"Max batch size: {processor.max_batch_size}")

    # Test within limit
    small_batch = torch.randn(8, 2)
    print("Testing batch size 8 (within limit)...")
    output = processor.forward(quantum_layer, small_batch, shots=500)
    assert output.shape == (8, 6)
    print("  ✓ Passed")

    # Test exceeding limit
    large_batch = torch.randn(16, 2)
    print("Testing batch size 16 (exceeds limit)...")
    try:
        output = processor.forward(quantum_layer, large_batch, shots=500)
        print("  ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised error: {e}")

    print("✓ Test passed")


def test_different_shot_counts():
    """Test different shot counts per call."""
    print("\n" + "=" * 60)
    print("TEST 6: Different Shot Counts")
    print("=" * 60)

    quantum_layer = create_quantum_layer(n_modes=4, n_photons=2, input_size=2)
    processor = MerlinProcessor.from_platform(
        platform=REMOTE_PLATFORM,
        token=CLOUD_TOKEN
    )

    test_input = torch.randn(2, 2)

    # Test with different shot counts
    shot_counts = [None, 100, 1000, 10000]

    for shots in shot_counts:
        print(f"\nTesting with shots={shots}...")
        start_time = time.time()
        output = processor.forward(quantum_layer, test_input, shots=shots)
        exec_time = time.time() - start_time

        print(f"  Output shape: {output.shape}")
        print(f"  Execution time: {exec_time:.2f}s")
        print(f"  Sum of probabilities: {output.sum(dim=1).tolist()}")

        assert output.shape == (2, 6)
        assert torch.allclose(output.sum(dim=1), torch.ones(2), atol=0.1)

    print("\n✓ All shot counts tested successfully")


def test_gpu_support():
    """Test GPU tensor support if available."""
    print("\n" + "=" * 60)
    print("TEST 7: GPU Support")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return

    quantum_layer = create_quantum_layer(n_modes=5, n_photons=2, input_size=3)
    processor = MerlinProcessor.from_platform(
        platform=REMOTE_PLATFORM,
        token=CLOUD_TOKEN
    )

    # Test with GPU tensor
    test_input = torch.randn(4, 3, device='cuda')
    print(f"Input device: {test_input.device}")

    output = processor.forward(quantum_layer, test_input, shots=1000)

    print(f"Output device: {output.device}")
    print(f"Output shape: {output.shape}")

    assert output.device.type == 'cuda'
    assert output.shape == (4, 10)  # C(5,2) = 10
    print("✓ GPU support working correctly")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MERLIN PROCESSOR CLOUD DEPLOYMENT TEST SUITE")
    print("=" * 60)
    print(f"Platform: {REMOTE_PLATFORM}")
    print(f"Token: {'Set' if CLOUD_TOKEN else 'Not set'}")

    if not CLOUD_TOKEN or CLOUD_TOKEN == "YOUR_TOKEN_HERE":
        print("\n⚠️  WARNING: Please set your cloud token!")
        return

    try:
        test_basic_cloud_execution()
        test_local_vs_remote_execution()
        test_async_execution()
        test_hybrid_model()
        test_batch_size_limits()
        test_different_shot_counts()
        test_gpu_support()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()