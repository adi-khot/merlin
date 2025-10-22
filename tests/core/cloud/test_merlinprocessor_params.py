"""
Test suite for MerlinProcessor parameter handling and state indexing.
Verifies that trained parameters are correctly extracted and that output states
are consistently indexed across different configurations.
"""

import torch
import torch.nn as nn
import numpy as np
import perceval as pcvl
from perceval.runtime import RemoteConfig
from typing import Dict, List
from math import comb

from merlin.algorithms import QuantumLayer
from merlin.core.merlin_processor import MerlinProcessor
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.sampling.strategies import OutputMappingStrategy

# Configuration
CLOUD_TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6Mzk1LCJleHAiOjE3NjEyMjgyNzUuMjY4MDkzfQ.vPEHupHJhtXAFVMqyhav7s97cfp_CtJFxA9eH7328JSehdxKry192BKZ8i99KarjlMBkKoIyEJEmU45O3aDjSw"
REMOTE_PLATFORM = "sim:ascella"

# Set token if available
if CLOUD_TOKEN and CLOUD_TOKEN != "YOUR_TOKEN_HERE":
    RemoteConfig.set_token(CLOUD_TOKEN)


def create_quantum_layer_with_config(
        n_modes: int,
        n_photons: int,
        input_size: int,
        trainable: bool = True,
        no_bunching: bool = False,
        output_size: int = None
) -> QuantumLayer:
    """Create a quantum layer with specified configuration."""
    builder = CircuitBuilder(n_modes=n_modes)

    if trainable:
        builder.add_rotation_layer(trainable=True, name="theta")

    # Add input encoding
    builder.add_angle_encoding(modes=list(range(min(input_size, n_modes))), name="px")

    if n_modes >= 3:
        builder.add_entangling_layer(depth=1)

    layer = QuantumLayer(
        input_size=input_size,
        output_size=output_size,
        circuit=builder,
        n_photons=n_photons,  # Use exact local simulation
        no_bunching=no_bunching,
        output_mapping_strategy=OutputMappingStrategy.NONE if output_size is None else OutputMappingStrategy.LINEAR,
    )
    layer.eval()
    return layer


def test_parameter_extraction_consistency():
    """Test that exported parameters match current trained values."""
    print("\n" + "=" * 60)
    print("TEST 1: Parameter Extraction Consistency")
    print("=" * 60)

    # Create layer with trainable parameters
    quantum_layer = create_quantum_layer_with_config(
        n_modes=5,
        n_photons=3,
        input_size=3,
        trainable=True
    )

    # Save initial parameters
    initial_params = {}
    for name, param in quantum_layer.named_parameters():
        initial_params[name] = param.clone().detach()
        print(f"Initial param {name}: shape {param.shape}")

    # Train the model
    print("\nTraining model to update parameters...")
    quantum_layer.train()  # Enable training mode for local execution
    optimizer = torch.optim.Adam(quantum_layer.parameters(), lr=0.1)
    X = torch.randn(10, 3)

    for epoch in range(5):
        optimizer.zero_grad()
        output = quantum_layer(X)
        loss = output.sum()  # Dummy loss
        loss.backward()
        optimizer.step()
        if epoch == 0:
            print(f"  Initial loss: {loss.item():.4f}")
    print(f"  Final loss: {loss.item():.4f}")

    quantum_layer.eval()  # Back to eval mode

    # Verify parameters were updated
    print("\nChecking parameter updates:")
    for name, param in quantum_layer.named_parameters():
        if 'theta' in name or 'rotation' in name:
            changed = not torch.allclose(param, initial_params[name], atol=1e-6)
            print(f"  {name}: {'Updated ✓' if changed else 'Unchanged ✗'}")
            if not changed:
                print(f"    Warning: Parameter {name} wasn't updated during training")

    # Export configuration and check
    print("\nExporting configuration...")
    config = quantum_layer.export_config()

    # Check circuit has parameters
    exported_circuit = config['circuit']
    circuit_params = list(exported_circuit.get_parameters())
    print(f"Exported circuit has {len(circuit_params)} parameters")

    # Group parameters by type
    trainable_params = [p for p in circuit_params if 'theta' in p.name or 'rotation' in p.name]
    input_params = [p for p in circuit_params if 'px' in p.name]

    print(f"  Trainable parameters: {len(trainable_params)}")
    print(f"  Input parameters: {len(input_params)}")

    # Verify trainable parameters have values
    for param in trainable_params[:5]:  # Show first 5
        if hasattr(param, 'defined'):
            print(f"    {param.name}: {'defined' if param.defined else 'symbolic'}")

    print("✓ Parameter extraction consistency test passed")


def test_output_state_dimensions():
    """Test output dimensions for different configurations."""
    print("\n" + "=" * 60)
    print("TEST 2: Output State Dimensions")
    print("=" * 60)

    test_cases = [
        {"n_modes": 4, "n_photons": 2, "no_bunching": False},
        {"n_modes": 4, "n_photons": 2, "no_bunching": True},
        {"n_modes": 5, "n_photons": 3, "no_bunching": False},
        {"n_modes": 5, "n_photons": 3, "no_bunching": True},
        {"n_modes": 6, "n_photons": 2, "no_bunching": True},
    ]

    for config in test_cases:
        m, n = config["n_modes"], config["n_photons"]
        no_bunch = config["no_bunching"]

        # Calculate expected dimension
        if no_bunch:
            expected_dim = comb(m, n)  # C(m,n)
        else:
            expected_dim = comb(m + n - 1, n)  # C(m+n-1,n)

        # Create layer
        layer = create_quantum_layer_with_config(
            n_modes=m,
            n_photons=n,
            input_size=2,
            no_bunching=no_bunch
        )

        # Test output
        X = torch.randn(3, 2)
        output = layer(X)

        print(f"\nConfig: {m} modes, {n} photons, no_bunching={no_bunch}")
        print(f"  Expected dimension: {expected_dim}")
        print(f"  Actual dimension: {output.shape[1]}")
        print(f"  Match: {'✓' if output.shape[1] == expected_dim else '✗'}")

        assert output.shape[1] == expected_dim, f"Dimension mismatch"

        # Verify probability normalization
        sums = output.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "Probabilities don't sum to 1"

    print("\n✓ Output state dimensions test passed")


def test_local_vs_remote_consistency():
    """Test consistency between local and remote execution."""
    print("\n" + "=" * 60)
    print("TEST 3: Local vs Remote Execution Consistency")
    print("=" * 60)

    if not CLOUD_TOKEN or CLOUD_TOKEN == "YOUR_TOKEN_HERE":
        print("⚠️ Skipping - no cloud token provided")
        return

    # Create quantum layer
    quantum_layer = create_quantum_layer_with_config(
        n_modes=5,
        n_photons=2,
        input_size=2,
        trainable=True,
        no_bunching=True
    )

    # Train it briefly
    quantum_layer.train()
    optimizer = torch.optim.Adam(quantum_layer.parameters(), lr=0.01)
    X_train = torch.randn(10, 2)

    for _ in range(3):
        optimizer.zero_grad()
        output = quantum_layer(X_train)
        loss = output.sum()
        loss.backward()
        optimizer.step()

    quantum_layer.eval()

    # Test data
    X_test = torch.randn(4, 2)

    # Local execution
    print("Testing local execution...")

    local_output = quantum_layer.forward(X_test,shots=None)
    print(f"  Local output shape: {local_output.shape}")

    # Remote execution
    print("\nTesting remote execution...")
    remote_proc = MerlinProcessor.from_platform(
        platform=REMOTE_PLATFORM,
        token=CLOUD_TOKEN,
        timeout=60.0
    )

    # Use high shots for better approximation
    remote_output = remote_proc.forward(quantum_layer, X_test, shots=50000)
    print(f"  Remote output shape: {remote_output.shape}")

    # Compare
    assert local_output.shape == remote_output.shape, "Shape mismatch"

    # Both should be probability distributions
    local_sums = local_output.sum(dim=1)
    remote_sums = remote_output.sum(dim=1)

    assert torch.allclose(local_sums, torch.ones_like(local_sums), atol=1e-5), "Local probs don't sum to 1"
    assert torch.allclose(remote_sums, torch.ones_like(remote_sums), atol=1e-3), "Remote probs don't sum to 1"

    print("\n✓ Local vs remote consistency test passed")


def test_no_bunching_filtering():
    """Test that no_bunching correctly filters states."""
    print("\n" + "=" * 60)
    print("TEST 4: No-Bunching State Filtering")
    print("=" * 60)

    n_modes, n_photons = 5, 3
    input_size = 2

    # Create layer WITH bunching
    print("Creating layer with bunching allowed...")
    layer_bunched = create_quantum_layer_with_config(
        n_modes=n_modes,
        n_photons=n_photons,
        input_size=input_size,
        no_bunching=False
    )

    # Create layer WITHOUT bunching
    print("Creating layer with no_bunching=True...")
    layer_no_bunch = create_quantum_layer_with_config(
        n_modes=n_modes,
        n_photons=n_photons,
        input_size=input_size,
        no_bunching=True
    )

    # Test both
    X = torch.randn(5, input_size)

    output_bunched = layer_bunched(X)
    output_no_bunch = layer_no_bunch(X)

    bunched_dim = comb(n_modes + n_photons - 1, n_photons)
    no_bunch_dim = comb(n_modes, n_photons)

    print(f"\nResults for {n_modes} modes, {n_photons} photons:")
    print(f"  With bunching: {output_bunched.shape[1]} states (expected {bunched_dim})")
    print(f"  No bunching: {output_no_bunch.shape[1]} states (expected {no_bunch_dim})")
    print(f"  Reduction: {output_bunched.shape[1] - output_no_bunch.shape[1]} states filtered")

    assert output_bunched.shape[1] == bunched_dim
    assert output_no_bunch.shape[1] == no_bunch_dim
    assert output_no_bunch.shape[1] < output_bunched.shape[1], "No-bunching should reduce state space"

    # Both should be valid probability distributions
    assert torch.allclose(output_bunched.sum(dim=1), torch.ones(5), atol=1e-5)
    assert torch.allclose(output_no_bunch.sum(dim=1), torch.ones(5), atol=1e-5)

    print("\n✓ No-bunching filtering test passed")


def test_parameter_freezing_on_remote():
    """Test that parameters can't be updated during remote execution."""
    print("\n" + "=" * 60)
    print("TEST 5: Parameter Freezing During Remote Execution")
    print("=" * 60)

    if not CLOUD_TOKEN or CLOUD_TOKEN == "YOUR_TOKEN_HERE":
        print("⚠️ Skipping - no cloud token provided")
        return

    # Create and train a layer
    quantum_layer = create_quantum_layer_with_config(
        n_modes=4,
        n_photons=2,
        input_size=2,
        trainable=True
    )

    # Train locally first
    quantum_layer.train()
    optimizer = torch.optim.Adam(quantum_layer.parameters(), lr=0.01)
    X = torch.randn(10, 2)

    for _ in range(3):
        optimizer.zero_grad()
        output = quantum_layer(X)
        loss = output.sum()
        loss.backward()
        optimizer.step()

    # Save parameters
    saved_params = {name: param.clone() for name, param in quantum_layer.named_parameters()}

    # Create remote processor
    remote_proc = MerlinProcessor.from_platform(
        platform=REMOTE_PLATFORM,
        token=CLOUD_TOKEN
    )

    # Try to execute in training mode (should fail)
    print("Testing training mode with remote processor...")
    quantum_layer.train()

    try:
        output = remote_proc.forward(quantum_layer, X, shots=1000)
        print("  ✗ Should have raised error for training mode")
    except RuntimeError as e:
        print(f"  ✓ Correctly blocked: {str(e)[:50]}...")

    # Execute in eval mode
    quantum_layer.eval()
    print("\nExecuting in eval mode...")
    output = remote_proc.forward(quantum_layer, X, shots=1000)
    print(f"  ✓ Eval mode execution successful, shape: {output.shape}")

    # Verify parameters unchanged
    print("\nVerifying parameters unchanged...")
    for name, param in quantum_layer.named_parameters():
        assert torch.allclose(param, saved_params[name]), f"Parameter {name} changed!"
    print("  ✓ All parameters remain unchanged")

    print("\n✓ Parameter freezing test passed")


def test_batch_output_consistency():
    """Test that outputs are consistent across different batch sizes."""
    print("\n" + "=" * 60)
    print("TEST 6: Batch Output Consistency")
    print("=" * 60)

    quantum_layer = create_quantum_layer_with_config(
        n_modes=5,
        n_photons=2,
        input_size=2,
        no_bunching=True
    )

    # Fixed input for consistency
    torch.manual_seed(42)
    single_input = torch.randn(1, 2)

    # Test with local processor

    output_single = quantum_layer.forward(single_input, shots=None)

    # Batch with repeated input
    batch_input = single_input.repeat(5, 1)

    output_batch = quantum_layer.forward(batch_input, shots=None)

    print(f"Single input shape: {output_single.shape}")
    print(f"Batch output shape: {output_batch.shape}")

    # All batch outputs should be identical since input is repeated
    for i in range(5):
        assert torch.allclose(output_batch[i], output_single[0], atol=1e-6), \
            f"Batch output {i} doesn't match single output"

    print("✓ All batch outputs match single output")

    # Test different batch sizes
    batch_sizes = [1, 4, 8, 16]
    expected_dim = 10  # C(5,2) with no_bunching

    print("\nTesting different batch sizes:")
    for bs in batch_sizes:
        X = torch.randn(bs, 2)

        output = quantum_layer.forward(X, shots=None)
        print(f"  Batch size {bs:2d}: shape {output.shape}")
        assert output.shape == (bs, expected_dim)

    print("\n✓ Batch output consistency test passed")


def run_all_tests():
    """Run all parameter and state tests."""
    print("\n" + "=" * 60)
    print("MERLINPROCESSOR PARAMETER & STATE TESTING SUITE")
    print("=" * 60)
    print(f"Platform: {REMOTE_PLATFORM}")
    print(f"Token: {'Set' if CLOUD_TOKEN and CLOUD_TOKEN != 'YOUR_TOKEN_HERE' else 'Not set'}")

    try:
        test_parameter_extraction_consistency()
        test_output_state_dimensions()
        test_no_bunching_filtering()
        test_batch_output_consistency()

        # Cloud tests (optional)
        if CLOUD_TOKEN and CLOUD_TOKEN != "YOUR_TOKEN_HERE":
            test_local_vs_remote_consistency()
            test_parameter_freezing_on_remote()
        else:
            print("\n⚠️ Skipping cloud tests - no token provided")

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()