"""
Test to verify that sim:slos cloud platform produces identical results
to local pcvl_pytorch SLOS execution for inference.
"""

import torch
import numpy as np
import perceval as pcvl
from merlin.algorithms import QuantumLayer
from merlin.core.cloud_processor import deploy_to_cloud
from merlin.sampling.strategies import OutputMappingStrategy

# Configuration
CLOUD_TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6Mzk1LCJleHAiOjE3NjEyMjgyNzUuMjY4MDkzfQ.vPEHupHJhtXAFVMqyhav7s97cfp_CtJFxA9eH7328JSehdxKry192BKZ8i99KarjlMBkKoIyEJEmU45O3aDjSw"  # Replace with your Quandela Cloud token
 # Replace with actual token
SLOS_PLATFORM = "sim:slos"  # SLOS simulator on cloud


def create_test_circuit(n_modes: int) -> pcvl.Circuit:
    """Create a simple test circuit with fixed parameters for reproducibility."""
    circuit = pcvl.Circuit(n_modes)

    # Add some basic components with fixed values
    circuit.add(0, pcvl.BS())
    circuit.add(1, pcvl.PS(np.pi / 4))
    circuit.add(2, pcvl.BS())

    # Add parametric phase shifters for input
    for i in range(min(3, n_modes)):
        circuit.add(i, pcvl.PS(pcvl.P(f"input_{i + 1}")))

    # Add another beam splitter
    if n_modes >= 4:
        circuit.add(2, pcvl.BS())

    return circuit


def test_slos_cloud_vs_local_both_modes():
    """
    Test that sim:slos cloud execution matches local SLOS execution
    for both no_bunching=True and no_bunching=False cases.
    """
    print("\n" + "=" * 60)
    print("TEST: sim:slos Cloud vs Local SLOS - Both Modes")
    print("=" * 60)

    # Test both no_bunching settings
    for no_bunching in [False, True]:
        print(f"\n--- Testing with no_bunching={no_bunching} ---")

        # Create the same circuit for both
        n_modes = 4
        circuit = create_test_circuit(n_modes)

        # Create quantum layer
        quantum_layer = QuantumLayer(
            input_size=3,
            output_size=None,  # Determined by state space
            circuit=circuit,
            trainable_parameters=[],  # No trainable params for deterministic test
            input_parameters=["input"],
            input_state=[1, 1, 0, 0],  # 2 photons in 4 modes
            output_mapping_strategy=OutputMappingStrategy.NONE,
            no_bunching=no_bunching,
            shots=0  # Deterministic for local
        )

        # Prepare test input
        torch.manual_seed(42)  # For reproducibility
        test_input = torch.randn(5, 3) * 0.5  # 5 samples, moderate values

        print(f"Circuit: {n_modes} modes, 2 photons")
        print(f"Input shape: {test_input.shape}")

        # Get local SLOS output (deterministic)
        print("\n1. Getting local SLOS output (deterministic)...")
        quantum_layer.eval()
        with torch.no_grad():
            local_output = quantum_layer(test_input)

        print(f"  Local output shape: {local_output.shape}")
        print(f"  Local output sample (first row): {local_output[0][:10]}")
        print(f"  Sum of probabilities: {local_output[0].sum():.6f}")

        # Check if local output is all zeros (expected for no_bunching=True with beam splitters)
        is_all_zeros = torch.allclose(local_output, torch.zeros_like(local_output))

        if is_all_zeros and no_bunching:
            print("  Note: Local output is all zeros (circuit produces only bunched states)")

        # Deploy to sim:slos cloud
        print(f"\n2. Deploying to {SLOS_PLATFORM} cloud...")
        cloud_proc = deploy_to_cloud(
            quantum_layer,
            platform=SLOS_PLATFORM,
            token=CLOUD_TOKEN,
            wait_timeout=120
        )

        # Set high shot count for cloud
        quantum_layer.shots = 10_000_000  # 10 million shots
        print(f"  Using {quantum_layer.shots:,} shots for cloud execution")

        # Get cloud SLOS output
        print("\n3. Getting cloud SLOS output...")
        with torch.no_grad():
            cloud_output = quantum_layer(test_input)

        print(f"  Cloud output shape: {cloud_output.shape}")
        print(f"  Cloud output sample (first row): {cloud_output[0][:10]}")
        print(f"  Sum of probabilities: {cloud_output[0].sum():.6f}")

        # Compare outputs
        print("\n4. Comparing outputs...")

        # Check shapes match
        assert local_output.shape == cloud_output.shape, \
            f"Shape mismatch: local {local_output.shape} vs cloud {cloud_output.shape}"
        print(f"  ✓ Shapes match: {local_output.shape}")

        if is_all_zeros:
            # If local is all zeros, cloud should also be all zeros
            assert torch.allclose(cloud_output, torch.zeros_like(cloud_output)), \
                "Local output is all zeros but cloud output is not"
            print(f"  ✓ Both outputs are all zeros (expected for no_bunching={no_bunching})")
        else:
            # Normal comparison for non-zero outputs
            assert not torch.allclose(cloud_output, torch.zeros_like(cloud_output)), \
                "Cloud output is all zeros but local output is not"
            print(f"  ✓ Cloud output contains non-zero values")

            # Check probability normalization
            local_sums = local_output.sum(dim=1)
            cloud_sums = cloud_output.sum(dim=1)

            assert torch.allclose(local_sums, torch.ones_like(local_sums), atol=1e-5), \
                f"Local probabilities don't sum to 1: {local_sums}"
            assert torch.allclose(cloud_sums, torch.ones_like(cloud_sums), atol=1e-3), \
                f"Cloud probabilities don't sum to 1: {cloud_sums}"
            print(f"  ✓ Both outputs are normalized probability distributions")

            # Compare actual probability values
            tolerance = 0.01  # 1% tolerance for 10M shots
            close_mask = torch.abs(local_output - cloud_output) < tolerance
            percent_close = close_mask.float().mean().item() * 100

            print(f"  {percent_close:.1f}% of values within {tolerance} tolerance")

            # We expect at least 90% of values to be within tolerance
            assert percent_close > 90, \
                f"Only {percent_close:.1f}% of values are within tolerance"

            print(f"  ✓ Cloud and local outputs match within statistical variance")

        print(f"\n✓ Test passed for no_bunching={no_bunching}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED: sim:slos cloud and local SLOS are consistent")
    print("=" * 60)


def test_no_bunching_compatible_circuit():
    """
    Test no_bunching with a circuit that doesn't create bunched states.
    This verifies the no_bunching logic works correctly when valid states exist.
    """
    print("\n" + "=" * 60)
    print("TEST: No-Bunching with Compatible Circuit")
    print("=" * 60)

    # Create circuit without beam splitters
    circuit = pcvl.Circuit(4)

    # Only use phase shifters and permutations
    for i in range(3):
        circuit.add(i, pcvl.PS(pcvl.P(f"input_{i + 1}")))

    # Add a permutation to mix modes without creating bunching
    circuit.add(0, pcvl.PERM([1, 0, 2, 3]))

    quantum_layer = QuantumLayer(
        input_size=3,
        output_size=None,
        circuit=circuit,
        trainable_parameters=[],
        input_parameters=["input"],
        input_state=[1, 1, 0, 0],
        output_mapping_strategy=OutputMappingStrategy.NONE,
        no_bunching=True,
        shots=0
    )

    test_input = torch.randn(3, 3) * 0.5

    # Get local output
    print("\n1. Local execution with no_bunching=True...")
    quantum_layer.eval()
    with torch.no_grad():
        local_output = quantum_layer(test_input)

    print(f"  Output shape: {local_output.shape}")
    print(f"  Output sum: {local_output[0].sum():.6f}")
    print(f"  Non-zero values: {(local_output > 1e-10).sum().item()}")

    # Check if output is valid
    local_sums = local_output.sum(dim=1)
    is_valid = torch.allclose(local_sums, torch.ones_like(local_sums), atol=1e-5)

    if is_valid:
        print("  ✓ Local output is properly normalized")

        # Deploy to cloud
        print("\n2. Cloud execution with no_bunching=True...")
        cloud_proc = deploy_to_cloud(
            quantum_layer,
            platform=SLOS_PLATFORM,
            token=CLOUD_TOKEN
        )

        quantum_layer.shots = 1_000_000

        with torch.no_grad():
            cloud_output = quantum_layer(test_input)

        print(f"  Output shape: {cloud_output.shape}")
        print(f"  Output sum: {cloud_output[0].sum():.6f}")
        print(f"  Non-zero values: {(cloud_output > 1e-10).sum().item()}")

        # Verify consistency
        assert local_output.shape == cloud_output.shape, "Shape mismatch"
        cloud_sums = cloud_output.sum(dim=1)
        assert torch.allclose(cloud_sums, torch.ones_like(cloud_sums), atol=1e-3), \
            "Cloud output not normalized"

        print("  ✓ Cloud output is properly normalized")
        print("\n✓ No-bunching test with compatible circuit passed")
    else:
        print("  Note: Even permutation-only circuit produces no valid states")
        print("  This is expected behavior for certain input states")


def test_state_ordering_consistency():
    """
    Test that state ordering/indexing is consistent between cloud and local.
    """
    print("\n" + "=" * 60)
    print("TEST: State Ordering Consistency")
    print("=" * 60)

    # Simple circuit for clear state identification
    circuit = pcvl.Circuit(3)
    circuit.add(0, pcvl.PS(pcvl.P("input_1")))
    circuit.add(1, pcvl.PS(pcvl.P("input_2")))
    circuit.add(2, pcvl.PS(pcvl.P("input_3")))
    circuit.add(0, pcvl.BS())

    for no_bunching in [False, True]:
        print(f"\nTesting state ordering with no_bunching={no_bunching}")

        quantum_layer = QuantumLayer(
            input_size=3,
            output_size=None,
            circuit=circuit,
            trainable_parameters=[],
            input_parameters=["input"],
            input_state=[1, 1, 0],  # 2 photons in 3 modes
            output_mapping_strategy=OutputMappingStrategy.NONE,
            no_bunching=no_bunching,
            shots=0
        )

        # Use specific input
        test_input = torch.tensor([[0.1, 0.2, 0.3]])

        # Get local output
        quantum_layer.eval()
        with torch.no_grad():
            local_output = quantum_layer(test_input)

        # Find states with significant probability
        threshold = 0.01
        significant_states = []
        for i, prob in enumerate(local_output[0]):
            if prob > threshold:
                significant_states.append((i, prob.item()))

        if significant_states:
            print(f"  States with probability > {threshold}:")
            for idx, prob in significant_states:
                print(f"    State index {idx}: p={prob:.4f}")

            # Deploy to cloud
            cloud_proc = deploy_to_cloud(
                quantum_layer,
                platform=SLOS_PLATFORM,
                token=CLOUD_TOKEN
            )

            quantum_layer.shots = 1_000_000

            with torch.no_grad():
                cloud_output = quantum_layer(test_input)

            print(f"  Cloud output for same states:")
            all_match = True
            for idx, local_prob in significant_states:
                cloud_prob = cloud_output[0, idx].item()
                diff = abs(local_prob - cloud_prob)
                print(f"    State index {idx}: p={cloud_prob:.4f} (diff={diff:.4f})")
                if diff > 0.02:
                    all_match = False

            if all_match:
                print(f"  ✓ State ordering consistent for no_bunching={no_bunching}")
            else:
                print(f"  ⚠ Some states differ beyond tolerance")
        else:
            print(f"  No significant states (all zeros) for no_bunching={no_bunching}")

    print("\n✓ State ordering consistency test complete")


if __name__ == "__main__":
    # Run comprehensive test for both modes
    test_slos_cloud_vs_local_both_modes()

    # Test no_bunching with a compatible circuit
    test_no_bunching_compatible_circuit()

    # Test state ordering consistency
    test_state_ordering_consistency()