"""
Test suite for cloud processor parameter handling and state indexing.
Verifies that trained parameters are correctly extracted and that output states
are consistently indexed across different circuit configurations.
"""

import torch
import torch.nn as nn
import numpy as np
import perceval as pcvl
from typing import Dict, List, Tuple
import warnings

from merlin.algorithms import QuantumLayer
from merlin.core.cloud_processor import CloudProcessor, deploy_to_cloud
from merlin.sampling.strategies import OutputMappingStrategy

# Configuration
# Configuration
CLOUD_TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6Mzk1LCJleHAiOjE3NjEyMjgyNzUuMjY4MDkzfQ.vPEHupHJhtXAFVMqyhav7s97cfp_CtJFxA9eH7328JSehdxKry192BKZ8i99KarjlMBkKoIyEJEmU45O3aDjSw"  # Replace with your Quandela Cloud token

REMOTE_PLATFORM = "sim:ascella"

"""
Test suite for cloud processor parameter handling and state indexing.
Verifies that trained parameters are correctly extracted and that output states
are consistently indexed across different circuit configurations.
"""

import torch
import torch.nn as nn
import numpy as np
import perceval as pcvl
from typing import Dict, List, Tuple
import warnings

from merlin.algorithms import QuantumLayer
from merlin.core.cloud_processor import CloudProcessor, deploy_to_cloud
from merlin.sampling.strategies import OutputMappingStrategy




def create_circuit_with_mixed_params(m: int, config: Dict) -> pcvl.Circuit:
    """
    Create circuit with mixed parameter types based on config.

    Config keys:
    - fixed_left: bool - Use fixed values for left interferometer
    - fixed_right: bool - Use fixed values for right interferometer
    - n_inputs: int - Number of input parameters (max 4)
    - trainable_prefix: str - Prefix for trainable parameters
    """
    c = pcvl.Circuit(m)

    # Left interferometer
    if config.get('fixed_left', False):
        wl = pcvl.GenericInterferometer(
            m,
            lambda i: pcvl.BS() // pcvl.PS(np.random.rand() * np.pi) //
                      pcvl.BS() // pcvl.PS(np.random.rand() * np.pi),
            shape=pcvl.InterferometerShape.RECTANGLE
        )
    else:
        prefix = config.get('trainable_prefix', 'theta')
        wl = pcvl.GenericInterferometer(
            m,
            lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"{prefix}_li{i}")) //
                      pcvl.BS() // pcvl.PS(pcvl.P(f"{prefix}_lo{i}")),
            shape=pcvl.InterferometerShape.RECTANGLE
        )

    # Input parameters
    c_var = pcvl.Circuit(m)
    n_inputs = config.get('n_inputs', 4)
    for i in range(min(n_inputs, m)):
        px = pcvl.P(f"input_{i + 1}")
        c_var.add(i + (m - min(n_inputs, m)) // 2, pcvl.PS(px))

    # Right interferometer
    if config.get('fixed_right', False):
        wr = pcvl.GenericInterferometer(
            m,
            lambda i: pcvl.BS() // pcvl.PS(np.random.rand() * np.pi) //
                      pcvl.BS() // pcvl.PS(np.random.rand() * np.pi),
            shape=pcvl.InterferometerShape.RECTANGLE
        )
    else:
        prefix = config.get('trainable_prefix', 'theta')
        wr = pcvl.GenericInterferometer(
            m,
            lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"{prefix}_ri{i}")) //
                      pcvl.BS() // pcvl.PS(pcvl.P(f"{prefix}_ro{i}")),
            shape=pcvl.InterferometerShape.RECTANGLE
        )

    c.add(0, wl, merge=True)
    c.add(0, c_var, merge=True)
    c.add(0, wr, merge=True)

    return c


def test_parameter_extraction_consistency():
    """Test that exported parameters match current trained values."""
    print("\n" + "=" * 60)
    print("TEST 1: Parameter Extraction Consistency")
    print("=" * 60)

    # Create circuit with known trainable parameters
    circuit = create_circuit_with_mixed_params(5, {
        'fixed_left': False,
        'fixed_right': False,
        'n_inputs': 3,
        'trainable_prefix': 'theta'
    })

    quantum_layer = QuantumLayer(
        input_size=3,
        output_size=None,
        circuit=circuit,
        trainable_parameters=["theta"],
        input_parameters=["input"],
        input_state=[1, 1, 1, 0, 0],
        output_mapping_strategy=OutputMappingStrategy.NONE,
        shots=0
    )

    # Train the model
    print("Training model to update parameters...")
    optimizer = torch.optim.Adam(quantum_layer.parameters(), lr=0.1)
    X = torch.randn(10, 3)

    initial_params = {}
    for name, param in quantum_layer.named_parameters():
        initial_params[name] = param.clone().detach()

    for _ in range(5):
        optimizer.zero_grad()
        output = quantum_layer(X)
        loss = output.sum()  # Dummy loss
        loss.backward()
        optimizer.step()

    # Export configuration and check parameters
    print("Exporting configuration...")
    config = quantum_layer.export_config()
    exported_circuit = config['circuit']

    # Verify trainable parameters were updated
    for name, param in quantum_layer.named_parameters():
        if 'theta' in name:
            assert not torch.allclose(param, initial_params[name]), f"Parameter {name} wasn't updated"

    # Check that exported circuit has the trained values
    circuit_params = {p.name: p for p in exported_circuit.get_parameters()}

    # Verify theta parameters have values (not symbolic)
    for p_name, p in circuit_params.items():
        if p_name.startswith('theta'):
            assert p.defined, f"Trainable parameter {p_name} should have a value"
            # Use float(p) to get the value from Perceval Parameter
            print(f"  {p_name}: {float(p):.4f}")
        elif p_name.startswith('input'):
            assert not p.defined, f"Input parameter {p_name} should remain symbolic"
            print(f"  {p_name}: [symbolic]")

    print("✓ Parameter extraction consistency test passed")


def test_mixed_circuit_configurations():
    """Test various combinations of fixed/trainable/input parameters."""
    print("\n" + "=" * 60)
    print("TEST 2: Mixed Circuit Configurations")
    print("=" * 60)

    test_configs = [
        {
            'name': 'Fully trainable',
            'config': {'fixed_left': False, 'fixed_right': False, 'n_inputs': 2},
            'trainable': ['theta'],
            'input': ['input']
        },
        {
            'name': 'Fixed left, trainable right',
            'config': {'fixed_left': True, 'fixed_right': False, 'n_inputs': 3},
            'trainable': ['theta'],
            'input': ['input']
        },
        {
            'name': 'All fixed except inputs',
            'config': {'fixed_left': True, 'fixed_right': True, 'n_inputs': 4},
            'trainable': [],
            'input': ['input']
        },
        {
            'name': 'Mixed with custom prefix',
            'config': {'fixed_left': False, 'fixed_right': True, 'n_inputs': 2, 'trainable_prefix': 'phi'},
            'trainable': ['phi'],
            'input': ['input']
        }
    ]

    for test_case in test_configs:
        print(f"\nTesting: {test_case['name']}")

        circuit = create_circuit_with_mixed_params(6, test_case['config'])

        quantum_layer = QuantumLayer(
            input_size=test_case['config']['n_inputs'],
            output_size=None,
            circuit=circuit,
            trainable_parameters=test_case['trainable'],
            input_parameters=test_case['input'],
            input_state=[1] * test_case['config']['n_inputs'] + [0] * (6 - test_case['config']['n_inputs']),
            output_mapping_strategy=OutputMappingStrategy.NONE,
            shots=0
        )

        # Quick training if there are trainable parameters
        if test_case['trainable']:
            optimizer = torch.optim.Adam(quantum_layer.parameters(), lr=0.01)
            X = torch.randn(5, test_case['config']['n_inputs'])
            for _ in range(3):
                optimizer.zero_grad()
                output = quantum_layer(X)
                loss = output.sum()
                loss.backward()
                optimizer.step()

        # Export and verify
        config = quantum_layer.export_config()
        exported_circuit = config['circuit']
        circuit_params = {p.name: p for p in exported_circuit.get_parameters()}

        # Count parameter types
        n_defined = sum(1 for p in circuit_params.values() if p.defined)
        n_symbolic = sum(1 for p in circuit_params.values() if not p.defined)

        print(f"  Defined parameters: {n_defined}")
        print(f"  Symbolic parameters: {n_symbolic}")

        # Verify input parameters remain symbolic
        for p_name in circuit_params:
            if p_name.startswith('input'):
                assert not circuit_params[p_name].defined, f"Input {p_name} should be symbolic"

        print(f"  ✓ Configuration validated")

    print("\n✓ Mixed circuit configurations test passed")


def test_output_state_indexing():
    """Test that output state indexing is consistent."""
    print("\n" + "=" * 60)
    print("TEST 3: Output State Indexing Consistency")
    print("=" * 60)

    # Create a small circuit for predictable output states
    circuit = create_circuit_with_mixed_params(4, {
        'fixed_left': False,
        'fixed_right': False,
        'n_inputs': 2,
        'trainable_prefix': 'theta'
    })

    quantum_layer = QuantumLayer(
        input_size=2,
        output_size=None,
        circuit=circuit,
        trainable_parameters=["theta"],
        input_parameters=["input"],
        input_state=[1, 1, 0, 0],  # 2 photons in 4 modes
        output_mapping_strategy=OutputMappingStrategy.NONE,
        no_bunching=False,  # Allow bunched states
        shots=0
    )

    # Run multiple times and verify consistent output size
    X = torch.randn(3, 2)

    print("Testing output consistency across multiple runs...")
    outputs = []
    for i in range(5):
        output = quantum_layer(X)
        outputs.append(output)
        print(f"  Run {i + 1}: Output shape {output.shape}, sum={output.sum(dim=1)}")

    # Verify all outputs have same shape
    for i in range(1, len(outputs)):
        assert outputs[i].shape == outputs[0].shape, f"Inconsistent output shape at run {i + 1}"

    # Verify probability normalization
    for output in outputs:
        sums = output.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "Probabilities don't sum to 1"

    print("✓ Output state indexing consistency test passed")


def test_no_bunching_state_filtering():
    """Test that no_bunching correctly filters output states."""
    print("\n" + "=" * 60)
    print("TEST 4: No-Bunching State Filtering")
    print("=" * 60)

    circuit = create_circuit_with_mixed_params(4, {
        'fixed_left': True,
        'fixed_right': True,
        'n_inputs': 2
    })

    # Test with bunching allowed
    print("\nWith bunching allowed:")
    layer_bunched = QuantumLayer(
        input_size=2,
        output_size=None,
        circuit=circuit,
        trainable_parameters=[],
        input_parameters=["input"],
        input_state=[1, 1, 0, 0],
        output_mapping_strategy=OutputMappingStrategy.NONE,
        no_bunching=False,
        shots=0
    )

    X = torch.randn(5, 2)
    output_bunched = layer_bunched(X)
    print(f"  Output dimension (bunched): {output_bunched.shape[1]}")

    # Test with no_bunching
    print("\nWith no_bunching:")
    layer_no_bunch = QuantumLayer(
        input_size=2,
        output_size=None,
        circuit=circuit,
        trainable_parameters=[],
        input_parameters=["input"],
        input_state=[1, 1, 0, 0],
        output_mapping_strategy=OutputMappingStrategy.NONE,
        no_bunching=True,
        shots=0
    )

    output_no_bunch = layer_no_bunch(X)
    print(f"  Output dimension (no bunching): {output_no_bunch.shape[1]}")

    # No-bunching should have fewer states
    assert output_no_bunch.shape[1] <= output_bunched.shape[1], \
        "No-bunching should not increase state space"

    # Verify probabilities still sum to 1
    sums = output_no_bunch.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        "No-bunching probabilities don't sum to 1"

    print("✓ No-bunching state filtering test passed")


def test_cloud_state_indexing_consistency():
    """Test that cloud and local execution produce consistent state indexing."""
    print("\n" + "=" * 60)
    print("TEST 5: Cloud vs Local State Indexing")
    print("=" * 60)

    if CLOUD_TOKEN == "YOUR_TOKEN_HERE":
        print("⚠️ Skipping cloud test - no token provided")
        return

    circuit = create_circuit_with_mixed_params(5, {
        'fixed_left': False,
        'fixed_right': False,
        'n_inputs': 3
    })

    quantum_layer = QuantumLayer(
        input_size=3,
        output_size=None,
        circuit=circuit,
        trainable_parameters=["theta"],
        input_parameters=["input"],
        input_state=[1, 1, 1, 0, 0],
        output_mapping_strategy=OutputMappingStrategy.NONE,
        shots=1000  # Use shots for cloud to get actual results
    )

    # Train briefly
    optimizer = torch.optim.Adam(quantum_layer.parameters(), lr=0.01)
    X_train = torch.randn(10, 3)
    for _ in range(3):
        optimizer.zero_grad()
        output = quantum_layer(X_train)
        loss = output.sum()
        loss.backward()
        optimizer.step()

    # Get local output
    quantum_layer.eval()
    X_test = torch.randn(3, 3)
    with torch.no_grad():
        # Use shots=0 for deterministic local output
        quantum_layer.shots = 0
        local_output = quantum_layer(X_test)
        quantum_layer.shots = 1000  # Reset for cloud

    print(f"Local output shape: {local_output.shape}")
    print(f"Local output sample:\n{local_output[0][:10]}")  # First 10 states

    # Deploy to cloud
    print("\nDeploying to cloud...")
    cloud_proc = deploy_to_cloud(
        quantum_layer,
        platform=REMOTE_PLATFORM,
        token=CLOUD_TOKEN,
        wait_timeout=120
    )

    # Get cloud output
    with torch.no_grad():
        cloud_output = quantum_layer(X_test)

    print(f"Cloud output shape: {cloud_output.shape}")
    print(f"Cloud output sample:\n{cloud_output[0][:10]}")

    # Compare shapes
    assert local_output.shape == cloud_output.shape, \
        f"Shape mismatch: local {local_output.shape} vs cloud {cloud_output.shape}"

    # Verify cloud output is not all zeros
    assert not torch.allclose(cloud_output, torch.zeros_like(cloud_output)), \
        "Cloud output is all zeros - execution may have failed"

    # Both should sum to 1 (probability distributions)
    local_sums = local_output.sum(dim=1)
    cloud_sums = cloud_output.sum(dim=1)
    assert torch.allclose(local_sums, torch.ones_like(local_sums), atol=1e-3), \
        "Local probabilities don't sum to 1"
    assert torch.allclose(cloud_sums, torch.ones_like(cloud_sums), atol=1e-3), \
        "Cloud probabilities don't sum to 1"

    print("✓ Cloud vs local state indexing test passed")


def test_parameter_update_after_cloud_deployment():
    """Test that parameters can't be updated after cloud deployment."""
    print("\n" + "=" * 60)
    print("TEST 6: Parameter Immutability After Cloud Deployment")
    print("=" * 60)

    if CLOUD_TOKEN == "YOUR_TOKEN_HERE":
        print("⚠️ Skipping cloud test - no token provided")
        return

    circuit = create_circuit_with_mixed_params(4, {
        'fixed_left': False,
        'fixed_right': False,
        'n_inputs': 2
    })

    quantum_layer = QuantumLayer(
        input_size=2,
        output_size=None,
        circuit=circuit,
        trainable_parameters=["theta"],
        input_parameters=["input"],
        input_state=[1, 1, 0, 0],
        output_mapping_strategy=OutputMappingStrategy.NONE,
        shots=0
    )

    # Train and save parameters
    optimizer = torch.optim.Adam(quantum_layer.parameters(), lr=0.01)
    X = torch.randn(10, 2)
    for _ in range(3):
        optimizer.zero_grad()
        output = quantum_layer(X)
        loss = output.sum()
        loss.backward()
        optimizer.step()

    # Save current parameters
    saved_params = {name: param.clone() for name, param in quantum_layer.named_parameters()}

    # Deploy to cloud
    print("Deploying to cloud...")
    cloud_proc = deploy_to_cloud(
        quantum_layer,
        platform=REMOTE_PLATFORM,
        token=CLOUD_TOKEN
    )

    # Verify training mode is blocked
    quantum_layer.train()
    try:
        output = quantum_layer(X)
        assert False, "Should have raised error in training mode"
    except RuntimeError as e:
        print("✓ Training mode correctly blocked after deployment")

    # Verify parameters haven't changed
    quantum_layer.eval()
    for name, param in quantum_layer.named_parameters():
        assert torch.allclose(param, saved_params[name]), \
            f"Parameter {name} changed after cloud deployment"

    print("✓ Parameter immutability test passed")


def run_all_cloud_processor_tests():
    """Run all cloud processor tests."""
    print("\n" + "=" * 60)
    print("CLOUD PROCESSOR PARAMETER & STATE TESTING SUITE")
    print("=" * 60)

    try:
        test_parameter_extraction_consistency()
        test_mixed_circuit_configurations()
        test_output_state_indexing()
        test_no_bunching_state_filtering()
        test_cloud_state_indexing_consistency()
        test_parameter_update_after_cloud_deployment()

        print("\n" + "=" * 60)
        print("ALL CLOUD PROCESSOR TESTS PASSED ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_cloud_processor_tests()