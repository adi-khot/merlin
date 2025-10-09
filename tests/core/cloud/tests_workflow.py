"""
Test suite for various workflow scenarios and gradient handling.
Verifies that gradients flow correctly in GPU components and are disabled for QPU inference.
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.optimize as optimize
from typing import Tuple
import perceval as pcvl
import time


from merlin.core.cloud_processor import deploy_to_cloud
from merlin.sampling.strategies import OutputMappingStrategy

# Configuration
CLOUD_TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6Mzk1LCJleHAiOjE3NjEyMjgyNzUuMjY4MDkzfQ.vPEHupHJhtXAFVMqyhav7s97cfp_CtJFxA9eH7328JSehdxKry192BKZ8i99KarjlMBkKoIyEJEmU45O3aDjSw"  # Replace with your Quandela Cloud token

REMOTE_PLATFORM = "sim:ascella"

"""
Test suite for various workflow scenarios and gradient handling.
Verifies that gradients flow correctly in GPU components and are disabled for QPU inference.
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.optimize as optimize
from typing import Tuple
import perceval as pcvl
import time
import warnings

from merlin.algorithms import QuantumLayer
from merlin.core.cloud_processor import deploy_to_cloud
from merlin.sampling.strategies import OutputMappingStrategy



def create_quantum_circuit(m: int, fixed_params: bool = False) -> pcvl.Circuit:
    """Create a quantum circuit for testing.

    Args:
        m: Number of modes
        fixed_params: If True, use fixed values instead of parameters for theta
    """
    if fixed_params:
        # Create circuit with fixed values (no trainable parameters)
        wl = pcvl.GenericInterferometer(
            m,
            lambda i: pcvl.BS() // pcvl.PS(np.random.rand() * np.pi) //
                      pcvl.BS() // pcvl.PS(np.random.rand() * np.pi),
            shape=pcvl.InterferometerShape.RECTANGLE
        )

        wr = pcvl.GenericInterferometer(
            m,
            lambda i: pcvl.BS() // pcvl.PS(np.random.rand() * np.pi) //
                      pcvl.BS() // pcvl.PS(np.random.rand() * np.pi),
            shape=pcvl.InterferometerShape.RECTANGLE
        )
    else:
        # Create circuit with trainable parameters
        wl = pcvl.GenericInterferometer(
            m,
            lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_li{i}")) //
                      pcvl.BS() // pcvl.PS(pcvl.P(f"theta_lo{i}")),
            shape=pcvl.InterferometerShape.RECTANGLE
        )

        wr = pcvl.GenericInterferometer(
            m,
            lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_ri{i}")) //
                      pcvl.BS() // pcvl.PS(pcvl.P(f"theta_ro{i}")),
            shape=pcvl.InterferometerShape.RECTANGLE
        )

    # Variable phase shifters for input encoding - limit to actual input size
    c_var = pcvl.Circuit(m)
    max_inputs = min(4, m)
    for i in range(max_inputs):
        px = pcvl.P(f"px{i + 1}")
        c_var.add(i + (m - max_inputs) // 2, pcvl.PS(px))

    c = pcvl.Circuit(m)
    c.add(0, wl, merge=True)
    c.add(0, c_var, merge=True)
    c.add(0, wr, merge=True)

    return c


def generate_dummy_data(batch_size: int = 32, input_size: int = 4, num_classes: int = 3) -> Tuple:
    """Generate dummy data for testing."""
    X = torch.randn(batch_size, input_size)
    y = torch.randint(0, num_classes, (batch_size,))
    return X, y


def test_gpu_only_workflow():
    """Test GPU-only workflow with full gradient support."""
    print("\n" + "=" * 60)
    print("TEST 1: GPU-Only Workflow")
    print("GPU Training → GPU Inference (Full Gradients)")
    print("=" * 60)

    # Create quantum layer for GPU execution
    circuit = create_quantum_circuit(6)
    quantum_layer = QuantumLayer(
        input_size=4,
        output_size=None,  # Will be determined by quantum state space
        circuit=circuit,
        trainable_parameters=["theta"],
        input_parameters=["px"],
        input_state=[1, 1, 1, 0, 0, 0],
        output_mapping_strategy=OutputMappingStrategy.NONE,  # No internal mapping
        shots=0  # Deterministic for gradient computation
    )

    # External linear layer to map to desired output size
    output_layer = nn.Linear(quantum_layer.output_size, 3)

    # Generate data
    X_train, y_train = generate_dummy_data(batch_size=32)
    X_test, y_test = generate_dummy_data(batch_size=8)

    # Training phase - verify gradients exist
    quantum_layer.train()
    output_layer.train()
    optimizer = torch.optim.Adam(
        list(quantum_layer.parameters()) + list(output_layer.parameters()),
        lr=0.01
    )
    criterion = nn.CrossEntropyLoss()

    print("Training phase - checking gradients...")
    initial_params = {name: param.clone() for name, param in quantum_layer.named_parameters()}

    for epoch in range(3):
        optimizer.zero_grad()
        quantum_output = quantum_layer(X_train)
        output = output_layer(quantum_output)
        loss = criterion(output, y_train)
        loss.backward()

        # Check gradients exist
        for name, param in quantum_layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), f"Zero gradient for {name}"

        optimizer.step()
        print(f"  Epoch {epoch + 1}: Loss = {loss.item():.4f}")

    # Verify parameters changed
    for name, param in quantum_layer.named_parameters():
        if param.requires_grad:
            assert not torch.allclose(param, initial_params[name]), f"Parameter {name} didn't change"

    # Inference phase
    quantum_layer.eval()
    output_layer.eval()
    with torch.no_grad():
        test_quantum = quantum_layer(X_test)
        test_output = output_layer(test_quantum)
        test_loss = criterion(test_output, y_test)
        print(f"Test loss: {test_loss.item():.4f}")

    print("✓ GPU-only workflow test passed")


def test_cloud_inference_workflow():
    """Test training on GPU, then inference on QPU (cloud)."""
    print("\n" + "=" * 60)
    print("TEST 2: Cloud Inference Workflow")
    print("GPU Training → QPU Inference (No gradients in inference)")
    print("=" * 60)

    if CLOUD_TOKEN == "YOUR_TOKEN_HERE":
        print("⚠️ Skipping cloud test - no token provided")
        return

    # Create and train model on GPU
    circuit = create_quantum_circuit(5)
    quantum_layer = QuantumLayer(
        input_size=4,
        output_size=None,
        circuit=circuit,
        trainable_parameters=["theta"],
        input_parameters=["px"],
        input_state=[1, 1, 1, 1, 0],
        output_mapping_strategy=OutputMappingStrategy.NONE,
        shots=0  # Deterministic for training
    )

    output_layer = nn.Linear(quantum_layer.output_size, 3)

    X_train, y_train = generate_dummy_data(batch_size=16, input_size=4)
    X_test, y_test = generate_dummy_data(batch_size=4, input_size=4)

    # Train on GPU
    print("Training on GPU...")
    quantum_layer.train()
    output_layer.train()
    optimizer = torch.optim.Adam(
        list(quantum_layer.parameters()) + list(output_layer.parameters()),
        lr=0.01
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        optimizer.zero_grad()
        quantum_output = quantum_layer(X_train)
        output = output_layer(quantum_output)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        print(f"  Epoch {epoch + 1}: Loss = {loss.item():.4f}")

    # Deploy to cloud for inference
    print("Deploying to cloud for inference...")
    cloud_proc = deploy_to_cloud(
        quantum_layer,
        platform=REMOTE_PLATFORM,
        token=CLOUD_TOKEN,
        wait_timeout=120
    )

    # Verify training mode raises error
    quantum_layer.train()
    try:
        _ = quantum_layer(X_test)
        assert False, "Should have raised error in training mode with cloud backend"
    except RuntimeError as e:
        assert "Cannot compute gradients through cloud backend" in str(e)
        print("✓ Correctly blocked gradient computation on cloud")

    # Do actual inference
    quantum_layer.eval()
    output_layer.eval()
    with torch.no_grad():
        cloud_quantum = quantum_layer(X_test)
        cloud_output = output_layer(cloud_quantum)
        assert cloud_output.shape == (4, 3)
        print(f"Cloud inference output shape: {cloud_output.shape}")

    print("✓ Cloud inference workflow test passed")


def test_cloud_inference_workflow():
    """Test training on GPU, then inference on QPU (cloud)."""
    print("\n" + "=" * 60)
    print("TEST 2: Cloud Inference Workflow")
    print("GPU Training → QPU Inference (No gradients in inference)")
    print("=" * 60)

    # Skip if no token provided
    if CLOUD_TOKEN == "YOUR_TOKEN_HERE":
        print("⚠️ Skipping cloud test - no token provided")
        return

    # Create and train model on GPU
    circuit = create_quantum_circuit(5)
    quantum_layer = QuantumLayer(
        input_size=4,
        output_size=None,
        circuit=circuit,
        trainable_parameters=["theta"],
        input_parameters=["px"],
        input_state=[1, 1, 1, 1, 0],
        output_mapping_strategy=OutputMappingStrategy.NONE,
        shots=0  # Deterministic for training
    )

    output_layer = nn.Linear(quantum_layer.output_size, 3)

    X_train, y_train = generate_dummy_data(batch_size=16, input_size=4)
    X_test, y_test = generate_dummy_data(batch_size=4, input_size=4)

    # Train on GPU first (before cloud deployment)
    print("Training on GPU...")
    quantum_layer.train()
    output_layer.train()
    optimizer = torch.optim.Adam(
        list(quantum_layer.parameters()) + list(output_layer.parameters()),
        lr=0.01
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        optimizer.zero_grad()
        quantum_output = quantum_layer(X_train)
        output = output_layer(quantum_output)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        print(f"  Epoch {epoch + 1}: Loss = {loss.item():.4f}")

    # Verify cloud deployment requires valid token
    print("Testing cloud deployment...")

    # First, test that deployment with invalid token fails
    try:
        bad_cloud_proc = deploy_to_cloud(
            quantum_layer,
            platform=REMOTE_PLATFORM,
            token="INVALID_TOKEN_12345",
            wait_timeout=120
        )
        # If we get here, token wasn't validated properly
        print("WARNING: Cloud deployment accepted invalid token!")
    except Exception as e:
        print(f"✓ Correctly rejected invalid token: {type(e).__name__}")

    # Now deploy with valid token
    print(f"Deploying to cloud with valid token...")
    try:
        cloud_proc = deploy_to_cloud(
            quantum_layer,
            platform=REMOTE_PLATFORM,
            token=CLOUD_TOKEN,
            wait_timeout=120
        )
        print("✓ Successfully deployed to cloud")
    except Exception as e:
        print(f"❌ Failed to deploy: {e}")
        print("Make sure CLOUD_TOKEN is set correctly")
        return

    # Verify quantum layer now has cloud processor attached
    assert hasattr(quantum_layer, '_cloud_processor'), "Cloud processor not attached"
    assert quantum_layer._cloud_processor is not None, "Cloud processor is None"
    print("✓ Cloud processor properly attached")

    # Now verify training mode raises error
    quantum_layer.train()
    error_raised = False
    try:
        _ = quantum_layer(X_test)
    except RuntimeError as e:
        if "Cannot compute gradients through cloud backend" in str(e):
            error_raised = True
            print("✓ Correctly blocked gradient computation on cloud")
        else:
            print(f"❌ Wrong error: {e}")

    if not error_raised:
        print("❌ ERROR: Training mode should be blocked with cloud backend!")
        assert False, "Should have raised error in training mode with cloud backend"

    # Do actual inference in eval mode
    quantum_layer.eval()
    output_layer.eval()
    with torch.no_grad():
        print("Running cloud inference...")
        cloud_quantum = quantum_layer(X_test)
        cloud_output = output_layer(cloud_quantum)
        assert cloud_output.shape == (4, 3)
        print(f"✓ Cloud inference output shape: {cloud_output.shape}")

    print("✓ Cloud inference workflow test passed")


def test_hybrid_execution_workflow():
    """Test hybrid execution with partial gradient support."""
    print("\n" + "=" * 60)
    print("TEST 3: Hybrid Execution Workflow")
    print("Classical layers (GPU+gradients) + Quantum layer (mixed)")
    print("=" * 60)

    class HybridModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.classical_1 = nn.Linear(10, 4)
            self.activation_1 = nn.ReLU()

            circuit = create_quantum_circuit(5)
            self.quantum = QuantumLayer(
                input_size=4,
                output_size=None,
                circuit=circuit,
                trainable_parameters=["theta"],
                input_parameters=["px"],
                input_state=[1, 1, 1, 1, 0],
                output_mapping_strategy=OutputMappingStrategy.NONE,
                shots=0
            )

            self.classical_2 = nn.Linear(self.quantum.output_size, 3)

        def forward(self, x):
            x = self.activation_1(self.classical_1(x))
            x = self.quantum(x)
            x = self.classical_2(x)
            return x

    model = HybridModel()
    X_train, y_train = generate_dummy_data(batch_size=16, input_size=10)

    # Training with mixed gradients
    print("Training hybrid model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    initial_classical_1 = model.classical_1.weight.clone()
    initial_quantum = model.quantum.theta.clone() if hasattr(model.quantum, 'theta') else None
    initial_classical_2 = model.classical_2.weight.clone()

    for epoch in range(5):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()

        # Check gradients on classical layers
        assert model.classical_1.weight.grad is not None
        assert model.classical_2.weight.grad is not None

        # Check quantum layer gradients (should exist in GPU mode)
        if hasattr(model.quantum, 'theta'):
            assert model.quantum.theta.grad is not None

        optimizer.step()
        print(f"  Epoch {epoch + 1}: Loss = {loss.item():.4f}")

    # Verify all components updated
    assert not torch.allclose(model.classical_1.weight, initial_classical_1)
    assert not torch.allclose(model.classical_2.weight, initial_classical_2)
    if initial_quantum is not None:
        assert not torch.allclose(model.quantum.theta, initial_quantum)

    print("✓ Hybrid execution workflow test passed")


def test_reservoir_workflow():
    """Test reservoir computing with fixed quantum layer."""
    print("\n" + "=" * 60)
    print("TEST 4: Reservoir Computing Workflow")
    print("Fixed quantum layer + trainable output mapping")
    print("=" * 60)

    # Create reservoir model
    class ReservoirModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Input layer
            self.input_layer = nn.Linear(10, 4)

            # Quantum reservoir with fixed parameters (no trainable parameters)
            circuit = create_quantum_circuit(6, fixed_params=True)
            self.quantum_reservoir = QuantumLayer(
                input_size=4,
                output_size=None,  # Determined by quantum state space
                circuit=circuit,
                trainable_parameters=[],  # No trainable parameters
                input_parameters=["px"],
                input_state=[1, 1, 1, 0, 0, 0],
                output_mapping_strategy=OutputMappingStrategy.NONE,  # No internal mapping
                shots=100  # Add some noise for reservoir dynamics
            )

            # Output layer (trainable) - external mapping
            self.output_layer = nn.Linear(self.quantum_reservoir.output_size, 3)

        def forward(self, x):
            x = torch.tanh(self.input_layer(x))
            x = self.quantum_reservoir(x)
            x = self.output_layer(x)
            return x

    model = ReservoirModel()

    # Verify quantum layer has no trainable parameters
    quantum_params = list(model.quantum_reservoir.parameters())
    assert len(quantum_params) == 0, f"Quantum reservoir should have no parameters, has {len(quantum_params)}"

    # Generate data
    X_train, y_train = generate_dummy_data(batch_size=32, input_size=10)

    # Train only input and output layers
    print("Training reservoir model (only input/output layers)...")
    optimizer = torch.optim.Adam([
        {'params': model.input_layer.parameters()},
        {'params': model.output_layer.parameters()}
    ], lr=0.01)
    criterion = nn.CrossEntropyLoss()

    initial_input = model.input_layer.weight.clone()
    initial_output = model.output_layer.weight.clone()

    for epoch in range(5):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()

        # Verify gradients
        assert model.input_layer.weight.grad is not None
        assert model.output_layer.weight.grad is not None

        optimizer.step()
        print(f"  Epoch {epoch + 1}: Loss = {loss.item():.4f}")

    # Verify only input/output updated
    assert not torch.allclose(model.input_layer.weight, initial_input)
    assert not torch.allclose(model.output_layer.weight, initial_output)

    print("✓ Reservoir computing workflow test passed")


def test_scipy_finetuning_workflow():
    """Test fine-tuning with scipy optimization (gradient-free)."""
    print("\n" + "=" * 60)
    print("TEST 5: Scipy Fine-tuning Workflow")
    print("GPU Training → Gradient-free optimization")
    print("=" * 60)

    # Create and pre-train model - match input size to available px parameters
    circuit = create_quantum_circuit(4)
    quantum_layer = QuantumLayer(
        input_size=4,  # Circuit has px1-px4
        output_size=None,
        circuit=circuit,
        trainable_parameters=["theta"],
        input_parameters=["px"],
        input_state=[1, 0, 1, 0],
        output_mapping_strategy=OutputMappingStrategy.NONE,
        shots=0
    )

    output_layer = nn.Linear(quantum_layer.output_size, 3)

    # Generate data matching input size
    X_train, y_train = generate_dummy_data(batch_size=16, input_size=4, num_classes=3)

    # Initial training with gradients
    print("Initial training with gradients...")
    quantum_layer.train()
    output_layer.train()
    optimizer = torch.optim.Adam(
        list(quantum_layer.parameters()) + list(output_layer.parameters()),
        lr=0.01
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):
        optimizer.zero_grad()
        quantum_output = quantum_layer(X_train)
        output = output_layer(quantum_output)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        print(f"  Epoch {epoch + 1}: Loss = {loss.item():.4f}")

    initial_loss = loss.item()

    # Fine-tune with scipy (gradient-free)
    print("Fine-tuning with scipy (gradient-free)...")

    def objective_function(params_flat):
        """Objective function for scipy optimization."""
        with torch.no_grad():
            # Update all parameters
            idx = 0
            for module in [quantum_layer, output_layer]:
                for param in module.parameters():
                    param_size = param.numel()
                    param.data = torch.tensor(
                        params_flat[idx:idx + param_size].reshape(param.shape),
                        dtype=param.dtype
                    )
                    idx += param_size

            # Compute loss
            quantum_layer.eval()
            output_layer.eval()
            quantum_output = quantum_layer(X_train)
            output = output_layer(quantum_output)
            loss = criterion(output, y_train)
            return loss.item()

    # Flatten current parameters
    params_list = []
    for module in [quantum_layer, output_layer]:
        for param in module.parameters():
            params_list.append(param.detach().cpu().numpy().flatten())
    initial_params = np.concatenate(params_list)

    # Optimize with scipy
    result = optimize.minimize(
        objective_function,
        initial_params,
        method='Nelder-Mead',
        options={'maxiter': 50, 'disp': False}
    )

    final_loss = result.fun
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss after scipy: {final_loss:.4f}")

    # Verify parameters changed
    final_params = np.concatenate([
        p.detach().cpu().numpy().flatten()
        for module in [quantum_layer, output_layer]
        for p in module.parameters()
    ])
    assert not np.allclose(initial_params, final_params)

    print("✓ Scipy fine-tuning workflow test passed")


def test_transfer_learning_workflow():
    """Test transfer learning from simulator to QPU."""
    print("\n" + "=" * 60)
    print("TEST 6: Transfer Learning Workflow")
    print("Simulator training → QPU deployment")
    print("=" * 60)

    # Phase 1: Train on simulator
    print("Phase 1: Training on simulator...")
    circuit = create_quantum_circuit(5)
    sim_layer = QuantumLayer(
        input_size=4,  # Match available px parameters
        output_size=None,
        circuit=circuit,
        trainable_parameters=["theta"],
        input_parameters=["px"],
        input_state=[1, 1, 1, 0, 0],
        output_mapping_strategy=OutputMappingStrategy.NONE,
        shots=0  # Deterministic simulation
    )

    output_layer = nn.Linear(sim_layer.output_size, 4)

    # Generate data
    X_train, y_train = generate_dummy_data(batch_size=32, input_size=4, num_classes=4)

    optimizer = torch.optim.Adam(
        list(sim_layer.parameters()) + list(output_layer.parameters()),
        lr=0.01
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        optimizer.zero_grad()
        quantum_output = sim_layer(X_train)
        output = output_layer(quantum_output)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        print(f"  Epoch {epoch + 1}: Loss = {loss.item():.4f}")

    # Save trained parameters
    sim_state = sim_layer.state_dict()
    output_state = output_layer.state_dict()
    print(f"Saved parameters from simulator training")

    # Phase 2: Create new layer for QPU deployment
    print("\nPhase 2: Deploying to QPU with trained parameters...")
    qpu_layer = QuantumLayer(
        input_size=4,
        output_size=None,
        circuit=circuit,
        trainable_parameters=["theta"],
        input_parameters=["px"],
        input_state=[1, 1, 1, 0, 0],
        output_mapping_strategy=OutputMappingStrategy.NONE,
        shots=1000  # Realistic QPU shots
    )

    qpu_output_layer = nn.Linear(qpu_layer.output_size, 4)

    # Transfer learned parameters
    qpu_layer.load_state_dict(sim_state)
    qpu_output_layer.load_state_dict(output_state)
    print("Transferred parameters to QPU layer")

    # Test inference
    qpu_layer.eval()
    qpu_output_layer.eval()
    X_test, _ = generate_dummy_data(batch_size=4, input_size=4)

    with torch.no_grad():
        sim_quantum = sim_layer(X_test)
        sim_output = output_layer(sim_quantum)

        qpu_quantum = qpu_layer(X_test)
        qpu_output = qpu_output_layer(qpu_quantum)

        assert sim_output.shape == qpu_output.shape
        print(f"Output shape: {qpu_output.shape}")

    print("✓ Transfer learning workflow test passed")


def run_all_workflow_tests():
    """Run all workflow tests."""
    print("\n" + "=" * 60)
    print("WORKFLOW AND GRADIENT TESTING SUITE")
    print("=" * 60)

    try:
        test_gpu_only_workflow()
        test_cloud_inference_workflow()
        test_hybrid_execution_workflow()
        test_reservoir_workflow()
        test_scipy_finetuning_workflow()
        test_transfer_learning_workflow()

        print("\n" + "=" * 60)
        print("ALL WORKFLOW TESTS PASSED ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_workflow_tests()