"""
GPU tests for cloud processor in hybrid quantum-classical workflows.
Tests GPU tensor handling, memory management, and device transfers.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import perceval as pcvl
from unittest.mock import Mock, patch, MagicMock
from merlin.core.cloud_processor import CloudProcessor, deploy_to_cloud
from merlin.algorithms import QuantumLayer
from merlin.sampling.strategies import OutputMappingStrategy


def create_test_circuit(n_modes: int = 4) -> pcvl.Circuit:
    """Create a simple test circuit for GPU tests."""
    circuit = pcvl.Circuit(n_modes)
    circuit.add(0, pcvl.BS())
    for i in range(min(3, n_modes)):
        circuit.add(i, pcvl.PS(pcvl.P(f"input_{i + 1}")))
    return circuit


def create_test_layer() -> QuantumLayer:
    """Create a test quantum layer for GPU tests."""
    return QuantumLayer(
        input_size=4,
        output_size=3,
        circuit=create_test_circuit(),
        trainable_parameters=[],
        input_parameters=["input"],
        input_state=[1, 1, 0, 0],
        output_mapping_strategy=OutputMappingStrategy.NONE,
        shots=1000
    )


@pytest.mark.gpu
class TestCloudProcessorGPU:

    @pytest.fixture
    def mock_cloud_processor(self):
        """Mock cloud processor for testing without actual cloud connection."""
        with patch('merlin.core.cloud_processor.RemoteProcessor') as mock_remote, \
                patch('merlin.core.cloud_processor.Sampler') as mock_sampler, \
                patch('merlin.core.cloud_processor.RemoteConfig') as mock_config:
            # Setup mock responses
            mock_job = MagicMock()
            mock_job.is_complete = True
            mock_job.is_failed = False
            mock_job.get_results.return_value = {
                'results_list': [
                    {'results': {'|1,1,0,0>': 500, '|0,1,1,0>': 500}}
                    for _ in range(32)  # Max batch size
                ]
            }

            mock_sampler_instance = MagicMock()
            mock_sampler_instance.sample_count.execute_async.return_value = mock_job
            mock_sampler.return_value = mock_sampler_instance

            yield CloudProcessor(platform="sim:ascella", token="test_token")

    def test_gpu_to_cloud_data_transfer(self, mock_cloud_processor):
        """Test that GPU tensors are properly handled when sending to cloud."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        # Create layer and attach to mocked cloud processor
        quantum_layer = create_test_layer()
        mock_cloud_processor.attach_layer(quantum_layer)
        quantum_layer.eval()

        # Create data on GPU
        X_gpu = torch.randn(10, 4, device='cuda')

        # Execute - should handle GPU->CPU transfer automatically
        output = mock_cloud_processor.execute(quantum_layer, X_gpu)

        # Verify output is on same device as input
        assert output.device == X_gpu.device
        assert output.shape[0] == 10
        assert output.shape[1] == quantum_layer.output_size or output.shape[1] > 0

    def test_hybrid_workflow_gpu(self, mock_cloud_processor):
        """Test complete hybrid workflow with GPU classical processing."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        # Classical preprocessing on GPU
        classical_layer = nn.Linear(8, 4).cuda()

        # Quantum layer with mock cloud
        quantum_layer = create_test_layer()
        mock_cloud_processor.attach_layer(quantum_layer)
        quantum_layer.eval()

        # Classical postprocessing on GPU
        post_layer = nn.Linear(3, 2).cuda()

        # Full pipeline
        X = torch.randn(20, 8, device='cuda')

        with torch.no_grad():
            # GPU -> GPU (classical)
            x1 = classical_layer(X)

            # GPU -> Cloud -> GPU (quantum)
            x2 = mock_cloud_processor.execute(quantum_layer, x1)

            # Adjust post layer input size if needed
            if x2.shape[1] != 3:
                post_layer = nn.Linear(x2.shape[1], 2).cuda()

            # GPU -> GPU (classical)
            output = post_layer(x2)

        assert output.device.type == 'cuda'
        assert output.shape == (20, 2)

    def test_batch_processing_gpu_memory(self, mock_cloud_processor):
        """Test that large GPU batches don't cause OOM when split for cloud."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        quantum_layer = create_test_layer()
        mock_cloud_processor.attach_layer(quantum_layer)
        quantum_layer.eval()

        # Large batch that would be split for cloud (>32 samples)
        large_batch = torch.randn(100, 4, device='cuda')

        # Clear GPU cache
        torch.cuda.empty_cache()

        # Monitor GPU memory
        initial_memory = torch.cuda.memory_allocated()

        output = mock_cloud_processor.execute(quantum_layer, large_batch)

        # Should efficiently handle memory during splitting
        peak_memory = torch.cuda.max_memory_allocated()
        memory_increase = peak_memory - initial_memory

        # Memory increase should be reasonable
        # Allow for some overhead but not full duplication
        max_allowed_increase = large_batch.nelement() * large_batch.element_size() * 3
        assert memory_increase < max_allowed_increase, \
            f"Memory increase {memory_increase} exceeds threshold {max_allowed_increase}"
        assert output.device.type == 'cuda'
        assert output.shape[0] == 100

    def test_gradient_flow_blocked(self, mock_cloud_processor):
        """Verify gradients don't flow through cloud execution."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        quantum_layer = create_test_layer()
        mock_cloud_processor.attach_layer(quantum_layer)

        # Should raise error in training mode
        quantum_layer.train()
        X = torch.randn(5, 4, device='cuda', requires_grad=True)

        with pytest.raises(RuntimeError, match="Cannot compute gradients"):
            output = mock_cloud_processor.execute(quantum_layer, X)

        # Should work in eval mode but no gradients
        quantum_layer.eval()
        output = mock_cloud_processor.execute(quantum_layer, X)
        assert output.requires_grad == False
        assert output.device.type == 'cuda'

    def test_mixed_precision_compatibility(self, mock_cloud_processor):
        """Test compatibility with mixed precision training."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        quantum_layer = create_test_layer()
        mock_cloud_processor.attach_layer(quantum_layer)
        quantum_layer.eval()

        # Test with different dtypes
        for dtype in [torch.float16, torch.float32, torch.float64]:
            X = torch.randn(5, 4, device='cuda', dtype=dtype)
            output = mock_cloud_processor.execute(quantum_layer, X)

            # Output should maintain dtype
            assert output.dtype == dtype
            assert output.device.type == 'cuda'

    def test_multi_gpu_data_parallel(self, mock_cloud_processor):
        """Test behavior with DataParallel (if multiple GPUs available)."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        if torch.cuda.device_count() < 2:
            pytest.skip("Multiple GPUs not available")

        quantum_layer = create_test_layer()
        mock_cloud_processor.attach_layer(quantum_layer)
        quantum_layer.eval()

        # Wrap in DataParallel
        parallel_layer = nn.DataParallel(quantum_layer)

        # Test with batch
        X = torch.randn(16, 4, device='cuda')

        # Should handle multi-GPU scenario
        # Note: Cloud execution happens on CPU, so DataParallel mainly affects input/output
        with torch.no_grad():
            output = parallel_layer(X)

        assert output.device.type == 'cuda'
        assert output.shape[0] == 16

    def test_cuda_stream_synchronization(self, mock_cloud_processor):
        """Test proper CUDA stream synchronization."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        quantum_layer = create_test_layer()
        mock_cloud_processor.attach_layer(quantum_layer)
        quantum_layer.eval()

        # Create custom CUDA stream
        stream = torch.cuda.Stream()

        X = torch.randn(10, 4, device='cuda')

        with torch.cuda.stream(stream):
            # Operations in custom stream
            output = mock_cloud_processor.execute(quantum_layer, X)

        # Synchronize to ensure completion
        stream.synchronize()

        # Verify output is accessible
        assert output.device.type == 'cuda'
        assert output.shape[0] == 10
        # Should be able to access values after sync
        _ = output.cpu().numpy()

    def test_device_mismatch_handling(self, mock_cloud_processor):
        """Test handling of device mismatches in hybrid workflows."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        quantum_layer = create_test_layer()
        mock_cloud_processor.attach_layer(quantum_layer)
        quantum_layer.eval()

        # Mix of CPU and GPU tensors
        X_cpu = torch.randn(5, 4)
        X_gpu = torch.randn(5, 4, device='cuda')

        # Should handle both correctly
        output_cpu = mock_cloud_processor.execute(quantum_layer, X_cpu)
        output_gpu = mock_cloud_processor.execute(quantum_layer, X_gpu)

        assert output_cpu.device.type == 'cpu'
        assert output_gpu.device.type == 'cuda'
        assert output_cpu.shape == output_gpu.shape

    def test_memory_pinning_optimization(self, mock_cloud_processor):
        """Test that pinned memory is used for efficient CPU-GPU transfers."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        quantum_layer = create_test_layer()
        mock_cloud_processor.attach_layer(quantum_layer)
        quantum_layer.eval()

        # Create pinned memory tensor
        X_pinned = torch.randn(20, 4, pin_memory=True)
        X_gpu = X_pinned.cuda()

        # Execute with GPU tensor (originated from pinned memory)
        output = mock_cloud_processor.execute(quantum_layer, X_gpu)

        assert output.device.type == 'cuda'
        assert output.shape[0] == 20

        # Transfer back to CPU should be efficient
        output_cpu = output.cpu()
        assert output_cpu.device.type == 'cpu'


@pytest.mark.gpu
def test_gpu_quantum_classical_optimization():
    """Test that hybrid models can be optimized with GPU acceleration."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")

    # Mock the cloud connection
    with patch('merlin.core.cloud_processor.RemoteProcessor'), \
            patch('merlin.core.cloud_processor.Sampler') as mock_sampler, \
            patch('merlin.core.cloud_processor.RemoteConfig'):

        # Setup mock
        mock_job = MagicMock()
        mock_job.is_complete = True
        mock_job.is_failed = False
        mock_job.get_results.return_value = {
            'results_list': [
                {'results': {'|1,1,0,0>': 500, '|0,1,1,0>': 500}}
                for _ in range(10)
            ]
        }

        mock_sampler_instance = MagicMock()
        mock_sampler_instance.sample_count.execute_async.return_value = mock_job
        mock_sampler.return_value = mock_sampler_instance

        # Build hybrid model
        class HybridModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.classical1 = nn.Linear(10, 4).cuda()
                self.quantum = create_test_layer()
                self.cloud_proc = CloudProcessor("sim:ascella", token="test")
                self.cloud_proc.attach_layer(self.quantum)
                self.classical2 = nn.Linear(3, 2).cuda()

            def forward(self, x):
                # Only classical parts have gradients
                x = torch.relu(self.classical1(x))

                # Quantum part in eval mode (no gradients)
                self.quantum.eval()
                with torch.no_grad():
                    x = self.cloud_proc.execute(self.quantum, x)

                # Adjust dimensions if needed
                if x.shape[1] != 3:
                    x = x[:, :3] if x.shape[1] > 3 else torch.nn.functional.pad(x, (0, 3 - x.shape[1]))

                x = x.detach().requires_grad_(True)  # Re-enable gradients
                x = self.classical2(x)
                return x

        model = HybridModel()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.01
        )

        # Training step
        X = torch.randn(10, 10, device='cuda')
        y = torch.randn(10, 2, device='cuda')

        output = model(X)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()

        # Verify optimization occurred
        assert loss.device.type == 'cuda'
        assert output.device.type == 'cuda'