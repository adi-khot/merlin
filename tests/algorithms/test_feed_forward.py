# MIT License
#
# Copyright (c) 2025 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Tests for the FeedForward class.
"""

import pytest
import torch

from merlin.algorithms.feed_forward import FeedForwardBlock


class TestFeedForwardBlock:
    """Comprehensive test suite for FeedForwardBlock."""

    def test_init_single_mode(self):
        """Test initialization with single conditional mode."""
        ff = FeedForwardBlock(input_size=6, m=6, n=2, depth=3, conditional_modes=[0])
        assert ff.m == 6
        assert ff.n_photons == 2
        assert ff.conditional_modes == [0]
        assert ff.depth == 3
        assert isinstance(ff.layers, dict)

    def test_init_multi_mode(self):
        """Test initialization with multiple conditional modes."""
        ff = FeedForwardBlock(input_size=6, m=6, n=2, depth=2, conditional_modes=[0, 1])
        assert ff.n_cond == 2
        assert all(isinstance(k, tuple) for k in ff.layers.keys())

    def test_generate_possible_tuples(self):
        """Ensure generated tuples are non-empty and valid."""
        ff = FeedForwardBlock(input_size=4, n=2, m=4, conditional_modes=[0, 1])
        tuples = ff.generate_possible_tuples()
        assert isinstance(tuples, list)
        assert len(tuples) > 0
        assert all(isinstance(t, tuple) for t in tuples)

    def test_parameters_method(self):
        """Check that parameters() returns a non-empty iterable."""
        ff = FeedForwardBlock(input_size=4, n=2, m=4, depth=2, conditional_modes=[0])
        params = list(ff.parameters())
        assert all(isinstance(p, torch.Tensor) for p in params)

    def test_forward_output_shape_and_sum(self):
        """Check forward pass produces valid probabilities summing to 1."""
        ff = FeedForwardBlock(input_size=4, n=2, m=4, depth=2, conditional_modes=[0])
        x = torch.rand(3, 4)  # batch of 3
        output = ff(x)
        assert isinstance(output, torch.Tensor)
        assert output.ndim == 2  # (batch_size, n_outputs)
        # Each row should sum approximately to 1
        probs_sum = output.sum(dim=1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-3)

    def test_forward_multi_mode(self):
        """Test forward with multiple conditional modes and output normalization."""
        ff = FeedForwardBlock(input_size=3, n=2, m=4, depth=2, conditional_modes=[0, 1])
        x = torch.rand(2, 3)
        output = ff(x)
        assert output.shape[0] == 2  # batch size preserved
        assert output.shape[1] > 0
        assert torch.allclose(output.sum(dim=1), torch.ones(2), atol=1e-3)

    def test_backward_pass(self):
        """Ensure gradients propagate correctly through the quantum layers."""
        ff = FeedForwardBlock(input_size=4, n=2, m=4, depth=2, conditional_modes=[0])
        x = torch.rand(1, 4, requires_grad=True)
        output = ff(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_indices_by_values_and_match_indices(self):
        """Test low-level index mapping helpers."""
        ff = FeedForwardBlock(input_size=2, n=2, m=4, depth=2, conditional_modes=[0, 1])
        keys = [(0, 1, 0), (1, 0, 1), (0, 0, 1)]
        idx_dict = ff._indices_by_values(keys, [0, 1])
        assert all(isinstance(v, torch.Tensor) for v in idx_dict.values())

        data_out = [(1, 0), (0, 1)]
        idx = ff._match_indices_multi(keys, data_out, [0, 1], (0, 1))
        assert isinstance(idx, torch.Tensor)

    def test_integration_with_optimizer(self):
        """Ensure block integrates cleanly with PyTorch optimizers."""
        ff = FeedForwardBlock(input_size=4, n=2, m=4, depth=2, conditional_modes=[0])
        x = torch.rand(1, 4)
        optimizer = torch.optim.Adam(ff.parameters(), lr=1e-3)

        output = ff(x)
        loss = output.pow(2).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        assert True  # No runtime errors

    def test_output_keys_consistency(self):
        """Ensure output keys are consistent after forward pass."""
        ff = FeedForwardBlock(input_size=3, n=2, m=3, depth=2, conditional_modes=[0])
        _ = ff(torch.rand(1, 3))
        keys = ff.get_output_keys()
        print(keys)
        assert isinstance(keys, list)
        assert len(keys) > 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available for GPU tests"
)
class TestFeedForwardBlockGPU:
    """GPU-only tests for FeedForwardBlock forward/backward behavior."""

    def test_forward_on_gpu(self):
        """Ensure forward pass works on GPU and outputs valid probabilities."""
        device = torch.device("cuda")
        ff = FeedForwardBlock(
            input_size=4, n=2, m=4, depth=2, conditional_modes=[0]
        ).to(device)
        x = torch.rand(8, 4, device=device)
        output = ff(x)
        assert output.device.type == "cuda"
        assert torch.allclose(
            output.sum(dim=1), torch.ones(8, device=device), atol=1e-3
        )

    def test_backward_on_gpu(self):
        """Ensure gradients flow correctly on GPU."""
        device = torch.device("cuda")
        ff = FeedForwardBlock(
            input_size=4, n=2, m=4, depth=2, conditional_modes=[0]
        ).to(device)
        x = torch.rand(4, 4, device=device, requires_grad=True)
        output = ff(x)
        loss = output.sum()
        loss.backward()

        # Verify gradients computed and reside on GPU
        assert x.grad is not None
        assert x.grad.device.type == "cuda"
        assert not torch.isnan(x.grad).any()

    def test_forward_backward_multi_mode_gpu(self):
        """Test feedforward & gradient with multiple conditional modes on GPU."""
        device = torch.device("cuda")
        ff = FeedForwardBlock(
            input_size=6, n=3, m=6, depth=2, conditional_modes=[0, 1]
        ).to(device)
        x = torch.rand(2, 6, device=device, requires_grad=True)

        output = ff(x)
        assert output.device.type == "cuda"
        assert output.shape[0] == 2
        assert torch.allclose(
            output.sum(dim=1), torch.ones(2, device=device), atol=1e-3
        )

        loss = output.mean()
        loss.backward()
        assert x.grad is not None
        assert x.grad.device.type == "cuda"

    def test_gpu_optimizer_step(self):
        """Ensure GPU-based optimization loop runs correctly."""
        device = torch.device("cuda")
        ff = FeedForwardBlock(
            input_size=4, n=2, m=4, depth=2, conditional_modes=[0]
        ).to(device)
        x = torch.rand(2, 4, device=device)
        optimizer = torch.optim.Adam(ff.parameters(), lr=1e-3)

        # Forward, backward, and optimizer step
        output = ff(x)
        loss = output.pow(2).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Ensure no NaNs or device mismatch
        assert not torch.isnan(loss).any()
        assert all(p.device.type == "cuda" for p in ff.parameters())


if __name__ == "__main__":
    pytest.main([__file__])
