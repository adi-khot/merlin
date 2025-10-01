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
import perceval as pcvl

from merlin.core.feed_forward import FeedForwardBlock, PoolingFeedForward, define_layer_no_input


class TestFeedForward:
    """Test suite for FeedForward class."""

    def test_init(self):
        """Test FeedForward initialization."""
        ff = FeedForwardBlock(input_size=6, m=6, n=2, depth=3, conditional_mode=0)
        assert ff.m == 6
        assert ff.n_photons == 2
        assert ff.conditional_mode == 0
        assert ff.depth == 3
        assert isinstance(ff.layers, dict)

    def test_generate_possible_tuples(self):
        """Test possible tuples generation."""
        ff = FeedForwardBlock(input_size=2, n=2, m=4, conditional_mode=0)
        tuples = ff.generate_possible_tuples()
        assert isinstance(tuples, list)
        assert len(tuples) > 0

    def test_parameters_method(self):
        """Test parameters() method returns generator."""
        ff = FeedForwardBlock(input_size=4, n=2, m=4, depth=2, conditional_mode=0)
        params = list(ff.parameters())
        assert len(params) > 0
        assert all(isinstance(p, torch.Tensor) for p in params)

    def test_forward_pass(self):
        """Test forward pass execution."""
        ff = FeedForwardBlock(input_size=4, n=2, m=4, depth=2, conditional_mode=0)
        x = torch.rand(1, 4)
        output = ff(x)
        assert isinstance(output, torch.Tensor)
        assert output.requires_grad

    def test_backward_pass(self):
        """Test backward pass execution."""
        ff = FeedForwardBlock(input_size=4, n=2, m=4, depth=2, conditional_mode=0)
        x = torch.rand(1, 4, requires_grad=True)

        output = ff(x)
        loss = output.sum()

        # Should not raise an error
        loss.backward()

        # Check gradients are computed
        assert x.grad is not None

    def test_indices_by_value(self):
        """Test indices_by_value method."""
        ff = FeedForwardBlock(input_size=2, n=2, m=4, depth=2, conditional_mode=0)
        keys = [(0, 1, 0), (1, 0, 1), (0, 0, 1)]
        idx_0, idx_1 = ff._indices_by_value(keys, 0)

        assert len(idx_0) == 2  # positions where first element is 0
        assert len(idx_1) == 1  # positions where first element is 1

    def test_match_indices(self):
        """Test match_indices method."""
        ff = FeedForwardBlock(input_size=2, n=2, m=4, depth=2, conditional_mode=0)
        data = [(0, 1, 0), (1, 0, 1), (0, 0, 1)]
        data_out = [(1, 0), (0, 1), (0, 1)]

        idx = ff._match_indices(data, data_out, k=0, k_value=0)
        assert isinstance(idx, torch.Tensor)

    def test_integration_with_optimizer(self):
        """Test integration with PyTorch optimizer."""
        ff = FeedForwardBlock(input_size=4, n=2, m=4, depth=2, conditional_mode=0)
        x = torch.rand(1, 4)
        optimizer = torch.optim.Adam(ff.parameters())

        # Forward pass
        output = ff(x)
        loss = output.pow(2).sum()

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Should complete without errors
        assert True


class TestPoolingFeedForward:
    """Test suite for PoolingFeedForward class."""

    def test_init_default_pooling(self):
        """Test PoolingFeedForward initialization with default pooling modes."""
        pff = PoolingFeedForward(n_modes=16, n_photons=2, n_output_modes=8)
        assert pff.n_modes == 16
        assert isinstance(pff.match_indices, torch.Tensor)
        assert isinstance(pff.exclude_indices, torch.Tensor)
        assert isinstance(pff.keys_out, list)

    def test_init_custom_pooling(self):
        """Test PoolingFeedForward initialization with custom pooling modes."""
        pooling_modes = [[0, 1], [2, 3], [4, 5], [6, 7]]
        pff = PoolingFeedForward(
            n_modes=8, n_photons=2, n_output_modes=4, pooling_modes=pooling_modes
        )
        assert pff.n_modes == 8
        assert isinstance(pff.match_indices, torch.Tensor)
        assert isinstance(pff.exclude_indices, torch.Tensor)

    def test_init_with_bunching(self):
        """Test PoolingFeedForward initialization with bunching allowed."""
        pff = PoolingFeedForward(
            n_modes=8, n_photons=2, n_output_modes=4, no_bunching=False
        )
        assert pff.n_modes == 8
        assert isinstance(pff.match_indices, torch.Tensor)

    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        pff = PoolingFeedForward(n_modes=16, n_photons=2, n_output_modes=8)
        batch_size = 4
        n_input_states = len(pff.match_indices) + len(pff.exclude_indices)
        amplitudes = torch.rand(batch_size, n_input_states)
        output = pff(amplitudes)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == batch_size
        assert output.shape[1] == len(pff.keys_out)

    def test_forward_pass_dtype_preservation(self):
        """Test that forward pass preserves dtype."""
        pff = PoolingFeedForward(n_modes=8, n_photons=2, n_output_modes=4)
        n_input_states = len(pff.match_indices) + len(pff.exclude_indices)

        # Test float32
        amplitudes_f32 = torch.rand(2, n_input_states, dtype=torch.float32)
        output_f32 = pff(amplitudes_f32)
        assert output_f32.dtype == torch.float32

        # Test float64
        amplitudes_f64 = torch.rand(2, n_input_states, dtype=torch.float64)
        output_f64 = pff(amplitudes_f64)
        assert output_f64.dtype == torch.float64

    def test_forward_pass_device_preservation(self):
        """Test that forward pass preserves device."""
        pff = PoolingFeedForward(n_modes=8, n_photons=2, n_output_modes=4)
        n_input_states = len(pff.match_indices) + len(pff.exclude_indices)
        amplitudes = torch.rand(2, n_input_states)
        output = pff(amplitudes)

        assert output.device == amplitudes.device

    def test_match_tuples_method(self):
        """Test match_tuples method."""
        pff = PoolingFeedForward(n_modes=8, n_photons=2, n_output_modes=4)
        keys_in = [(0, 0, 1, 1, 0, 0, 0, 0), (1, 0, 1, 0, 0, 0, 0, 0)]
        keys_out = [(0, 1, 0, 0), (1, 1, 0, 0)]
        pooling_modes = [[0, 1], [2, 3], [4, 5], [6, 7]]

        match_indices, exclude_indices = pff.match_tuples(
            keys_in, keys_out, pooling_modes
        )

        assert isinstance(match_indices, list)
        assert isinstance(exclude_indices, list)

    def test_backward_pass_integration(self):
        """Test backward pass through PoolingFeedForward."""
        pff = PoolingFeedForward(n_modes=8, n_photons=2, n_output_modes=4)
        n_input_states = len(pff.match_indices) + len(pff.exclude_indices)
        amplitudes = torch.rand(2, n_input_states, requires_grad=True)

        output = pff(amplitudes)
        loss = output.sum()
        loss.backward()

        assert amplitudes.grad is not None

    def test_integration_with_quantum_layer(self):
        """Test integration with quantum layers (as in the main script)."""
        pff = PoolingFeedForward(n_modes=16, n_photons=2, n_output_modes=8)
        pre_layer = define_layer_no_input(16, 2)
        post_layer = define_layer_no_input(8, 2)

        # Forward pass
        _, amplitudes = pre_layer(return_amplitudes=True)
        amplitudes = pff(amplitudes)
        post_layer.set_input_state(amplitudes)
        res = post_layer()

        assert isinstance(res, torch.Tensor)
        assert res.requires_grad

    def test_optimization_with_layers(self):
        """Test optimization loop with PoolingFeedForward between layers."""
        from itertools import chain

        pff = PoolingFeedForward(n_modes=8, n_photons=2, n_output_modes=4)
        pre_layer = define_layer_no_input(8, 2)
        post_layer = define_layer_no_input(4, 2)

        params = chain(pre_layer.parameters(), post_layer.parameters())
        optimizer = torch.optim.Adam(params)

        # Run a few optimization steps
        for _ in range(3):
            _, amplitudes = pre_layer(return_amplitudes=True)
            amplitudes = pff(amplitudes)
            post_layer.set_input_state(amplitudes)
            res = post_layer().pow(2).sum()
            res.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Should complete without errors
        assert True

    def test_different_pooling_ratios(self):
        """Test various pooling ratios."""
        test_cases = [
            (16, 8),  # 2:1 pooling
            (16, 4),  # 4:1 pooling
            (12, 6),  # 2:1 pooling
            (9, 3),   # 3:1 pooling
        ]

        for n_modes, n_output_modes in test_cases:
            pff = PoolingFeedForward(
                n_modes=n_modes, n_photons=2, n_output_modes=n_output_modes
            )
            n_input_states = len(pff.match_indices) + len(pff.exclude_indices)
            amplitudes = torch.rand(2, n_input_states)
            output = pff(amplitudes)

            assert output.shape[1] == len(pff.keys_out)

    def test_single_photon_case(self):
        """Test PoolingFeedForward with a single photon."""
        pff = PoolingFeedForward(n_modes=8, n_photons=1, n_output_modes=4)
        n_input_states = len(pff.match_indices) + len(pff.exclude_indices)
        amplitudes = torch.rand(2, n_input_states)
        output = pff(amplitudes)

        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 2

    def test_batch_size_one(self):
        """Test PoolingFeedForward with batch size of 1."""
        pff = PoolingFeedForward(n_modes=8, n_photons=2, n_output_modes=4)
        n_input_states = len(pff.match_indices) + len(pff.exclude_indices)
        amplitudes = torch.rand(1, n_input_states)
        output = pff(amplitudes)

        assert output.shape[0] == 1

    def test_large_batch_size(self):
        """Test PoolingFeedForward with large batch size."""
        pff = PoolingFeedForward(n_modes=8, n_photons=2, n_output_modes=4)
        n_input_states = len(pff.match_indices) + len(pff.exclude_indices)
        batch_size = 32
        amplitudes = torch.rand(batch_size, n_input_states)
        output = pff(amplitudes)

        assert output.shape[0] == batch_size


if __name__ == "__main__":
    pytest.main([__file__])
