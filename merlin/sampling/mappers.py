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
Output mapping implementations for quantum-to-classical conversion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from .strategies import OutputMappingStrategy, MeasurementStrategy, GroupingPolicy


class OutputMapper:
    """
    Handles mapping quantum states or measurements to classical outputs.

    This class provides factory methods for creating different types of output mappers
    that convert quantum states or measurements to classical outputs.
    """

    @staticmethod
    def create_mapping(
        strategy: OutputMappingStrategy | MeasurementStrategy,
        input_size: int,
        output_size: int = None,
        grouping_policy: GroupingPolicy = None,
        no_bunching: bool = True,
        keys: list[tuple[int, ...]] = None
    ):
        """
        Create an output mapping based on the specified strategy.

        Args:
            strategy: The measurement mapping strategy to use
            input_size: Size of the input probability distribution
            output_size: Desired size of the output tensor. In the case of ModeExpectation measurement strategy, this
                         argument is ignored and the output size is the number of modes.
            grouping_policy: If specified and strategy == MeasurementStrategy.FockGrouping, the GroupingPolicy will be
                             used to map the quantum probability distributions to classical outputs.
            no_bunching: If True (default), the per-mode probability of finding at least one photon is returned.
                         Otherwise, it is the per-mode expected number of photons that is returned. This is only used
                         for the ModeExpectation measurement strategy.
            keys: List of tuples that represent the possible quantum Fock states. This is only used for ModeExpectation
                  measurement strategy.
                  For example, keys = [(0,1,0,2), (1,0,1,0), ...]

        Returns:
            A PyTorch module that maps input_size to output_size

        Raises:
            ValueError: If strategy is unknown or sizes are incompatible for FockDistribution or StateVector strategies
            DeprecationWarning: If strategy is an OutputMappingStrategy
        """
        if type(strategy) == OutputMappingStrategy:
            warnings.warn('OutputMappingStrategy is deprecated and will be removed in version 0.3. '
                  'Use MeasurementStrategy instead. Switching to MeasurementStrategy.FockDistribution by default.',
                          DeprecationWarning, stacklevel=2)
            strategy = MeasurementStrategy.FockDistribution

        if strategy == MeasurementStrategy.FockDistribution:
            if input_size != output_size:
                raise ValueError(
                    f"Distribution size ({input_size}) must equal "
                    f"output size ({output_size}) when using FockDistribution measurement strategy"
                )
            return FockDistribution(input_size, output_size)
        elif strategy == MeasurementStrategy.FockGrouping:
            if grouping_policy is None:
                return FockGrouping(input_size, output_size)
            else:
                return FockGrouping(input_size, output_size, grouping_policy)
        elif strategy == MeasurementStrategy.ModeExpectation:
            if output_size is not None:
                warnings.warn(f'When using ModeExpectation measurement strategy, the output size is the number of '
                              f'modes, so the argument output_size={output_size} is ignored.')
            return ModeExpectation(input_size, no_bunching, keys)
        elif strategy == MeasurementStrategy.StateVector:
            if input_size != output_size:
                raise ValueError(
                    f"Distribution size ({input_size}) must equal "
                    f"output size ({output_size}) when using StateVector measurement strategy"
                )
            return StateVector(input_size, output_size)
        elif strategy == MeasurementStrategy.CustomObservable:
            return CustomObservable(input_size, output_size)
        else:
            raise ValueError(f"Unknown measurement strategy: {strategy}")


class FockDistribution(nn.Module):
    """Maps quantum state amplitudes or measurements to the complete Fock state probability distribution."""

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the converter from Fock state amplitudes or measurement counts to Fock state probability
        distribution.

        Args:
            input_size: Size of the possible Fock states or measurements
            output_size: Desired size of the output tensor
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        """Compute the probability distribution of possible Fock states from amplitudes or counts.

        Args:
            x: Input Fock states amplitudes or measurement counts of shape (n_batch, input_size) or (input_size,)

        Returns:
            Fock states probability tensor of shape (batch_size, output_size) or (output_size,)
        """
        single_input = x.ndim == 1
        if single_input:
            x = x.unsqueeze(0)

        # Determine if x represents amplitudes (normalized squared norm)
        norm = torch.sum(x.abs() ** 2, dim=1, keepdim=True)
        is_amplitude = torch.allclose(norm, torch.ones_like(norm), atol=1e-6)

        if is_amplitude:
            prob = x.abs() ** 2
        else:
            prob = x / torch.sum(x, dim=1, keepdim=True)

        return prob.squeeze(0) if single_input else prob


class FockGrouping(nn.Module):
    """
    Maps quantum state amplitudes or measurements to a grouping of the complete Fock state probability distribution.

    Grouping options:

    1. GroupingPolicy.LexGrouping

    This mapper groups consecutive elements of the probability distribution into equal-sized buckets and sums them to
    produce the output. If the input size is not evenly divisible by the output size, padding is applied.

    2. GroupingPolicy.ModGrouping

    This mapper groups elements of the probability distribution based on their index modulo the output size. Elements
    with the same modulo value are summed together to produce the output.
    """

    def __init__(self, input_size: int, output_size: int, grouping_policy: GroupingPolicy = GroupingPolicy.LexGrouping):
        """
        Initialize the converter from Fock state amplitudes or measurement counts to a grouping of Fock state
        probability distribution.

        Args:
            input_size: Size of the possible Fock states or measurements
            output_size: Desired size of the output tensor
            grouping_policy: Policy used for grouping (GroupingPolicy.LexGrouping or GroupingPolicy.ModGrouping)
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.grouping_policy = grouping_policy

    def forward(self, x):
        """
        Compute the probability distribution of possible Fock states from amplitudes or counts and map it to
        {output_size}, using {grouping_policy}.

        Args:
            x: Input Fock states amplitudes or measurement counts of shape (n_batch, input_size) or (input_size,)

        Returns:
            Grouped Fock states probability tensor of shape (batch_size, output_size) or (output_size,)
        """
        fock_distribution = FockDistribution(self.input_size, self.output_size)
        prob = fock_distribution(x)

        # LexGrouping
        if self.grouping_policy == GroupingPolicy.LexGrouping:
            pad_size = (self.output_size - (self.input_size % self.output_size)) % self.output_size
            if pad_size > 0:
                padded = F.pad(prob, (0, pad_size))
            else:
                padded = prob

            if prob.dim() == 2:
                return padded.view(prob.shape[0], self.output_size, -1).sum(dim=-1)
            else:
                return padded.view(self.output_size, -1).sum(dim=-1)

        # ModGrouping
        elif self.grouping_policy == GroupingPolicy.ModGrouping:
            if self.output_size > self.input_size:
                if prob.dim() == 2:
                    pad_size = self.output_size - self.input_size
                    padded = F.pad(prob, (0, pad_size))
                    return padded
                else:
                    pad_size = self.output_size - self.input_size
                    padded = F.pad(prob, (0, pad_size))
                    return padded

            indices = torch.arange(self.input_size, device=prob.device)
            group_indices = indices % self.output_size

            if prob.dim() == 2:
                batch_size = prob.shape[0]
                result = torch.zeros(
                    batch_size,
                    self.output_size,
                    device=prob.device,
                    dtype=prob.dtype,
                )
                for b in range(batch_size):
                    result[b] = torch.zeros(
                        self.output_size,
                        device=prob.device,
                        dtype=prob.dtype,
                    )
                    result[b].index_add_(0, group_indices, prob[b])
                return result
            else:
                result = torch.zeros(
                    self.output_size,
                    device=prob.device,
                    dtype=prob.dtype,
                )
                result.index_add_(0, group_indices, prob)
                return result

        else:
            raise ValueError(f"Unknown grouping policy: {self.grouping_policy}")


class ModeExpectation(nn.Module):
    """
    Maps probability per state distributions to probability of containing at least one photon per mode. This can also
    be interpreted as the expectation value per mode if considering threshold detectors. These detectors can tell
    whether a photon is detected or not, but cannot tell if multiple photons are detected at the same time.
    """

    def __init__(self, input_size: int, no_bunching: bool, keys: list[tuple[int, ...]]):
        """Initialize the expectation grouping mapper.

        Args:
            input_size: Size of the input probability distribution
            no_bunching: If True (default), the per-mode probability of finding at least one photon is returned.
                         Otherwise, it is the per-mode expected number of photons that is returned.
            keys: List of tuples describing the possible Fock states output from the circuit preceding the output
                  mapping. e.g., [(0,1,0,2), (1,0,1,0), ...]
        """
        super().__init__()
        self.input_size = input_size
        self.no_bunching = no_bunching
        self.keys = keys

        if not keys:
            raise ValueError("Keys list cannot be empty")

        if len(set(len(key) for key in keys)) > 1:
            raise ValueError("All keys must have the same length (number of modes)")

        if len(keys) != input_size:
            raise ValueError(
                f"Number of keys ({len(keys)}) must match input_size ({input_size})"
            )

        # Create mask and register as buffer
        keys_tensor = torch.tensor(keys, dtype=torch.long)
        if no_bunching:
            mask = (keys_tensor >= 1).T.float()
        else:
            mask = keys_tensor.T.float()
        self.register_buffer("mask", mask)

    def marginalize_photon_presence(self, probability_distribution: torch.Tensor) -> torch.Tensor:
        """
        Marginalize Fock state probabilities to get per-mode occupation probabilities.

        Computes the probability that each mode contains at least one photon
        by summing over all Fock states where that mode is occupied.

        Args:
            probability_distribution (torch.Tensor): Tensor of shape (N, num_keys) with probabilities
                for each Fock state, with requires_grad=True

        Returns:
            torch.Tensor: Shape (N, num_modes) with marginal probability that
                each mode has at least one photon
        """
        marginalized = probability_distribution @ self.mask.T
        return marginalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Marginalize the per state probability distribution into a per mode probability detection distribution.

        Args:
            x: Input Fock states amplitudes or measurement counts of shape (n_batch, input_size) or (input_size,)

        Returns:
            Grouped probability tensor of shape (batch_size, output_size) or (output_size,)
        """
        # Validate input
        if x.dim() not in [1, 2]:
            raise ValueError("Input must be 1D or 2D tensor")

        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"Last dimension must be {self.input_size}, got {x.shape[-1]}"
            )

        # Get probabilities
        fock_distribution = FockDistribution(self.input_size, self.input_size)
        prob = fock_distribution(x)

        # Handle both 1D and 2D inputs uniformly
        original_shape = prob.shape
        if prob.dim() == 1:
            prob = prob.unsqueeze(0)

        marginalized_probs = self.marginalize_photon_presence(prob)

        if len(original_shape) == 1:
            marginalized_probs = marginalized_probs.squeeze(0)

        return marginalized_probs


class StateVector(nn.Module):
    """
    Output the Fock state vector directly. This can only be done with a simulator because amplitudes cannot be retrieved
    from the state count measurements on a GPU.
    """

    def __init__(self, input_size: int, output_size: int):
        """Initialize the state vector measurement strategy."""
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the fock state vector amplitudes."""
        original_shape = x.shape
        if x.ndim == 1:
            x = x.unsqueeze(0)
        n_batch, n_amplitudes = x.shape
        assert torch.allclose(torch.sum( x.abs()**2, dim=1), torch.ones(n_batch), atol=1e-6), ('Amplitudes are needed '
               'to use the StateVector measurement strategy.')
        if len(original_shape) == 1:
            x = x.squeeze(0)
        return x


class CustomObservable(nn.Module):
    """

    """

    def __init__(self, input_size: int, output_size: int):
        #TODO
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #TODO
        return x