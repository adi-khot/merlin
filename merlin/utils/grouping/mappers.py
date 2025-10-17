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

"""Grouping policies."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LexGrouping(nn.Module):
    """
    Maps tensor to a lexical grouping of its components.

    This mapper groups consecutive elements of the input tensor into equal-sized buckets and sums them to
    produce the output. If the input size is not evenly divisible by the output size, padding is applied.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
    ):
        """
        Initialize the converter from input tensor to a lexical grouping of its elements.

        Args:
            input_size: Size of the input tensor
            output_size: Desired size of the output tensor
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        """
        Map the input tensor to the desired output_size utilizing lexical grouping.

        Args:
            x: Input tensor of shape (n_batch, input_size) or (input_size,)

        Returns:
            Grouped tensor of shape (batch_size, output_size) or (output_size,)
        """
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"Input tensor's last dimension ({x.shape[-1]}) does not correspond to the provided input_size ({self.input_size})"
            )

        pad_size = (
            self.output_size - (self.input_size % self.output_size)
        ) % self.output_size
        if pad_size > 0:
            padded = F.pad(x, (0, pad_size))
        else:
            padded = x

        if x.dim() == 2:
            return padded.reshape(x.shape[0], self.output_size, -1).sum(dim=-1)
        else:
            return padded.reshape(self.output_size, -1).sum(dim=-1)


class ModGrouping(nn.Module):
    """
    Maps tensor to a modulo grouping of its components.

    This mapper groups elements of the input tensor based on their index modulo the output size. Elements
    with the same modulo value are summed together to produce the output.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
    ):
        """
        Initialize the converter from input tensor to a modulo grouping of its elements.

        Args:
            input_size: Size of the input tensor
            output_size: Desired size of the output tensor
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        """
        Map the input tensor to the desired output_size utilizing modulo grouping.

        Args:
            x: Input tensor of shape (n_batch, input_size) or (input_size,)

        Returns:
            Grouped tensor of shape (batch_size, output_size) or (output_size,)
        """
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"Input tensor's last dimension ({x.shape[-1]}) does not correspond to the provided input_size ({self.input_size})"
            )

        if self.output_size > self.input_size:
            if x.dim() == 2:
                pad_size = self.output_size - self.input_size
                padded = F.pad(x, (0, pad_size))
                return padded
            else:
                pad_size = self.output_size - self.input_size
                padded = F.pad(x, (0, pad_size))
                return padded

        indices = torch.arange(self.input_size, device=x.device)
        group_indices = indices % self.output_size

        if x.dim() == 2:
            batch_size = x.shape[0]
            result = torch.zeros(
                batch_size,
                self.output_size,
                device=x.device,
                dtype=x.dtype,
            )
            result.index_add_(1, group_indices, x)
            return result
        else:
            result = torch.zeros(
                self.output_size,
                device=x.device,
                dtype=x.dtype,
            )
            result.index_add_(0, group_indices, x)
            return result
