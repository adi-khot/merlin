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
Merlin - Photonic Quantum Neural Networks for PyTorch

A comprehensive framework for integrating photonic quantum circuits
into PyTorch neural networks with automatic differentiation support.
"""

# Core API - Most users will only need these
from .algorithms.feed_forward import FeedForwardBlock, PoolingFeedForward
from .algorithms.kernels import FeatureMap, FidelityKernel
from .algorithms.layer import QuantumLayer
from .algorithms.loss import NKernelAlignment
from .builder.circuit_builder import CircuitBuilder

# Essential enums
# Advanced components (for power users)
from .core.computation_space import ComputationSpace
from .core.generators import CircuitGenerator, CircuitType, StateGenerator, StatePattern
from .core.photonicbackend import PhotonicBackend
from .measurement import (
    Amplitudes,
    ModeExpectations,
    OutputMapper,
    Probabilities,
)
from .measurement.autodiff import AutoDiffProcess
from .measurement.process import SamplingProcess
from .measurement.strategies import MeasurementStrategy
from .pcvl_pytorch import CircuitConverter, build_slos_distribution_computegraph
from .utils.combinadics import Combinadics
from .utils.grouping import LexGrouping, ModGrouping
from .utils.torch_codes import FeatureEncoder

# Version and metadata
__version__ = "0.1.0"
__author__ = "Merlin Team"
__description__ = "Photonic Quantum Machine Learning Framework"

# Public API - what users see with `import merlin as ML`
__all__ = [
    # Core classes (most common usage)
    "QuantumLayer",
    "PhotonicBackend",
    # Configuration enums
    "CircuitType",
    "StatePattern",
    "ComputationSpace",
    "MeasurementStrategy",
    "Combinadics",
    # Advanced components
    "CircuitGenerator",
    "StateGenerator",
    "FeatureEncoder",
    "SamplingProcess",
    "AutoDiffProcess",
    "OutputMapper",
    "Probabilities",
    "ModeExpectations",
    "Amplitudes",
    "LexGrouping",
    "ModGrouping",
    "CircuitConverter",
    "build_slos_distribution_computegraph",
    "NKernelAlignment",
    "FeatureMap",
    "FidelityKernel",
    "FeedForwardBlock",
    "PoolingFeedForward",
    "CircuitBuilder",
]
