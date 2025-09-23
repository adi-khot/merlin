"""
Merlin algorithms package containing various quantum machine learning algorithms.
"""

from .kernels import FeatureMap, FidelityKernel
from .loss import NKernelAlignment

__all__ = ["FeatureMap", "FidelityKernel", "NKernelAlignment"]
