"""
Utility subpackages for Merlin.

This module exposes commonly used utility helpers so they can be imported from
``merlin.utils`` while keeping the actual implementations in their dedicated
subpackages.
"""

from .grouping import LexGrouping, ModGrouping
from .torch_utils import FeatureEncoder

__all__ = ["LexGrouping", "ModGrouping", "FeatureEncoder"]
