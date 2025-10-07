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
Output mapping strategy definitions.
"""
import warnings
from enum import Enum


class OutputMappingStrategy(Enum):
    """
    Strategy for mapping quantum probability distributions to classical outputs.

    This class is deprecated and will be removed in v0.3.
    """

    LINEAR = "linear"
    GROUPING = "grouping"
    LEXGROUPING = "lexgrouping"
    MODGROUPING = "modgrouping"
    NONE = "none"
    warnings.warn('OutputMappingStrategy is deprecated and will be removed in version 0.3. '
                  'Use MeasurementStrategy instead.', DeprecationWarning, stacklevel=2)


class MeasurementStrategy(Enum):
    """Strategy for measuring quantum states or counts and possibly apply mapping to classical outputs."""

    FockDistribution = "fockdistribution"
    FockGrouping = "fockgrouping"
    ModeExpectation = "modeexpectation"
    StateVector = "statevector"
    CustomObservable = "customobservable"


class GroupingPolicy(Enum):
    """Policy for the grouping of quantum probability distributions to classical outputs."""

    ModGrouping = "modgrouping"
    LexGrouping = "lexgrouping"