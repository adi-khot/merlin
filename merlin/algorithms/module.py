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

from __future__ import annotations

import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, ClassVar

import torch.nn as nn


class MerlinModule(nn.Module):
    # Allow both tuple[str, bool] and str for backwards compatibility.
    # Mapping[...] here lets subclasses override with a plain dict[...] safely.
    _deprecated_params: ClassVar[Mapping[str, tuple[str, bool] | str]] = {}
    """Generic MerLin module with shared utility functions

    Merlin remote execution policy:
      - `_force_simulation` (bool) defaults to False. When True, the layer MUST run locally.
        The variable is set with property (getter and setter): `force_local`.
      - `supports_offload()` reports whether remote offload is possible (via `export_config()`).
      - `should_offload(processor, shots)` encapsulates the current offload policy:
            return supports_offload() and not force_local
      - `as_simulation()` provide local context forcing use as simulation
    """

    # -------------------- Execution policy & helpers --------------------

    @property
    def force_local(self) -> bool:
        """When True, this layer must run locally (Merlin will not offload it)."""
        return self._force_simulation

    @force_local.setter
    def force_local(self, value: bool) -> None:
        self._force_simulation = bool(value)

    @contextmanager
    def as_simulation(self):
        """Temporarily force local simulation within the context."""
        prev = self.force_local
        self.force_local = True
        try:
            yield self
        finally:
            self.force_local = prev

    # Offload capability & policy (queried by MerlinProcessor)
    def supports_offload(self) -> bool:
        """Return True if this layer is technically offloadable."""
        return hasattr(self, "export_config") and callable(self.export_config)

    def should_offload(self, _processor=None, _shots=None) -> bool:
        """Return True if this layer should be offloaded under current policy."""
        return self.supports_offload() and not self.force_local

    @classmethod
    def _validate_kwargs(cls, method_name: str, kwargs: dict[str, Any]) -> None:
        if not kwargs:
            return

        deprecated_raise: list[str] = []
        deprecated_warn: list[str] = []
        unknown: list[str] = []

        for key in sorted(kwargs):
            full_name = f"{method_name}.{key}"
            if full_name in cls._deprecated_params:
                # support old-style str values for backwards compatibility
                val = cls._deprecated_params[full_name]
                if isinstance(val, tuple):
                    message, raise_error = val
                else:
                    message, raise_error = (str(val), True)

                if raise_error:
                    deprecated_raise.append(
                        f"Parameter '{key}' is deprecated. {message}"
                    )
                else:
                    deprecated_warn.append(
                        f"Parameter '{key}' is deprecated. {message}"
                    )
            else:
                unknown.append(key)

        # Emit non-fatal deprecation warnings
        if deprecated_warn:
            warnings.warn(" ".join(deprecated_warn), DeprecationWarning, stacklevel=2)

        # Raise for deprecated parameters that are marked fatal
        if deprecated_raise:
            raise ValueError(" ".join(deprecated_raise))

        if unknown:
            unknown_list = ", ".join(unknown)
            raise ValueError(
                f"Unexpected keyword argument(s): {unknown_list}. "
                f"Check the {cls} signature for supported parameters."
            )

    def __init__(self):
        super().__init__()

        # execution policy: when True, always simulate locally (do not offload)
        self._force_simulation: bool = False
