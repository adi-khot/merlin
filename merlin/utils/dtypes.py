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
Utilities for converting between various dtype representations and torch dtypes.
"""

from __future__ import annotations

import numpy as np
import torch

# Canonical mapping from generic dtype representations to torch dtypes.
_TORCH_DTYPE_MAP: dict[object, torch.dtype] = {
    "float": torch.float64,
    "complex": torch.complex128,
    "float64": torch.float64,
    "float32": torch.float32,
    "complex128": torch.complex128,
    "complex64": torch.complex64,
    torch.float64: torch.float64,
    torch.float32: torch.float32,
    torch.complex128: torch.complex128,
    torch.complex64: torch.complex64,
    np.float64: torch.float64,
    np.float32: torch.float32,
    np.complex128: torch.complex128,
    np.complex64: torch.complex64,
}


def _normalize_torch_dtype(dtype_like: object) -> torch.dtype:
    """
    Convert the input into a torch dtype or raise TypeError if unsupported.
    """
    if dtype_like is None:
        raise TypeError("Unsupported dtype None.")

    if isinstance(dtype_like, torch.dtype):
        return dtype_like

    try:
        return to_torch_dtype(dtype_like)
    except TypeError as exc:
        raise TypeError(f"Unsupported dtype {dtype_like!r}.") from exc


def _build_float_complex_pairs() -> tuple[tuple[torch.dtype, torch.dtype], ...]:
    """
    Construct the supported float/complex dtype pairs for the current torch build.
    """
    pairs: list[tuple[torch.dtype, torch.dtype]] = []

    if hasattr(torch, "complex32"):
        pairs.append((torch.float16, torch.complex32))

    pairs.extend([
        (torch.float32, torch.complex64),
        (torch.float64, torch.complex128),
    ])
    return tuple(pairs)


_FLOAT_COMPLEX_PAIRS = _build_float_complex_pairs()


def to_torch_dtype(
    dtype_like: object, *, default: torch.dtype | None = None
) -> torch.dtype:
    """
    Convert common dtype representations (strings, numpy dtypes, torch dtypes) into torch dtypes.

    Args:
        dtype_like: Input representation to convert.
        default: Fallback dtype if the representation is unknown. Defaults to torch.float32.

    Returns:
        torch.dtype corresponding to the requested representation.

    Raises:
        TypeError: If the value cannot be mapped and no default is provided.
    """
    if dtype_like is None:
        if default is not None:
            return default
        return torch.float32

    mapped = _TORCH_DTYPE_MAP.get(dtype_like)
    if mapped is not None:
        return mapped

    # Torch dtypes sometimes expose names like torch.float64 but may not match by identity.
    if isinstance(dtype_like, torch.dtype):
        return dtype_like

    # numpy dtype instances may not match direct mapping if they carry metadata, try the name.
    if isinstance(dtype_like, (np.dtype,)):
        mapped = _TORCH_DTYPE_MAP.get(dtype_like.type)
        if mapped is not None:
            return mapped

    if default is not None:
        return default

    raise TypeError(f"Unsupported dtype representation: {dtype_like!r}")


def complex_dtype_for(dtype_like: object) -> torch.dtype:
    """
    Return the matching complex dtype for the provided float or complex dtype.

    Args:
        dtype_like: Representation of a torch dtype (string, numpy dtype, torch dtype, ...).

    Returns:
        torch complex dtype corresponding to the provided representation.

    Raises:
        TypeError: If the dtype cannot be mapped to a supported float/complex pair.
    """
    dtype = _normalize_torch_dtype(dtype_like)

    for float_dtype, complex_dtype in _FLOAT_COMPLEX_PAIRS:
        if dtype in (float_dtype, complex_dtype):
            return complex_dtype

    supported = ", ".join(
        f"{float_dtype}->{complex_dtype}"
        for float_dtype, complex_dtype in _FLOAT_COMPLEX_PAIRS
    )
    raise TypeError(f"Unsupported dtype {dtype}. Supported mappings: {supported}.")


def float_dtype_for(dtype_like: object) -> torch.dtype:
    """
    Return the matching float dtype for the provided float or complex dtype.

    Args:
        dtype_like: Representation of a torch dtype (string, numpy dtype, torch dtype, ...).

    Returns:
        torch float dtype corresponding to the provided representation.

    Raises:
        TypeError: If the dtype cannot be mapped to a supported float/complex pair.
    """
    dtype = _normalize_torch_dtype(dtype_like)

    for float_dtype, complex_dtype in _FLOAT_COMPLEX_PAIRS:
        if dtype in (float_dtype, complex_dtype):
            return float_dtype

    supported = ", ".join(
        f"{complex_dtype}->{float_dtype}"
        for float_dtype, complex_dtype in _FLOAT_COMPLEX_PAIRS
    )
    raise TypeError(f"Unsupported dtype {dtype}. Supported mappings: {supported}.")


def resolve_float_complex(dtype: torch.dtype) -> tuple[torch.dtype, torch.dtype]:
    """
    Given a torch dtype representing either the float or complex side, return the matching pair.

    Args:
        dtype: torch float or complex dtype.

    Returns:
        Tuple (float_dtype, complex_dtype) ensuring the pair is internally consistent.

    Raises:
        TypeError: If the dtype is not one of the supported float/complex types.
    """
    for float_dtype, complex_dtype in _FLOAT_COMPLEX_PAIRS:
        if dtype in (float_dtype, complex_dtype):
            return float_dtype, complex_dtype

    supported = ", ".join(
        f"({float_dtype},{complex_dtype})"
        for float_dtype, complex_dtype in _FLOAT_COMPLEX_PAIRS
    )
    raise TypeError(
        f"Unsupported dtype {dtype}. Supported float/complex pairs: {supported}."
    )


__all__ = [
    "to_torch_dtype",
    "complex_dtype_for",
    "float_dtype_for",
    "resolve_float_complex",
]
