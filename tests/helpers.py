"""Helper utilities for loading Merlin modules without heavy-side effects."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

# Absolute path to the source package: <repo>/merlin/merlin
_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "merlin"


def _ensure_package(package_name: str, parts: list[str]) -> None:
    """Register a lightweight namespace package for ``package_name`` if needed."""
    if package_name in sys.modules:
        return

    package_path = _PACKAGE_ROOT.joinpath(*parts)
    module = types.ModuleType(package_name)
    module.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    sys.modules[package_name] = module


def load_merlin_module(module_name: str):
    """Load a Merlin module without importing the top-level ``merlin`` package.

    This helper creates namespace packages on the fly so that modules can be
    loaded directly from the source tree without triggering heavy optional
    dependencies that are imported at package import time.
    """
    if not module_name.startswith("merlin."):
        raise ValueError("Expected a Merlin module name starting with 'merlin.'")

    if module_name in sys.modules:
        return sys.modules[module_name]

    parts = module_name.split(".")
    # Ensure every parent package is registered as a namespace package
    for i in range(1, len(parts)):
        parent_name = ".".join(parts[:i])
        _ensure_package(parent_name, parts[1:i])

    module_path = _PACKAGE_ROOT.joinpath(*parts[1:])
    if module_path.is_dir():
        file_path = module_path / "__init__.py"
    else:
        file_path = module_path.with_suffix(".py")

    if not file_path.exists():
        raise ModuleNotFoundError(f"Cannot locate source for module '{module_name}'")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load specification for '{module_name}'")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["load_merlin_module"]
