from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register top-level test options so they are available at pytest startup.

    The actual cloud-focused helpers live in `tests/core/cloud/conftest.py`,
    but registering the CLI option here ensures `--run-cloud-tests` is
    recognized when pytest parses command-line arguments.
    """
    parser.addoption(
        "--run-cloud-tests",
        action="store_true",
        default=False,
        help="include tests under tests/core/cloud",
    )
