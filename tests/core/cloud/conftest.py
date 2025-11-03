# tests/core/cloud/conftest.py
from __future__ import annotations

import perceval as pcvl
import pytest
from perceval.runtime import RemoteConfig
from pathlib import Path


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """By default, only skip tests that truly require a cloud token.

    - If --run-cloud-tests is NOT passed: skip tests that use the
      'remote_processor' fixture (i.e. tests that require an actual
      remote platform/token). Other tests under tests/core/cloud that do
      not require the token will still run.
    - If --run-cloud-tests IS passed: do not skip at collection time;
      tests depending on 'remote_processor' will be skipped later by the
      fixture itself when no token is configured.
    """
    if config.getoption("--run-cloud-tests"):
        return

    skip_marker = pytest.mark.skip(
        reason="requires Quandela Cloud token; run with --run-cloud-tests and configure RemoteConfig"
    )

    for item in items:
        fixturenames = getattr(item, "fixturenames", ())
        # Skip only tests that depend on remote access
        if "remote_processor" in fixturenames:
            item.add_marker(skip_marker)


def _has_configured_token() -> bool:
    """
    True only if a token has already been configured (e.g. via
    RemoteConfig.set_token(...) in user env or prior setup).
    We intentionally do NOT read environment variables here.
    """
    rc = RemoteConfig()
    token = (rc.get_token() or "").strip()
    return bool(token)


@pytest.fixture(scope="session")
def has_cloud_token() -> bool:
    """Whether a Quandela Cloud token is already configured."""
    return _has_configured_token()


@pytest.fixture
def remote_processor(has_cloud_token: bool):
    """
    Provide a RemoteProcessor if a token is configured; otherwise skip.
    We do not configure the token here—this is on the user environment.
    """
    if not has_cloud_token:
        pytest.skip(
            "Quandela Cloud token not configured: please call RemoteConfig.set_token(...) "
            "or set up your environment before running cloud tests."
        )
    return pcvl.RemoteProcessor("sim:slos")
