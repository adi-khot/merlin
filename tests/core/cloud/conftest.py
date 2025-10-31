# tests/core/cloud/conftest.py
from __future__ import annotations

import perceval as pcvl
import pytest
from perceval.runtime import RemoteConfig


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
            "Quandela Cloud token not configured (RemoteConfig.set_token was not called)."
        )
    return pcvl.RemoteProcessor("sim:slos")
