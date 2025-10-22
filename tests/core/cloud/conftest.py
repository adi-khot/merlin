# Shared setup for cloud tests.
#
# - Reads QUANDELA_TOKEN env var (or the fallback TOKEN below).
# - Sets RemoteConfig token once (session scope).
# - Exposes `has_cloud_token` fixture and `remote_processor` fixture (sim:slos).
#
# Any test that needs the cloud can depend on `remote_processor`; it will
# auto-skip if no token is present.

import os
import pytest
import perceval as pcvl
from perceval.runtime import RemoteConfig

# Fallback (ONLY used if QUANDELA_TOKEN env var is unset/empty).
TOKEN_FALLBACK = "<PUT_TOKEN_HERE>"
TOKEN_FALLBACK = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6Mzk1LCJleHAiOjE3NjEyMjgyNzUuMjY4MDkzfQ.vPEHupHJhtXAFVMqyhav7s97cfp_CtJFxA9eH7328JSehdxKry192BKZ8i99KarjlMBkKoIyEJEmU45O3aDjSw"

def _resolved_token() -> str:
    t = os.environ.get("QUANDELA_TOKEN", "").strip()
    return t if t else TOKEN_FALLBACK


@pytest.fixture(scope="session", autouse=True)
def _configure_quandela_token():
    token = _resolved_token()
    if token and token != "<PUT_TOKEN_HERE>":
        RemoteConfig.set_token(token)


@pytest.fixture(scope="session")
def has_cloud_token() -> bool:
    token = _resolved_token()
    return bool(token and token != "<PUT_TOKEN_HERE>")


@pytest.fixture
def remote_processor(has_cloud_token):
    if not has_cloud_token:
        pytest.skip("No QUANDELA_TOKEN configured; skipping cloud-dependent test.")
    # Default to the exact SLOS simulator used throughout tests
    return pcvl.RemoteProcessor("sim:slos")
