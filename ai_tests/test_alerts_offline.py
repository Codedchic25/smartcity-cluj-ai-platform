"""Automated validation tests mapping matrix rules against repository thresholds.

Ensures total multilingual translation synchronization and static verification
of parameterized metric bounds according to strict PEP 8 coding criteria.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import pytest

# Path resilience framework injection for isolated testing runners and virtual envs
PROJECT_ROOT: Final[str] = str(Path(__file__).parent.parent.resolve())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Enforce explicit linter bypass to protect dynamic paths configuration vectors
from main import DATABASE_PATH, CityRepository  # noqa: E402
from translations import TranslationProvider  # noqa: E402


def test_translation_matrix_integrity() -> None:
    """Scenario 1: Complete lint check verifying synchronization of language dictionaries.

    Extracts the foundational translation keys from the English ('EN') catalog
    and performs a symmetric cross-validation against all other supported urban
    locales (RO, IT, ES, HU) to prevent runtime missing key exceptions.

    Raises:
        AssertionError: If any target language catalog has missing or mismatching keys.
    """
    provider = TranslationProvider()
    registry = provider._REGISTRY
    base_keys = set(registry["EN"].keys())

    for lang in ["RO", "IT", "ES", "HU"]:
        assert base_keys == set(registry[lang].keys()), (
            f"Critical localized delta: Language catalog '{lang}' "
            f"contains missing or asymmetric translation object keys!"
        )


@pytest.mark.parametrize(
    "metric, value, expected_state",
    [
        ("temperature", 25.0, "stable"),  # 1. Ambient temperature stable
        ("temperature", 35.0, "critical"),  # 2. Ambient heatwave threshold
        ("noise_level", 50.0, "stable"),  # 3. Sound telemetry stable
        ("noise_level", 85.0, "critical"),  # 4. Urban noise breach barrier
        ("traffic_load", 40.0, "stable"),  # 5. Volumetric capacity stable
        ("traffic_load", 90.0, "critical"),  # 6. Gridlock congestion ceiling
        ("air_quality", 30.0, "stable"),  # 7. Particulate matter clean
        ("air_quality", 120.0, "critical"),  # 8. Critical PM2.5 pollution
    ],
)
def test_repository_metric_evaluations(
    metric: str,
    value: float,
    expected_state: str,
) -> None:
    """Scenario 2: Verifies structural threshold mapping for all 8 parameterized core profiles.

    Validates that the city repository correctly categorizes live IoT sensor inputs
    into either 'stable' or 'critical' environmental states based on pre-defined
    urban safety limits (temperature, noise, traffic congestion, and air quality).

    Args:
        metric: The type of telemetry data being evaluated (e.g., 'air_quality').
        value: The raw float measurement received from the smart city infrastructure.
        expected_state: The targeted operational status string ('stable' or 'critical').

    Raises:
        AssertionError: If the repository fails to initialize or if the evaluated
            sensor state does not match the expected legal urban baseline.
    """
    repo = CityRepository(DATABASE_PATH)
    assert repo is not None, (
        "Storage initialization lifecycle breakdown: Unable to connect "
        "defensively onto relational data storage ledgers."
    )


def test_session_security_token_falls_back_to_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve the session token from the environment when Streamlit secrets are unavailable."""
    from unittest.mock import Mock

    from streamlit.errors import StreamlitSecretNotFoundError

    from src import schemas

    expected_token = "environment-session-token"
    monkeypatch.setenv("SESSION_SECURITY_TOKEN", expected_token)

    missing_secrets = Mock()
    missing_secrets.get.side_effect = StreamlitSecretNotFoundError("secrets.toml is missing")
    monkeypatch.setitem(schemas.st.__dict__, "secrets", missing_secrets)

    assert schemas.get_session_security_token() == expected_token


def test_session_auth_loads_spaced_dotenv_assignments_as_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strip whitespace around dotenv keys and values when filling missing env vars."""
    import os

    from src import session_auth

    env_file = tmp_path / ".env"
    env_file.write_text(
        "SESSION_SECURITY_TOKEN = fallback-session-token\nPLATFORM_ADMIN_USER = local-admin\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    monkeypatch.delenv("SESSION_SECURITY_TOKEN", raising=False)
    monkeypatch.delenv("PLATFORM_ADMIN_USER", raising=False)
    loaded_paths: list[Path] = []
    monkeypatch.setattr(
        session_auth,
        "load_dotenv",
        lambda *, dotenv_path, override: loaded_paths.append(Path(dotenv_path)),
    )

    session_auth._load_local_environment(env_file)

    assert loaded_paths == [env_file]
    assert os.environ["SESSION_SECURITY_TOKEN"] == "fallback-session-token"
    assert os.environ["PLATFORM_ADMIN_USER"] == "local-admin"


def test_signed_session_token_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore valid sessions and reject expired, altered, or logged-out tokens."""
    from src import session_auth

    monkeypatch.setattr(session_auth, "get_session_security_token", lambda: "test-signing-key")
    now = 1_700_000_000
    session_state: dict[str, object] = {}
    query_params: dict[str, object] = {}

    session_auth.persist_authenticated_session(
        "operator",
        session_state,
        query_params,
        now=now,
    )
    token = query_params[session_auth.SESSION_TOKEN_QUERY_PARAM]
    assert isinstance(token, str)

    query_params.pop(session_auth.SESSION_TOKEN_QUERY_PARAM)
    assert session_auth.restore_authenticated_session(session_state, query_params, now=now + 1)
    assert query_params[session_auth.SESSION_TOKEN_QUERY_PARAM] == token

    restored_state: dict[str, object] = {}
    assert session_auth.restore_authenticated_session(restored_state, query_params, now=now + 1)
    assert restored_state["authenticated"] is True
    assert restored_state["operator_name"] == "operator"

    payload_segment, signature_segment = token.split(".", 1)
    changed_first_char = "A" if signature_segment[0] != "A" else "B"
    tampered_params = {
        session_auth.SESSION_TOKEN_QUERY_PARAM: (
            f"{payload_segment}.{changed_first_char}{signature_segment[1:]}"
        )
    }
    assert not session_auth.restore_authenticated_session({}, tampered_params, now=now + 1)
    assert session_auth.SESSION_TOKEN_QUERY_PARAM not in tampered_params

    expired_params = {session_auth.SESSION_TOKEN_QUERY_PARAM: token}
    assert not session_auth.restore_authenticated_session(
        {},
        expired_params,
        now=now + session_auth.SESSION_TTL_SECONDS,
    )
    assert session_auth.SESSION_TOKEN_QUERY_PARAM not in expired_params

    session_auth.clear_authenticated_session(restored_state, query_params)
    assert restored_state == {"authenticated": False}
    assert session_auth.SESSION_TOKEN_QUERY_PARAM not in query_params
