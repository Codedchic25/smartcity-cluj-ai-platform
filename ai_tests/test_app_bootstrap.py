"""Regression coverage for application startup and missing active sensors."""

from __future__ import annotations

import hashlib
import importlib.util
import sqlite3
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from main import AuthenticationService, CityRepository, SmartCityApplicationEngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StubTranslationProvider:
    """Provide minimal labels for page display tests without UI dependencies."""

    def __init__(self, default_language: str = "EN") -> None:
        """Accept the language argument used by the page components."""
        self._default_language = default_language

    def get(self, key: str, default: str | None = None) -> str:
        """Return a stable label for UI calls."""
        return default or key

    def get_all_translations(self) -> dict[str, str]:
        """Return an empty translation map for these no-render tests."""
        return {}


def _load_page_module(module_name: str, file_name: str) -> ModuleType:
    """Load a multipage script as a module without starting its Streamlit page."""
    page_path = PROJECT_ROOT / "pages" / file_name
    spec = importlib.util.spec_from_file_location(module_name, page_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load page module: {page_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_streamlit_page(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace page-only UI operations and capture warnings for assertions."""
    warnings: list[str] = []
    monkeypatch.setattr(module, "TranslationProvider", StubTranslationProvider)
    monkeypatch.setattr(module, "render_full_global_sidebar", lambda _translations: None)
    monkeypatch.setattr(module.st, "warning", warnings.append)
    monkeypatch.setattr(module.st, "title", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module.st, "divider", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module.st, "markdown", lambda *_args, **_kwargs: None)
    return warnings


def test_engine_initialization_seeds_empty_database_without_overwriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bootstrap an empty database and preserve settings on subsequent startup."""
    database_path = tmp_path / "smart_city.db"
    monkeypatch.setenv(
        "PLATFORM_ADMIN_PASS",
        hashlib.sha256(b"test-password").hexdigest(),
    )
    engine = SmartCityApplicationEngine(
        CityRepository(str(database_path)),
        AuthenticationService(),
    )

    engine.initialize()
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sensors").fetchone()[0] == 7
        assert connection.execute("SELECT COUNT(*) FROM settings").fetchone()[0] == 7
        assert connection.execute("SELECT COUNT(*) FROM city_stats").fetchone()[0] == 84
        connection.execute("UPDATE settings SET temp_limit = 47.5 WHERE sensor_id = 1")

    engine.initialize()
    with sqlite3.connect(database_path) as connection:
        assert (
            connection.execute("SELECT temp_limit FROM settings WHERE sensor_id = 1").fetchone()[0]
            == 47.5
        )


def test_dashboard_warns_when_active_sensor_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dashboard warns and returns instead of dereferencing a missing sensor."""
    module = _load_page_module("dashboard_page_test", "1_Dashboard.py")
    warnings = _stub_streamlit_page(module, monkeypatch)
    monkeypatch.setattr(module, "render_full_global_sidebar", lambda _translations: "missing")
    monkeypatch.setattr(module.DashboardPage, "enforce_authentication", lambda _self: None)

    page = module.DashboardPage()
    page._repository = SimpleNamespace(load_all_sensors=lambda: [])
    page.display()

    assert warnings
    assert "No monitoring stations" in warnings[0]


def test_settings_warns_when_active_sensor_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings does not render sensor forms for an absent selection."""
    module = _load_page_module("settings_page_test", "2_Settings.py")
    warnings = _stub_streamlit_page(module, monkeypatch)
    monkeypatch.setattr(module.SettingsPage, "enforce_authentication", lambda _self: None)

    page = module.SettingsPage()
    page._repository = SimpleNamespace(load_all_sensors=lambda: [])
    monkeypatch.setattr(page, "_render_export_section", lambda _provider: None)
    page.display()

    assert warnings
    assert "No active monitoring station" in warnings[0]


def test_analytics_warns_when_active_sensor_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Analytics displays a warning instead of using a fallback sensor identifier."""
    module = _load_page_module("analytics_page_test", "3_Analytics.py")
    warnings = _stub_streamlit_page(module, monkeypatch)
    monkeypatch.setattr(module.AnalyticsPage, "enforce_authentication", lambda _self: None)

    page = module.AnalyticsPage()
    page._db_path = Path("missing-test-database.db")
    page.display()

    assert warnings
    assert "No active monitoring station" in warnings[0]
