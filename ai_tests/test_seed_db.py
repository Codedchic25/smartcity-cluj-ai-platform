"""Regression coverage for safe, repeatable database seeding behavior."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import seed_db


def test_default_seed_preserves_existing_settings_and_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seed an empty DB once, then skip existing data without overwriting it."""
    database_path = tmp_path / "smart_city.db"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))

    assert seed_db.seed_relational_infrastructure() is True

    with sqlite3.connect(database_path) as connection:
        telemetry_id = connection.execute("SELECT id FROM city_stats LIMIT 1").fetchone()[0]
        connection.execute("UPDATE settings SET temp_limit = 47.5 WHERE sensor_id = 1")
        connection.execute("UPDATE city_stats SET temperature = 99.0 WHERE id = ?", (telemetry_id,))

    assert seed_db.seed_relational_infrastructure() is False

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sensors").fetchone()[0] == 7
        assert connection.execute("SELECT COUNT(*) FROM settings").fetchone()[0] == 7
        assert connection.execute("SELECT COUNT(*) FROM city_stats").fetchone()[0] == 84
        assert (
            connection.execute("SELECT temp_limit FROM settings WHERE sensor_id = 1").fetchone()[0]
            == 47.5
        )
        assert (
            connection.execute(
                "SELECT temperature FROM city_stats WHERE id = ?",
                (telemetry_id,),
            ).fetchone()[0]
            == 99.0
        )


def test_force_flag_resets_existing_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The explicit CLI force option replaces existing data with demo defaults."""
    database_path = tmp_path / "smart_city.db"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    assert seed_db.seed_relational_infrastructure() is True

    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE settings SET temp_limit = 47.5 WHERE sensor_id = 1")
        connection.execute("UPDATE city_stats SET timestamp = 'custom-marker' WHERE sensor_id = 1")

    assert seed_db.main(["--force"]) == 0

    with sqlite3.connect(database_path) as connection:
        assert (
            connection.execute("SELECT temp_limit FROM settings WHERE sensor_id = 1").fetchone()[0]
            == 32.0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM city_stats WHERE timestamp = 'custom-marker'"
            ).fetchone()[0]
            == 0
        )
        assert connection.execute("SELECT COUNT(*) FROM city_stats").fetchone()[0] == 84


def test_seed_errors_propagate_from_library_function(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Library callers receive database exceptions rather than a process exit."""
    database_path = tmp_path / "missing" / "smart_city.db"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))

    with pytest.raises(sqlite3.OperationalError):
        seed_db.seed_relational_infrastructure()
