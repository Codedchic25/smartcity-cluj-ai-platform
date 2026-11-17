"""Centralized Pydantic data contracts and environment security settings.

Provides strict runtime validation for IoT telemetry, sensor alert thresholds,
automated QA compliance reports, and session authentication boundaries.
"""

from __future__ import annotations

import os
from typing import Literal

import streamlit as st
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from streamlit.errors import StreamlitSecretNotFoundError


class AppSecuritySettings(BaseSettings):
    """Environment-backed security and database configuration ledger."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_path: str = Field(default="app.db", alias="DATABASE_PATH")
    session_security_token: str = Field(default="", alias="SESSION_SECURITY_TOKEN")


class SensorNode(BaseModel):
    """Geospatial metadata contract for an urban telemetry station."""

    id: int = Field(ge=1)
    name: str = Field(min_length=2)
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)


class TelemetryReading(BaseModel):
    """Validated sensor telemetry payload preventing out-of-bounds corruption."""

    timestamp: str
    temperature: float = Field(ge=-50.0, le=65.0)
    noise_level: float = Field(ge=0.0, le=160.0)
    traffic_load: float = Field(ge=0.0, le=100.0)
    air_quality: float = Field(ge=0.0, le=1000.0)
    soil_moisture: float = Field(ge=0.0, le=100.0)


class SensorThreshold(BaseModel):
    """Per-station operational safety margins with domain-validated defaults."""

    temperature: float = Field(default=32.0, ge=-20.0, le=60.0)
    noise_level: float = Field(default=75.0, ge=20.0, le=150.0)
    traffic_load: float = Field(default=80.0, ge=0.0, le=100.0)
    air_quality: float = Field(default=80.0, ge=0.0, le=500.0)
    soil_moisture: float = Field(default=35.0, ge=0.0, le=100.0)


class QualityMetricCard(BaseModel):
    """Normalized compliance metric model for the Quality Control Center."""

    status: str
    passed: int = Field(ge=0)
    total: int = Field(ge=0)
    color: Literal["green", "orange", "red"] = "orange"
    detail: str | None = None


def get_session_security_token() -> str:
    """Resolve the session token from Streamlit secrets or environment variables."""
    try:
        secret_token = st.secrets.get("SESSION_SECURITY_TOKEN", "")
    except StreamlitSecretNotFoundError:
        secret_token = ""
    return str(secret_token or os.getenv("SESSION_SECURITY_TOKEN", ""))


def enforce_session_auth() -> None:
    """Validate authentication and re-synchronize the active signed URL token."""
    from src.session_auth import restore_authenticated_session

    if restore_authenticated_session(st.session_state, st.query_params):
        return

    st.warning("🔒 Restricted access! Please authenticate on the main platform index view.")
    st.stop()


def get_resolved_db_path() -> str:
    """Resolve the active SQLite file path cleanly across local and CI environments."""
    env_val = os.environ.get("DATABASE_PATH")
    if env_val:
        return env_val
    return AppSecuritySettings().database_path
