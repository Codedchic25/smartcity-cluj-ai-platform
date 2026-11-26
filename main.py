"""Smart City Cluj Telemetry Core Production Application Entrypoint Layer.

Provides secure initialization routines, constant-time cryptographic validation,
and PEP 8 compliant line lengths fully optimized for enterprise CI/CD gates.
"""

from __future__ import annotations

import hmac
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import streamlit as st
from argon2 import PasswordHasher
from argon2.exceptions import (
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from translations import TranslationProvider

# Absolute project root framework injection for relative path resilience
PROJECT_ROOT: Final[str] = str(Path(__file__).parent.resolve())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Force load variables directly from the local .env file
ENV_FILE_PATH: Final[Path] = Path(PROJECT_ROOT) / ".env"
if ENV_FILE_PATH.is_file():
    load_dotenv(dotenv_path=ENV_FILE_PATH, override=True)

from seed_db import seed_relational_infrastructure  # noqa: E402
from src.session_auth import (  # noqa: E402
    persist_authenticated_session,
    restore_authenticated_session,
)

# Secure infrastructure logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER: Final[logging.Logger] = logging.getLogger("SmartCity.Core")

# Global immutable configuration constants
ENV_DB_PATH = os.environ.get("DATABASE_PATH", "app.db")
DATABASE_PATH: Final[str] = ENV_DB_PATH if ENV_DB_PATH else "app.db"


@dataclass(frozen=True, slots=True)
class AlertThresholds:
    """Immutable data contract encapsulating baseline metric configurations."""

    temperature: float
    noise_level: float
    traffic_load: float
    air_quality: float
    soil_moisture: float


class AuthenticationService:
    """Secure operator authentication using Argon2id password hashing."""

    USERNAME_ENV: Final[str] = "PLATFORM_ADMIN_USER"
    PASSWORD_HASH_ENV: Final[str] = "PLATFORM_ADMIN_PASS"

    def __init__(self) -> None:
        """Initialize authentication credentials from environment variables."""
        try:
            username_secret = st.secrets.get(self.USERNAME_ENV, "")
            password_secret = st.secrets.get(self.PASSWORD_HASH_ENV, "")
        except StreamlitSecretNotFoundError:
            username_secret = ""
            password_secret = ""

        username_source = username_secret or os.environ.get(self.USERNAME_ENV)

        self._expected_username: Final[str] = (
            username_source.strip() if username_source and username_source.strip() else "admin"
        )

        hash_source = password_secret or os.environ.get(self.PASSWORD_HASH_ENV)

        if not hash_source or not hash_source.strip():
            raise RuntimeError(
                f"Critical Security Fault: Password hash "
                f"'{self.PASSWORD_HASH_ENV}' is not configured."
            )

        self._configured_hash: Final[str] = hash_source.strip()
        self._password_hasher: Final[PasswordHasher] = PasswordHasher()

    def verify_credentials(
        self,
        username_input: str,
        password_input: str,
    ) -> bool:
        """Verify operator credentials using Argon2id."""
        clean_user_input = username_input.strip()

        if not hmac.compare_digest(
            clean_user_input,
            self._expected_username,
        ):
            return False

        if not password_input:
            return False

        try:
            return self._password_hasher.verify(
                self._configured_hash,
                password_input,
            )
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False


class CityRepository:
    """Data Access Object (DAO) managing atomic relational operations against SQLite."""

    def __init__(self, target_database_path: str = DATABASE_PATH) -> None:
        """Instantiate the persistent ledger link wrapper mapping explicit paths."""
        import sqlite3

        self._sqlite3 = sqlite3
        self._db_file: Final[str] = target_database_path

    def _connect(self) -> Any:
        """Generate an active connection pipeline enforcing strict write timeouts."""
        return self._sqlite3.connect(self._db_file, timeout=5.0)

    @property
    def database_path(self) -> str:
        """Expose the configured SQLite path for aligned initialization and seeding."""
        return self._db_file

    def initialize_database(self) -> None:
        """Execute defensive DDL blueprints to establish baseline structural layouts."""
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sensors (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL
                );
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS city_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sensor_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    noise_level REAL NOT NULL,
                    traffic_load REAL NOT NULL,
                    air_quality REAL NOT NULL,
                    soil_moisture REAL NOT NULL,
                    FOREIGN KEY (sensor_id) REFERENCES sensors(id)
                    ON DELETE CASCADE
                );
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    sensor_id INTEGER PRIMARY KEY,
                    temp_limit REAL NOT NULL DEFAULT 32.0,
                    noise_limit REAL NOT NULL DEFAULT 75.0,
                    traffic_limit REAL NOT NULL DEFAULT 80.0,
                    air_limit REAL NOT NULL DEFAULT 80.0,
                    soil_limit REAL NOT NULL DEFAULT 35.0,
                    FOREIGN KEY (sensor_id) REFERENCES sensors(id)
                    ON DELETE CASCADE
                );
                """
            )
            connection.commit()


class SmartCityApplicationEngine:
    """Core domain orchestrator encapsulating business boundaries and streams."""

    def __init__(self, repository: CityRepository, auth_service: AuthenticationService) -> None:
        """Inject explicit decoupling interfaces targeting modular system instances."""
        self._repository: Final[CityRepository] = repository
        self._auth: Final[AuthenticationService] = auth_service

    def initialize(self) -> None:
        """Fire safe proactive schema assertions before processing operations."""
        self._repository.initialize_database()
        seed_relational_infrastructure(database_path=self._repository.database_path)

    def handle_operator_login(self, user_token: str, password_token: str) -> bool:
        """Forward credentials tracking downstream token matching execution grids."""
        return self._auth.verify_credentials(user_token, password_token)


def process_url_security_gate() -> None:
    """Restore a login from an unexpired signed token in the current URL."""
    restore_authenticated_session(st.session_state, st.query_params)


def render_login_screen_interface(engine: SmartCityApplicationEngine) -> None:
    """Render a restricted administrative gateway login frame protecting components."""
    title_html = (
        "<h2 style='text-align: center; color: white;'>🔒 Administrative Access Portal</h2>"
    )
    st.markdown(title_html, unsafe_allow_html=True)
    st.write(
        "Please authenticate using valid operational tokens to unlock urban telemetry metrics."
    )

    with st.form("security_authentication_frame"):
        username_field = st.text_input("Username Input Ledger:")
        password_field = st.text_input("Password Security Key:", type="password")

        dispatch_trigger = st.form_submit_button("Authenticate System Profile", type="primary")

        if dispatch_trigger:
            if engine.handle_operator_login(username_field, password_field):
                try:
                    persist_authenticated_session(
                        username_field,
                        st.session_state,
                        st.query_params,
                    )
                except RuntimeError as exc:
                    LOGGER.error("Unable to create signed login session: %s", exc)
                    st.error(
                        "Secure login session could not be created. Check SESSION_SECURITY_TOKEN."
                    )
                    return
                st.success("✨ Access granted. Initializing system boards...")
                st.rerun()
            else:
                st.error("❌ Authentication failure: Invalid credentials token.")


def render_production_dashboard(engine: SmartCityApplicationEngine) -> None:
    """Render the central core platform interface wrapper following authentication."""
    _ = engine
    from app.ai.ai_interface import render_full_global_sidebar

    if "lang" not in st.session_state:
        st.session_state["lang"] = "RO"

    provider_instance = TranslationProvider(default_language=st.session_state["lang"])

    app_title = provider_instance.get("app_title", "Smart City Core Platform")
    title_payload = f"<h1>🏙️ {app_title}</h1>"
    st.markdown(title_payload, unsafe_allow_html=True)

    operator_session = st.session_state.get("operator_name", "System")
    st.markdown(f"🤖 **Status:** Authorized Session active as *{operator_session}*")
    st.divider()

    selected_zone = render_full_global_sidebar(provider_instance.get_all_translations())

    st.subheader(f"📍 Selected Zone Focus: {selected_zone}")
    st.info(
        "💡 System components synchronized cleanly against local app.db. "
        "Navigate through the sidebar menu to access Analytics and Settings."
    )


def main() -> None:
    """Main application lifecycle execution bounds initialization router loop."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    process_url_security_gate()

    if "app_engine" not in st.session_state:
        try:
            repository_layer = CityRepository()
            security_gateway = AuthenticationService()
            st.session_state["app_engine"] = SmartCityApplicationEngine(
                repository_layer, security_gateway
            )
            st.session_state["app_engine"].initialize()
        except Exception as exc:
            st.set_page_config(
                page_title="Smart City - Error",
                layout="centered",
                page_icon="❌",
            )
            st.error(f"❌ Critical Infrastructure Failure: {exc}")
            return

    application_engine = st.session_state["app_engine"]

    if not st.session_state["authenticated"]:
        st.set_page_config(
            page_title="Smart City Cluj - Gateway",
            layout="centered",
            page_icon="🔒",
        )
        render_login_screen_interface(application_engine)
    else:
        st.set_page_config(
            page_title="Smart City Cluj-Napoca",
            layout="wide",
            page_icon="🏙️",
        )
        render_production_dashboard(application_engine)


if __name__ == "__main__":
    main()
