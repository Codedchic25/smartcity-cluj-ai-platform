"""Administration and system boundaries panel with dynamic per-sensor threshold controls.

Provides production-ready OOP repository wrappers, Pydantic runtime validation,
structured memory-buffered file export layers, and ledger synchronization.
"""

from __future__ import annotations

import io
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Final

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Final[str] = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.ai.ai_interface import render_full_global_sidebar  # noqa: E402
from src.schemas import enforce_session_auth  # noqa: E402
from translations import TranslationProvider  # noqa: E402

LOGGER: Final[logging.Logger] = logging.getLogger("SmartCity.Settings")


class SettingsEnvironmentConfig(BaseSettings):
    """Environment-backed database configuration settings.

    Attributes:
        database_path: Filesystem location of the active SQLite database.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_path: str = Field(default="app.db", alias="DATABASE_PATH")


class SensorSummary(BaseModel):
    """Lightweight Pydantic contract representing an urban monitoring station.

    Attributes:
        id: Unique positive integer identifier of the station.
        name: Human-readable urban location label.
    """

    id: int = Field(ge=1)
    name: str = Field(min_length=2)


class SensorThreshold(BaseModel):
    """Per-station operational safety margins with domain-validated boundaries.

    Attributes:
        temperature: Maximum safe air temperature in degrees Celsius.
        noise_level: Maximum safe acoustic noise level in decibels (dB).
        traffic_load: Maximum safe road congestion percentage (0 to 100).
        air_quality: Maximum safe particulate matter PM2.5 index.
        soil_moisture: Minimum safe volumetric soil moisture percentage (0 to 100).
    """

    temperature: float = Field(default=32.0, ge=-20.0, le=60.0)
    noise_level: float = Field(default=75.0, ge=20.0, le=150.0)
    traffic_load: float = Field(default=80.0, ge=0.0, le=100.0)
    air_quality: float = Field(default=80.0, ge=0.0, le=500.0)
    soil_moisture: float = Field(default=35.0, ge=0.0, le=100.0)


class SettingsRepository:
    """Manages transactional state persistence for granular system boundaries."""

    def __init__(self) -> None:
        """Initialize the repository infrastructure link wrappers using strict typing."""
        env_path: Final[str | None] = os.environ.get("DATABASE_PATH")
        default_path: Final[str] = env_path or SettingsEnvironmentConfig().database_path
        self._db_path: Final[Path] = Path(default_path)

    @property
    def _active_db_path(self) -> Path:
        """Resolve the active database path safely from structural scopes.

        Returns:
            Path: Verified filesystem path pointing to an existing SQLite file.
        """
        return self._db_path if self._db_path.is_file() else Path("app.db")

    def load_all_sensors(self) -> list[SensorSummary]:
        """Fetch and validate all registered stations to generate form navigation.

        Returns:
            list[SensorSummary]: Ordered list of validated station summaries.
        """
        query: Final[str] = "SELECT id, name FROM sensors ORDER BY id"
        try:
            with sqlite3.connect(str(self._active_db_path), timeout=10.0) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(query).fetchall()
                return [SensorSummary.model_validate(dict(row)) for row in rows]
        except (sqlite3.Error, ValidationError) as err:
            LOGGER.error("Failed to fetch and validate sensors: %s", err)
            return []

    def load_sensor_threshold(self, sensor_id: int) -> SensorThreshold:
        """Fetch active alert boundary rules mapped into a Pydantic contract.

        Args:
            sensor_id: Unique database identifier of the target sensor station.

        Returns:
            SensorThreshold: Validated threshold boundaries or safe domain defaults.
        """
        query: Final[str] = """
            SELECT
                temp_limit AS temperature,
                noise_limit AS noise_level,
                traffic_limit AS traffic_load,
                air_limit AS air_quality,
                soil_limit AS soil_moisture
            FROM settings WHERE sensor_id = ? LIMIT 1
        """
        try:
            with sqlite3.connect(str(self._active_db_path), timeout=10.0) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(query, (sensor_id,)).fetchone()
                if row is not None:
                    return SensorThreshold.model_validate(dict(row))
        except (sqlite3.Error, ValidationError) as err:
            LOGGER.error("Threshold validation failure for sensor %d: %s", sensor_id, err)
        return SensorThreshold()

    def save_sensor_threshold(self, sensor_id: int, threshold: SensorThreshold) -> bool:
        """Persist validated configurations securely inside an atomic upsert context.

        Args:
            sensor_id: Unique database identifier of the target sensor station.
            threshold: Validated Pydantic threshold boundary contract.

        Returns:
            bool: True if the ACID transaction committed cleanly, False otherwise.
        """
        query: Final[str] = """
            INSERT INTO settings (
                sensor_id, temp_limit, noise_limit, traffic_limit, air_limit, soil_limit
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(sensor_id) DO UPDATE SET
                temp_limit = excluded.temp_limit,
                noise_limit = excluded.noise_limit,
                traffic_limit = excluded.traffic_limit,
                air_limit = excluded.air_limit,
                soil_limit = excluded.soil_limit
        """
        try:
            with sqlite3.connect(str(self._active_db_path), timeout=15.0) as conn:
                conn.execute(
                    query,
                    (
                        sensor_id,
                        threshold.temperature,
                        threshold.noise_level,
                        threshold.traffic_load,
                        threshold.air_quality,
                        threshold.soil_moisture,
                    ),
                )
                conn.commit()
                return True
        except sqlite3.Error as err:
            LOGGER.critical("ACID update exception for sensor %d: %s", sensor_id, err)
            return False

    def fetch_raw_telemetry_dump(self) -> pd.DataFrame:
        """Extract a flattened join matrix representing historical sensory data.

        Returns:
            pd.DataFrame: Joined chronological telemetry records across all stations.
        """
        query: Final[str] = """
            SELECT s.name AS station_name, c.timestamp, c.temperature,
                   c.noise_level, c.traffic_load, c.air_quality, c.soil_moisture
            FROM city_stats c
            JOIN sensors s ON c.sensor_id = s.id
            ORDER BY c.timestamp DESC
        """
        try:
            with sqlite3.connect(str(self._active_db_path), timeout=15.0) as conn:
                return pd.read_sql_query(query, conn)
        except (sqlite3.Error, pd.errors.DatabaseError) as err:
            LOGGER.error("Failed to compile analytics join dump: %s", err)
            return pd.DataFrame()


class DataExportService:
    """Isolated binary spreadsheet compiling engine preventing memory leaks."""

    @staticmethod
    def generate_csv_bytes(df: pd.DataFrame) -> bytes:
        """Convert a flattened chronological dataframe dump cleanly into UTF-8 CSV bytes.

        Args:
            df: Source telemetry dataframe to serialize.

        Returns:
            bytes: UTF-8 encoded CSV payload or empty bytes if dataframe is empty.
        """
        if df.empty:
            return b""
        return df.to_csv(index=False).encode("utf-8")

    @staticmethod
    def generate_excel_bytes(df: pd.DataFrame) -> bytes:
        """Compile a styled spreadsheet using memory-buffered XlsxWriter engines.

        Args:
            df: Source telemetry dataframe to serialize into Excel format.

        Returns:
            bytes: Binary XLSX workbook payload or empty bytes upon failure.
        """
        if df.empty:
            return b""
        output: Final[io.BytesIO] = io.BytesIO()
        try:
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name="Urban Telemetry", index=False)
                worksheet = writer.sheets["Urban Telemetry"]
                for idx, col in enumerate(df.columns):
                    series = df[col]
                    max_len: int = max(int(series.astype(str).map(len).max()), len(col)) + 3
                    worksheet.set_column(idx, idx, max_len)
            return output.getvalue()
        except Exception as exc:
            LOGGER.error("Spreadsheet compiler execution failure: %s", exc)
            return b""


class SettingsPage:
    """Orchestrates configuration state boundaries and administrative web forms."""

    def __init__(self) -> None:
        """Initialize the layout compilation loops and services framework."""
        self._repository: Final[SettingsRepository] = SettingsRepository()
        self._export_service: Final[DataExportService] = DataExportService()

    @staticmethod
    def _resolve_active_sensor(
        sensors: list[SensorSummary], selected_name: str | None
    ) -> SensorSummary | None:
        """Resolve the sidebar selection only when it exists in the database."""
        if not selected_name:
            return None
        return next((sensor for sensor in sensors if sensor.name == selected_name), None)

    def enforce_authentication(self) -> None:
        """Route authorization gate protection layer validating system session state."""
        enforce_session_auth()

    def _render_threshold_tabs(
        self, sensors_list: list[SensorSummary], provider: TranslationProvider
    ) -> None:
        """Render per-station threshold configuration forms with Pydantic validation.

        Args:
            sensors_list: Validated list of registered urban sensor nodes.
            provider: Active translation helper for localized UI labels.
        """
        tab_names: Final[list[str]] = [f"📍 {node.name}" for node in sensors_list]
        ui_tabs = st.tabs(tab_names)

        for index, node in enumerate(sensors_list):
            with ui_tabs[index]:
                st.write(f"Customize criteria profiles for: **{node.name}**")
                current_limits: SensorThreshold = self._repository.load_sensor_threshold(node.id)

                with st.form(key=f"form_sensor_node_limits_id_{node.id}"):
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        val_temp = st.number_input(
                            f"🌡️ {provider.get('temp')} Bound (°C)",
                            value=current_limits.temperature,
                            step=0.5,
                            key=f"input_t_{node.id}",
                        )
                    with col2:
                        val_noise = st.number_input(
                            f"🔊 {provider.get('noise')} Limit (dB)",
                            value=current_limits.noise_level,
                            step=1.0,
                            key=f"input_n_{node.id}",
                        )
                    with col3:
                        val_traffic = st.number_input(
                            f"🚗 {provider.get('traffic')} Ceiling (%)",
                            value=current_limits.traffic_load,
                            step=5.0,
                            key=f"input_tr_{node.id}",
                        )
                    with col4:
                        val_air = st.number_input(
                            f"🌫️ {provider.get('air_quality')} Max Index",
                            value=current_limits.air_quality,
                            step=5.0,
                            key=f"input_a_{node.id}",
                        )
                    with col5:
                        val_soil = st.number_input(
                            f"🌱 {provider.get('soil_moisture')} Floor (%)",
                            value=current_limits.soil_moisture,
                            step=1.0,
                            key=f"input_s_{node.id}",
                        )

                    commit_btn: Final[bool] = st.form_submit_button(
                        label=f"💾 Save Boundaries for Node {node.id}",
                        use_container_width=True,
                    )

                    if commit_btn:
                        try:
                            validated_threshold = SensorThreshold(
                                temperature=float(val_temp),
                                noise_level=float(val_noise),
                                traffic_load=float(val_traffic),
                                air_quality=float(val_air),
                                soil_moisture=float(val_soil),
                            )
                            is_saved = self._repository.save_sensor_threshold(
                                node.id, validated_threshold
                            )
                            if is_saved:
                                st.success(f"✨ Custom guidelines committed for: {node.name}")
                                st.rerun()
                            else:
                                st.error(
                                    "❌ ACID pipeline exception encountered on write operation."
                                )
                        except ValidationError as val_err:
                            st.error(f"⚠️ Invalid threshold boundary values: {val_err}")

    def _render_export_section(self, provider: TranslationProvider) -> None:
        """Render the dataset preview table and CSV/Excel download buttons.

        Args:
            provider: Active translation helper for localized UI labels.
        """
        st.markdown(f"#### 📥 {provider.get('tab_export_cfg')}")
        raw_dump: Final[pd.DataFrame] = self._repository.fetch_raw_telemetry_dump()

        if raw_dump.empty:
            st.warning("⚠️ Telemetry matrix logging channels returned empty data blocks.")
            return

        st.dataframe(raw_dump.head(10), use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            csv_data: Final[bytes] = self._export_service.generate_csv_bytes(raw_dump)
            st.download_button(
                label="📄 Download Structured CSV Dataset File",
                data=csv_data,
                file_name="smart_city_cluj_telemetry.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with exp_col2:
            xlsx_data: Final[bytes] = self._export_service.generate_excel_bytes(raw_dump)
            st.download_button(
                label="📊 Download Dynamic Excel Matrix Spreadsheet",
                data=xlsx_data,
                file_name="smart_city_cluj_matrix.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    def display(self) -> None:
        """Execute unified interface layout compilation loops with segmented tabs."""
        self.enforce_authentication()

        provider: Final[TranslationProvider] = TranslationProvider(
            default_language=st.session_state.get("lang", "EN")
        )
        active_translations_dict: Final[dict[str, str]] = provider.get_all_translations()

        selected_sensor: Final[str | None] = render_full_global_sidebar(active_translations_dict)

        st.title(f"⚙️ {provider.get('settings_title')}")
        st.caption(
            "⚡ Granular Operational Configuration Framework: "
            "Adjust tailored safety margins for individual urban micro-climates."
        )
        st.divider()

        st.markdown(f"#### 🚨 {provider.get('tab_alerts_cfg')} (Per-Station Fine Tuning)")
        sensors_list: Final[list[SensorSummary]] = self._repository.load_all_sensors()
        active_node = self._resolve_active_sensor(sensors_list, selected_sensor)

        if active_node is None:
            st.warning(
                "⚠️ No active monitoring station is available. Run "
                "`uv run python seed_db.py` from the project root, then refresh this page."
            )
        else:
            self._render_threshold_tabs(sensors_list, provider)

        st.divider()
        self._render_export_section(provider)


if __name__ == "__main__":
    page = SettingsPage()
    page.display()
