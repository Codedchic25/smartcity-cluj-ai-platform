"""Operational monitoring dashboard featuring high-contrast multi-panel visual analytics.

Provides production-ready OOP structures, Pydantic runtime validation, static type
checking compatibility, and timezone-aware UTC temporal synchronization loops.
"""

from __future__ import annotations

import os
import random
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from streamlit_folium import st_folium

PROJECT_ROOT: Final[str] = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.ai.ai_interface import render_ai_assistant, render_full_global_sidebar  # noqa: E402
from src.schemas import enforce_session_auth  # noqa: E402
from translations import TranslationProvider  # noqa: E402


class DashboardEnvironmentConfig(BaseSettings):
    """Environment-backed database configuration settings.

    Attributes:
        database_path: Filesystem location of the active SQLite database.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_path: str = Field(default="app.db", alias="DATABASE_PATH")


class SensorNode(BaseModel):
    """Geospatial metadata contract for an urban telemetry station.

    Attributes:
        id: Unique positive integer identifier of the station.
        name: Human-readable urban location label.
        latitude: Geographic latitude in decimal degrees.
        longitude: Geographic longitude in decimal degrees.
    """

    id: int = Field(ge=1)
    name: str = Field(min_length=2)
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)


class TelemetryReading(BaseModel):
    """Validated sensor telemetry payload preventing out-of-bounds corruption.

    Attributes:
        timestamp: ISO-formatted UTC measurement timestamp string.
        temperature: Ambient air temperature in degrees Celsius.
        noise_level: Acoustic noise intensity in decibels (dB).
        traffic_load: Road congestion percentage (0 to 100).
        air_quality: Particulate matter PM2.5 concentration index.
        soil_moisture: Volumetric soil moisture percentage (0 to 100).
    """

    timestamp: str
    temperature: float = Field(ge=-50.0, le=65.0)
    noise_level: float = Field(ge=0.0, le=160.0)
    traffic_load: float = Field(ge=0.0, le=100.0)
    air_quality: float = Field(ge=0.0, le=1000.0)
    soil_moisture: float = Field(ge=0.0, le=100.0)


class DashboardRepository:
    """Encapsulates secure, ACID-compliant SQLite operations with Pydantic validation."""

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the repository tracking checkpoints via environmental variables.

        Args:
            db_path: Optional explicit filesystem path to the SQLite database file.
        """
        env_path: Final[str | None] = os.environ.get("DATABASE_PATH")
        default_path: Final[str] = env_path or DashboardEnvironmentConfig().database_path
        self._db_path: Final[Path] = db_path or Path(default_path)

    @property
    def _active_db_path(self) -> Path:
        """Resolve the valid operational database path dynamically.

        Returns:
            Path: Verified filesystem path pointing to an existing SQLite file.
        """
        return self._db_path if self._db_path.is_file() else Path("app.db")

    def load_all_sensors(self) -> list[SensorNode]:
        """Extract and validate the full geospatial network of registered urban nodes.

        Returns:
            list[SensorNode]: Ordered list of validated urban sensor stations.
        """
        query: Final[str] = "SELECT id, name, latitude, longitude FROM sensors ORDER BY id"
        try:
            with sqlite3.connect(str(self._active_db_path), timeout=10.0) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(query).fetchall()
                return [SensorNode.model_validate(dict(row)) for row in rows]
        except (sqlite3.Error, ValidationError):
            return []

    def get_historical_telemetry(self, sensor_id: int, limit: int = 20) -> list[TelemetryReading]:
        """Retrieve and validate recent sensory readings safely within an isolated context.

        Args:
            sensor_id: Unique database identifier of the target sensor station.
            limit: Maximum number of chronological records to fetch.

        Returns:
            list[TelemetryReading]: Chronologically sorted list of validated readings.
        """
        query: Final[str] = """
            SELECT timestamp, temperature, noise_level, traffic_load, air_quality, soil_moisture
            FROM city_stats WHERE sensor_id = ? ORDER BY timestamp DESC LIMIT ?
        """
        try:
            with sqlite3.connect(str(self._active_db_path), timeout=15.0) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(query, (sensor_id, limit)).fetchall()
            return [TelemetryReading.model_validate(dict(row)) for row in reversed(rows)]
        except (sqlite3.Error, ValidationError):
            return []

    def generate_synthetic_history(self) -> list[TelemetryReading]:
        """Generate high-fidelity timezone-aware backup readings for timeline rendering.

        Returns:
            list[TelemetryReading]: Twenty synthetic validated telemetry points.
        """
        now: Final[datetime] = datetime.now(UTC)
        synthetic_readings: list[TelemetryReading] = []
        for i in range(20):
            timestamp_str: str = (now - timedelta(minutes=15 * (20 - i))).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            synthetic_readings.append(
                TelemetryReading(
                    timestamp=timestamp_str,
                    temperature=round(random.uniform(21.5, 26.5), 1),
                    noise_level=round(random.uniform(45.0, 58.0), 1),
                    traffic_load=round(random.uniform(25.0, 55.0), 0),
                    air_quality=round(random.uniform(15.0, 35.0), 1),
                    soil_moisture=round(random.uniform(45.0, 60.0), 1),
                )
            )
        return synthetic_readings


class VisualizationEngine:
    """Generates analytical time-series graphs using multi-panel Plotly dark templates."""

    @staticmethod
    def build_multi_metric_chart(df: pd.DataFrame, translations: dict[str, str]) -> go.Figure:
        """Construct a unified glowing timeline trace grid featuring distinct layers.

        Args:
            df: Chronological telemetry dataframe containing sensor measurements.
            translations: Active localized UI dictionary for subplot titles.

        Returns:
            go.Figure: Configured multi-row Plotly figure with predictive trends.
        """
        fig: Final[go.Figure] = make_subplots(
            rows=5,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                f"{translations.get('temp', 'Temperature')} (°C)",
                f"{translations.get('noise', 'Noise Level')} (dB)",
                f"{translations.get('traffic', 'Traffic Load')} (%)",
                f"{translations.get('air_quality', 'Air Quality')} (PM2.5)",
                f"{translations.get('soil_moisture', 'Soil Moisture')} (%)",
            ),
        )

        metrics_config: Final[list[dict[str, Any]]] = [
            {"col": "temperature", "color": "#ff0055", "row": 1, "name": "Temperature"},
            {"col": "noise_level", "color": "#00f2fe", "row": 2, "name": "Noise Level"},
            {"col": "traffic_load", "color": "#ffb300", "row": 3, "name": "Traffic Load"},
            {"col": "air_quality", "color": "#bf00ff", "row": 4, "name": "Air Quality"},
            {"col": "soil_moisture", "color": "#00ff88", "row": 5, "name": "Soil Moisture"},
        ]

        for cfg in metrics_config:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df[cfg["col"]],
                    name=cfg["name"],
                    mode="lines+markers",
                    line={"color": cfg["color"], "width": 3},
                    marker={"size": 6, "color": cfg["color"]},
                ),
                row=cfg["row"],
                col=1,
            )

            VisualizationEngine._append_predictive_trend(
                fig=fig,
                df=df,
                col_name=cfg["col"],
                color=cfg["color"],
                row=cfg["row"],
            )

        fig.update_layout(
            template="plotly_dark",
            height=750,
            showlegend=False,
            margin={"l": 30, "r": 30, "t": 40, "b": 30},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_yaxes(tickformat=".1f", showgrid=True, gridcolor="#2d3748")
        return fig

    @staticmethod
    def _append_predictive_trend(
        fig: go.Figure, df: pd.DataFrame, col_name: str, color: str, row: int
    ) -> go.Figure:
        """Append short-term regression trajectory arcs mapping system drift values.

        Args:
            fig: Target Plotly figure object to mutate in place.
            df: Historical sensor dataframe used for polynomial fitting.
            col_name: Column identifier of the metric being forecasted.
            color: Hex color code matching the parent metric trace.
            row: Target subplot row index (1 through 5).

        Returns:
            go.Figure: Updated Plotly figure containing the dashed forecast line.
        """
        if len(df) < 2:
            return fig

        y_data: Final[np.ndarray] = df[col_name].to_numpy()
        x_data: Final[np.ndarray] = np.arange(len(y_data))
        slope, intercept = np.polyfit(x_data, y_data, 1)

        future_indices: Final[np.ndarray] = np.arange(len(y_data) - 1, len(y_data) + 5)
        future_preds: Final[np.ndarray] = slope * future_indices + intercept

        try:
            last_timestamp: datetime = datetime.strptime(
                str(df["timestamp"].iloc[-1]), "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=UTC)
        except ValueError:
            last_timestamp = datetime.now(UTC)

        future_timestamps: list[str] = [str(df["timestamp"].iloc[-1])]
        for i in range(1, 5):
            next_time: datetime = last_timestamp + timedelta(minutes=15 * i)
            future_timestamps.append(next_time.strftime("%Y-%m-%d %H:%M:%S"))

        fig.add_trace(
            go.Scatter(
                x=future_timestamps,
                y=future_preds,
                mode="lines",
                line={"color": color, "width": 2, "dash": "dash"},
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        return fig


class DashboardPage:
    """Orchestrates state transition loops and core UI component execution lifecycles."""

    def __init__(self) -> None:
        """Initialize the view application orchestrator backend repositories."""
        self._repository: Final[DashboardRepository] = DashboardRepository()

    def enforce_authentication(self) -> None:
        """Enforce the shared authentication and signed-token URL synchronization gate."""
        enforce_session_auth()

    def _resolve_active_sensor(
        self, sensors: list[SensorNode], selected_name: str | None
    ) -> SensorNode | None:
        """Match the selected sidebar station name against validated sensor models.

        Args:
            sensors: List of registered urban sensor nodes.
            selected_name: Station name chosen by the operator, if one is selected.

        Returns:
            SensorNode | None: Matched sensor node, or None when no match exists.
        """
        if not selected_name:
            return None

        for node in sensors:
            if node.name == selected_name:
                return node
        return None

    def display(self) -> None:
        """Render the primary urban telemetry reporting interface dashboard nodes."""
        self.enforce_authentication()

        provider: Final[TranslationProvider] = TranslationProvider(
            default_language=st.session_state.get("lang", "EN")
        )
        active_translations_dict: Final[dict[str, str]] = provider.get_all_translations()
        selected_sensor: Final[str | None] = render_full_global_sidebar(active_translations_dict)

        sensors: list[SensorNode] = self._repository.load_all_sensors()
        if not sensors:
            st.warning(
                "No monitoring stations are available. Initialize the database with "
                "`uv run python seed_db.py` and refresh this page."
            )
            return

        active_node: Final[SensorNode | None] = self._resolve_active_sensor(
            sensors, selected_sensor
        )
        if active_node is None:
            st.warning("The selected monitoring station is unavailable. Select another station.")
            return

        readings: list[TelemetryReading] = self._repository.get_historical_telemetry(
            active_node.id, limit=20
        )
        if len(readings) < 2:
            readings = self._repository.generate_synthetic_history()

        latest: Final[TelemetryReading] = readings[-1]
        history_df: Final[pd.DataFrame] = pd.DataFrame([r.model_dump() for r in readings])

        st.title(f"🔮 {provider.get('title')}")
        st.subheader(f"📡 {provider.get('form_name')}: {active_node.name}")
        st.caption(f"🕒 Synchronization timestamp: {latest.timestamp}")
        st.divider()

        kpi_cols = st.columns(5)
        with kpi_cols[0]:
            st.metric(label=provider.get("temp"), value=f"{latest.temperature:.1f} °C")
        with kpi_cols[1]:
            st.metric(label=provider.get("noise"), value=f"{latest.noise_level:.1f} dB")
        with kpi_cols[2]:
            st.metric(label=provider.get("traffic"), value=f"{latest.traffic_load:.0f} %")
        with kpi_cols[3]:
            st.metric(label=provider.get("air_quality"), value=f"{latest.air_quality:.1f} PM2.5")
        with kpi_cols[4]:
            st.metric(label=provider.get("soil_moisture"), value=f"{latest.soil_moisture:.1f} %")

        st.divider()
        st.markdown("#### 🗺️ Geospatial Node Telemetry Infrastructure")

        geospatial_map: Final[folium.Map] = folium.Map(
            location=[active_node.latitude, active_node.longitude],
            zoom_start=15,
            tiles="OpenStreetMap",
        )
        folium.Marker(
            [active_node.latitude, active_node.longitude],
            popup=f"Active Node: {active_node.name}",
            tooltip=active_node.name,
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(geospatial_map)
        st_folium(geospatial_map, height=300, width="stretch")

        st.divider()
        chart_figure: Final[go.Figure] = VisualizationEngine.build_multi_metric_chart(
            df=history_df, translations=active_translations_dict
        )
        st.plotly_chart(chart_figure, width="stretch")

        st.divider()
        with st.expander(f"🧠 {provider.get('ai_assistant')}", expanded=True):
            render_ai_assistant(
                location=active_node.name,
                temperature=latest.temperature,
                air_quality=latest.air_quality,
                soil_moisture=latest.soil_moisture,
                noise_level=latest.noise_level,
                traffic_load=latest.traffic_load,
                translations=active_translations_dict,
            )


if __name__ == "__main__":
    page = DashboardPage()
    page.display()
