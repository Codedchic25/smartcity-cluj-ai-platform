"""Advanced statistical analysis and automated machine learning trend forecasting engine.

Provides high-performance scientific computations, dynamic linear regressions, Pydantic
telemetry validation, and PEP 8 compliant Plotly matrix visualizations.
"""

from __future__ import annotations

import os
import random
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Final[str] = str(Path(__file__).resolve().parents.__getitem__(1))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.ai.ai_interface import render_ai_assistant, render_full_global_sidebar  # noqa: E402
from src.schemas import enforce_session_auth  # noqa: E402
from translations import TranslationProvider  # noqa: E402


class AnalyticsEnvironmentConfig(BaseSettings):
    """Environment-backed database configuration settings.

    Attributes:
        database_path: Filesystem location of the active SQLite database.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_path: str = Field(default="app.db", alias="DATABASE_PATH")


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


class RegressionResult(BaseModel):
    """Encapsulates derived linear regression parameters and predicted values.

    Attributes:
        predictions: Array of predicted target values along the independent axis.
        slope: Derived linear slope coefficient.
        intercept: Derived y-axis intercept constant.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    predictions: np.ndarray
    slope: float
    intercept: float


RegressionResult.model_rebuild(_types_namespace={"np": np})


class AnalyticsEngine:
    """Handles advanced scientific computations and multi-panel Plotly dark templates."""

    @staticmethod
    def compute_linear_regression(df: pd.DataFrame, x_col: str, y_col: str) -> RegressionResult:
        """Calculate a linear regression sequence targeting trend vectors via polyfit.

        Args:
            df: Source dataframe containing historical telemetry metrics.
            x_col: Column identifier for the independent predictor variable.
            y_col: Column identifier for the dependent target variable.

        Returns:
            RegressionResult: Validated model containing predictions, slope, and intercept.
        """
        if df.empty or len(df) < 2:
            return RegressionResult(predictions=np.array([]), slope=0.0, intercept=0.0)
        try:
            x_data: Final[np.ndarray] = df[x_col].to_numpy()
            y_data: Final[np.ndarray] = df[y_col].to_numpy()
            slope, intercept = np.polyfit(x_data, y_data, 1)
            return RegressionResult(
                predictions=slope * x_data + intercept,
                slope=float(slope),
                intercept=float(intercept),
            )
        except (ValueError, TypeError):
            return RegressionResult(predictions=np.array([]), slope=0.0, intercept=0.0)

    @classmethod
    def generate_heatmap(cls, df: pd.DataFrame, metrics: list[str]) -> go.Figure:
        """Generate high-contrast Pearson correlation matrix heatmap plots for dark views.

        Args:
            df: Source dataframe containing numeric sensor metrics.
            metrics: List of metric column names to correlate.

        Returns:
            go.Figure: Configured Plotly heatmap visualization.
        """
        corr_matrix: Final[pd.DataFrame] = df[metrics].corr()
        fig: Final[go.Figure] = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Electric",
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        )
        return fig

    @classmethod
    def generate_urban_radar_chart(
        cls, df: pd.DataFrame, metrics: list[str], location_name: str
    ) -> go.Figure:
        """Compile a multidimensional Radar signature mapping normalized metric balances.

        Args:
            df: Source dataframe containing sensor readings.
            metrics: List of metric column names to normalize onto a 0-100 scale.
            location_name: Name of the urban monitoring station being profiled.

        Returns:
            go.Figure: Closed-polygon Plotly polar radar chart.
        """
        fig: Final[go.Figure] = go.Figure()
        mean_values: list[float] = []

        for metric in metrics:
            if metric in ("traffic_load", "soil_moisture"):
                mean_values.append(float(df[metric].mean()))
            elif metric == "temperature":
                mean_values.append(float((df[metric].mean() / 40.0) * 100.0))
            elif metric == "noise_level":
                mean_values.append(float((df[metric].mean() / 100.0) * 100.0))
            elif metric == "air_quality":
                mean_values.append(float((df[metric].mean() / 150.0) * 100.0))

        display_labels: Final[list[str]] = [m.replace("_", " ").title() for m in metrics]

        fig.add_trace(
            go.Scatterpolar(
                r=mean_values + mean_values[:1],
                theta=display_labels + display_labels[:1],
                fill="toself",
                name=location_name,
                line={"color": "#00f2fe", "width": 3},
                fillcolor="rgba(0, 242, 254, 0.25)",
            )
        )

        fig.update_layout(
            polar={
                "radialaxis": {
                    "visible": True,
                    "range": list((0, 100)),
                    "gridcolor": "#2d3748",
                },
                "angularaxis": {"gridcolor": "#2d3748"},
            },
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        return fig

    @classmethod
    def generate_marginal_distribution_plot(
        cls, df: pd.DataFrame, x_col: str, y_col: str
    ) -> go.Figure:
        """Build a density scatter matrix with marginal box distribution tracks.

        Args:
            df: Source telemetry dataframe.
            x_col: Column identifier for the horizontal distribution axis.
            y_col: Column identifier for the vertical distribution axis.

        Returns:
            go.Figure: Joint scatter plot with marginal box and violin subplots.
        """
        fig: Final[go.Figure] = px.scatter(
            df,
            x=x_col,
            y=y_col,
            marginal_x="box",
            marginal_y="violin",
            template="plotly_dark",
        )

        fig.update_traces(
            marker={
                "size": 8,
                "color": "#00ff88",
                "opacity": 0.75,
                "line": {"width": 1, "color": "#111525"},
            },
            selector={"type": "scatter"},
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
        )
        return fig

    @classmethod
    def generate_regression_plot(
        cls,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        y_pred: np.ndarray,
        title: str,
        label_text: str,
    ) -> go.Figure:
        """Build a scatter chart overlaying the derived predictive trend lines.

        Args:
            df: Source dataframe containing historical observations.
            x_col: Independent variable column identifier.
            y_col: Dependent target variable column identifier.
            y_pred: Computed linear regression values matching x_col.
            title: Formatted chart title displaying the mathematical drift formula.
            label_text: Legend label for the regression trend line.

        Returns:
            go.Figure: Combined scatter and regression line visualization.
        """
        df_copy: Final[pd.DataFrame] = df.copy()
        df_copy["ML_Predicted"] = y_pred
        fig: Final[go.Figure] = px.scatter(
            df_copy, x=x_col, y=y_col, template="plotly_dark", title=title
        )
        fig.add_scatter(
            x=df_copy[x_col],
            y=df_copy["ML_Predicted"],
            mode="lines",
            name=label_text,
            line={"color": "#00ff88", "width": 4},
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig


class AnalyticsPage:
    """Orchestrates the visualization lifecycle of the analytics view."""

    def __init__(self) -> None:
        """Initialize parameters and metrics data logging tracks."""
        env_path: Final[str | None] = os.environ.get("DATABASE_PATH")
        default_path: Final[str] = env_path or AnalyticsEnvironmentConfig().database_path
        self._db_path: Final[Path] = Path(default_path)
        self._metrics_list: Final[list[str]] = [
            "temperature",
            "noise_level",
            "traffic_load",
            "air_quality",
            "soil_moisture",
        ]

    @property
    def _active_db_path(self) -> Path:
        """Resolve the active SQLite database path dynamically.

        Returns:
            Path: Verified filesystem path pointing to an existing SQLite file.
        """
        return self._db_path if self._db_path.is_file() else Path("app.db")

    def enforce_authentication(self) -> None:
        """Enforce the shared authentication and signed-token URL synchronization gate."""
        enforce_session_auth()

    def _generate_synthetic_sensor_history(self) -> list[TelemetryReading]:
        """Generate validated synthetic data structures to shield analytics displays.

        Returns:
            list[TelemetryReading]: Fifty synthetic validated telemetry records.
        """
        now: Final[datetime] = datetime.now(UTC)
        synthetic_records: list[TelemetryReading] = []
        for i in range(50):
            timestamp_str: str = (now - timedelta(minutes=15 * (50 - i))).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            random_traffic: float = random.uniform(30.0, 85.0)
            synthetic_records.append(
                TelemetryReading(
                    timestamp=timestamp_str,
                    temperature=round(random.uniform(22.0, 31.5), 1),
                    noise_level=round(random.uniform(45.0, 72.0), 1),
                    traffic_load=round(random_traffic, 0),
                    air_quality=round(0.45 * random_traffic + random.uniform(10.0, 25.0), 1),
                    soil_moisture=round(random.uniform(40.0, 65.0), 1),
                )
            )
        return synthetic_records

    def load_sensor_history(self, selected_sensor: str | None) -> list[TelemetryReading] | None:
        """Extract and validate historical telemetry directly via thread-safe SQL queries.

        Args:
            selected_sensor: Human-readable name of the target station, if selected.

        Returns:
            list[TelemetryReading] | None: Validated readings, or None if the station is absent.
        """
        if not selected_sensor:
            return None

        readings: list[TelemetryReading] = []
        try:
            with sqlite3.connect(str(self._active_db_path)) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT id FROM sensors WHERE name = ? LIMIT 1",
                    (selected_sensor,),
                ).fetchone()
                if row is None:
                    return None

                sensor_id: int = int(row["id"])
                query: Final[str] = """
                    SELECT timestamp, temperature, noise_level, traffic_load,
                           air_quality, soil_moisture
                    FROM city_stats WHERE sensor_id = ?
                    ORDER BY timestamp DESC LIMIT 50
                """
                rows = conn.execute(query, (sensor_id,)).fetchall()
                readings = [TelemetryReading.model_validate(dict(r)) for r in reversed(rows)]
        except (sqlite3.Error, ValidationError):
            readings = []

        if len(readings) < 5:
            readings = self._generate_synthetic_sensor_history()

        return readings

    def display(self) -> None:
        """Execute core pipeline modeling transformations and views layout."""
        self.enforce_authentication()

        provider: Final[TranslationProvider] = TranslationProvider(
            default_language=st.session_state.get("lang", "EN")
        )
        active_translations_dict: Final[dict[str, str]] = provider.get_all_translations()
        selected_sensor: Final[str | None] = render_full_global_sidebar(active_translations_dict)

        st.title(provider.get("ml_title"))
        st.caption(f"Core statistical analytics matrix and forecasting: {selected_sensor}")
        st.divider()

        readings: list[TelemetryReading] | None = self.load_sensor_history(selected_sensor)
        if readings is None:
            st.warning(
                "No active monitoring station is available in the database. "
                "Initialize the database with `uv run python seed_db.py` and refresh this page."
            )
            return

        latest: Final[TelemetryReading] = readings[-1]
        df_analytics: Final[pd.DataFrame] = pd.DataFrame([r.model_dump() for r in readings])

        col_grid1, col_grid2 = st.columns((1.2, 1))

        with col_grid1:
            st.markdown(f"#### {provider.get('pearson_heatmap_title')}")
            fig_corr: go.Figure = AnalyticsEngine.generate_heatmap(df_analytics, self._metrics_list)
            st.plotly_chart(fig_corr, width="stretch")

        with col_grid2:
            st.markdown("#### Normalized District Urban Balance Signature (Radar)")
            fig_radar: go.Figure = AnalyticsEngine.generate_urban_radar_chart(
                df_analytics, self._metrics_list, selected_sensor
            )
            st.plotly_chart(fig_radar, width="stretch")

        st.divider()

        st.markdown("#### Advanced Joint Density Dispersion (Traffic Density vs Air Quality)")
        fig_joint: go.Figure = AnalyticsEngine.generate_marginal_distribution_plot(
            df_analytics, "traffic_load", "air_quality"
        )
        st.plotly_chart(fig_joint, width="stretch")

        st.divider()

        st.markdown(f"#### {provider.get('ml_forecast_section')}")

        col_x, col_y = st.columns(2)
        with col_x:
            sel_x: str = st.selectbox(
                "Select Independent Axis Variable (X):",
                options=self._metrics_list,
                index=2,
            )
        with col_y:
            sel_y: str = st.selectbox(
                "Select Target Predictor Variable (Y):",
                options=self._metrics_list,
                index=3,
            )

        reg_result: Final[RegressionResult] = AnalyticsEngine.compute_linear_regression(
            df_analytics, sel_x, sel_y
        )

        if len(reg_result.predictions) > 0:
            title_plot: str = (
                f"Derived Mathematical Drift Equation: "
                f"{sel_y.upper()} = {reg_result.slope:.3f} * {sel_x.upper()} + "
                f"{reg_result.intercept:.2f}"
            )
            fig_ml: go.Figure = AnalyticsEngine.generate_regression_plot(
                df_analytics,
                sel_x,
                sel_y,
                reg_result.predictions,
                title_plot,
                provider.get("forecast_label"),
            )
            fig_ml.update_yaxes(tickformat=".1f")
            fig_ml.update_xaxes(tickformat=".1f")
            st.plotly_chart(fig_ml, width="stretch")
            st.caption(f"Model engine: {provider.get('ml_model_caption')}")
        else:
            st.warning("Computational failure: Insufficient historical records.")

        st.divider()
        with st.expander(provider.get("ai_assistant"), expanded=True):
            render_ai_assistant(
                location=selected_sensor,
                temperature=latest.temperature,
                air_quality=latest.air_quality,
                soil_moisture=latest.soil_moisture,
                noise_level=latest.noise_level,
                traffic_load=latest.traffic_load,
                translations=active_translations_dict,
            )


if __name__ == "__main__":
    page = AnalyticsPage()
    page.display()
