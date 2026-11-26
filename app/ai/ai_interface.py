"""Smart City IoT Platform - Interface Core Components.

Enforces absolute path resilience, precise nominal attribute mapping,
and production-grade code quality alignment with strict linter rule governance.
"""

from __future__ import annotations

import math
import os
import re
import sys
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, Protocol

import pandas as pd
import streamlit as st

# Path resilience injection framework for distributed virtual testing environments
PROJECT_ROOT: Final[str] = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.ai.rag_pipeline import (  # noqa: E402
    TelemetryEvidence,
    extract_context_numbers,
    format_retrieved_context,
    retrieve_telemetry,
    sanitize_untrusted_input,
)
from src.schemas import enforce_session_auth  # noqa: E402
from src.session_auth import clear_authenticated_session  # noqa: E402


@dataclass(frozen=True, slots=True)
class SensorThresholdMatrix:
    """Immutable data contract encapsulating operational limits for an IoT zone."""

    temperature_limit: float
    noise_limit: float
    traffic_limit: float
    air_quality_limit: float
    soil_moisture_limit: float


class ContextReaderInterface(ABC):
    """Abstract contract enforcing data isolation for infrastructure telemetry."""

    @abstractmethod
    def get_latest_alerts_context(self, limit_lines: int = 5) -> str:
        """Extract recent historical security alerts without blocking file handles."""

    @abstractmethod
    def fetch_sensor_names(self) -> list[str]:
        """Query data nodes to resolve active telemetry station names."""

    @abstractmethod
    def fetch_operational_thresholds(self, location_name: str) -> SensorThresholdMatrix:
        """Map nominal attributes explicitly to resolve sensor warning barriers."""

    @abstractmethod
    def fetch_recent_telemetry_trend(self, location_name: str, limit: int = 5) -> str:
        """Summarize recent selected-station readings without inferring from one record."""


class AiProviderInterface(Protocol):
    """Structural protocol defining LLM inference engines ensuring loose coupling."""

    def generate_completion(self, system_prompt_context: str) -> str:
        """Dispatches an inference request payload toward the cloud gateway grid."""
        ...


@st.cache_data(ttl=300, show_spinner=False)
def _cached_ai_completion(
    rendered_prompt: str,
    location: str,
    temperature: float,
    air_quality: float,
    soil_moisture: float,
    noise_level: float,
    traffic_load: float,
    language: str,
    provider_cache_key: str,
    _provider: AiProviderInterface,
) -> str:
    """Cache generated recommendations for matching telemetry and language inputs."""
    return _provider.generate_completion(rendered_prompt)


class TelemetryContextReader(ContextReaderInterface):
    """Handles low-level data extraction from local security journals and databases."""

    def __init__(
        self,
        log_path: str | Path = "security_alerts.log",
        database_path: str | Path = "app.db",
    ) -> None:
        """Initialize core engine components with verified fallback properties."""
        import sqlite3

        self._sqlite3 = sqlite3
        self._log_path: Final[Path] = Path(log_path)
        self._database_path: Final[Path] = Path(os.environ.get("DATABASE_PATH", database_path))

        # Clean 7-station fallback list perfectly aligned with PEP 8 and Ruff E501
        self._fallback_sensors: Final[list[str]] = [
            "Mărăști - Sens Giratoriu",
            "Mănăștur - Str. Primăverii",
            "Piața Unirii - Centru Urban",
            "Gheorgheni - Iulius Area",
            "Zorilor - Observatorului",
            "Grădini Mănăștur - Calea Florești",
            "Iris - Zona Industrială",
        ]
        self._default_thresholds: Final[SensorThresholdMatrix] = SensorThresholdMatrix(
            temperature_limit=32.0,
            noise_limit=75.0,
            traffic_limit=80.0,
            air_quality_limit=80.0,
            soil_moisture_limit=35.0,
        )

    def get_latest_alerts_context(self, limit_lines: int = 5) -> str:
        """Extract recent historical security alerts without blocking file handles."""
        if os.path.exists(self._log_path):
            try:
                with open(self._log_path, encoding="utf-8") as file:
                    lines = file.readlines()[-limit_lines:]
                    return "".join(lines)
            except OSError:
                pass
        return "No active security breach events logged."

    def fetch_sensor_names(self) -> list[str]:
        """Query data nodes to resolve active telemetry station names."""
        if self._database_path.is_file():
            try:
                with self._sqlite3.connect(str(self._database_path), timeout=5) as connection:
                    df = pd.read_sql_query("SELECT name FROM sensors ORDER BY id", connection)
                    if not df.empty:
                        return df["name"].tolist()
            except (self._sqlite3.Error, pd.errors.DatabaseError):
                pass
        return self._fallback_sensors

    def fetch_operational_thresholds(self, location_name: str) -> SensorThresholdMatrix:
        """Map nominal attributes explicitly to resolve sensor warning barriers."""
        if not self._database_path.is_file():
            return self._default_thresholds

        query: Final[str] = """
            SELECT temp_limit, noise_limit, traffic_limit, air_limit, soil_limit
            FROM settings
            JOIN sensors ON sensors.id = settings.sensor_id
            WHERE name = ?;
        """
        try:
            with self._sqlite3.connect(str(self._database_path), timeout=5) as conn:
                conn.row_factory = self._sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, (location_name,))
                row = cursor.fetchone()

                if row is not None:
                    return SensorThresholdMatrix(
                        temperature_limit=float(row["temp_limit"]),
                        noise_limit=float(row["noise_limit"]),
                        traffic_limit=float(row["traffic_limit"]),
                        air_quality_limit=float(row["air_limit"]),
                        soil_moisture_limit=float(row["soil_limit"]),
                    )
        except self._sqlite3.Error:
            pass

        return self._default_thresholds

    def fetch_recent_telemetry_trend(self, location_name: str, limit: int = 5) -> str:
        """Summarize recent readings for one station, or state why no trend is available."""
        heading = "[RECENT TELEMETRY TREND]"
        if not self._database_path.is_file():
            return (
                f"{heading}\nNo recent station history is available; "
                "no trend can be established."
            )

        query: Final[str] = """
            SELECT city_stats.temperature, city_stats.noise_level,
                   city_stats.traffic_load, city_stats.air_quality,
                   city_stats.soil_moisture
            FROM city_stats
            JOIN sensors ON sensors.id = city_stats.sensor_id
            WHERE sensors.name = ?
            ORDER BY city_stats.timestamp DESC, city_stats.id DESC
            LIMIT ?;
        """
        try:
            with self._sqlite3.connect(str(self._database_path), timeout=5) as connection:
                rows = connection.execute(
                    query,
                    (location_name, max(2, min(int(limit), 20))),
                ).fetchall()
        except (self._sqlite3.Error, TypeError, ValueError):
            return (
                f"{heading}\nRecent station history is unavailable; "
                "no trend can be established."
            )

        readings: list[tuple[float, ...]] = []
        for row in reversed(rows):
            try:
                values = tuple(float(value) for value in row)
            except (TypeError, ValueError):
                continue
            if len(values) == 5 and all(math.isfinite(value) for value in values):
                readings.append(values)

        if len(readings) < 2:
            return (
                f"{heading}\nFewer than two valid station readings are available; "
                "no trend can be established."
            )

        metrics = (
            ("Temperature", 0, "°C"),
            ("Noise level", 1, "dB"),
            ("Traffic load", 2, "%"),
            ("Air quality PM2.5", 3, "µg/m³"),
            ("Soil moisture", 4, "%"),
        )
        trend_lines = [
            f"Historical comparison across {len(readings)} recent readings "
            "(oldest to newest):"
        ]
        for label, index, unit in metrics:
            start_value = readings[0][index]
            end_value = readings[-1][index]
            delta = round(end_value - start_value, 1)
            direction = "stable" if delta == 0 else "rising" if delta > 0 else "falling"
            trend_lines.append(
                f"- {label}: {direction} "
                f"({start_value:g} to {end_value:g} {unit}; delta {delta:+.1f} {unit})"
            )
        return f"{heading}\n" + "\n".join(trend_lines)


class UrbanAiAssistantComponent:
    """Renders the intelligent AI advisory dashboard card with absolute state persistence."""

    def __init__(
        self,
        context_reader: ContextReaderInterface | None = None,
        ai_provider: AiProviderInterface | None = None,
    ) -> None:
        """Initialize the AI component anchoring deep template path variables."""
        from app.ai.groq_provider import GroqProvider

        self._context_reader: Final[ContextReaderInterface] = (
            context_reader or TelemetryContextReader()
        )
        self._ai_provider: Final[AiProviderInterface] = ai_provider or GroqProvider()
        if isinstance(self._ai_provider, GroqProvider):
            self._ai_cache_key: Final[str] = (
                f"{type(self._ai_provider).__module__}."
                f"{type(self._ai_provider).__qualname__}:"
                f"{self._ai_provider.configured_model}"
            )
        else:
            self._ai_cache_key = (
                f"{type(self._ai_provider).__module__}."
                f"{type(self._ai_provider).__qualname__}:{id(self._ai_provider)}"
            )
        self._prompt_template_path: Final[str] = "ai_tests/prompts.txt"
        self._think_block_pattern: Final[re.Pattern] = re.compile(
            r"<think>.*?</think>", re.DOTALL | re.IGNORECASE
        )
        self._think_tag_pattern: Final[re.Pattern] = re.compile(r"</?think>", re.IGNORECASE)

    def _load_prompt_template(self) -> str | None:
        """Safely read execution limits and guardrails from files."""
        try:
            with open(self._prompt_template_path, encoding="utf-8") as file:
                return file.read()
        except OSError:
            st.error(
                f"❌ Production template file '{self._prompt_template_path}' "
                f"was not found or is inaccessible on disk."
            )
            return None

    def _clean_reasoning_artifacts(self, text: str) -> str:
        """Remove advanced analytical raw trace fragments."""
        cleaned = self._think_block_pattern.sub("", text)
        cleaned = self._think_tag_pattern.sub("", cleaned)
        return cleaned.strip()

    @staticmethod
    def _normalize_grounding_text(text: str) -> str:
        decomposed = unicodedata.normalize("NFKD", text.casefold())
        return "".join(char for char in decomposed if not unicodedata.combining(char))

    @classmethod
    def _response_mentions_metric_value(
        cls,
        response: str,
        metric_aliases: tuple[str, ...],
        value: float,
    ) -> bool:
        normalized_response = cls._normalize_grounding_text(response)
        expected_value = Decimal(str(value)).normalize()
        for alias in metric_aliases:
            normalized_alias = cls._normalize_grounding_text(alias)
            for match in re.finditer(
                rf"(?<!\w){re.escape(normalized_alias)}(?!\w)",
                normalized_response,
            ):
                nearby_text = normalized_response[
                    max(0, match.start() - 32) : min(
                        len(normalized_response), match.end() + 48
                    )
                ]
                if expected_value in extract_context_numbers(nearby_text):
                    return True
        return False

    @classmethod
    def _response_mentions_threshold_breach(
        cls,
        response: str,
        metric_aliases: tuple[str, ...],
        measured_value: float,
        threshold: float,
        direction: str,
    ) -> bool:
        normalized_response = cls._normalize_grounding_text(response)
        expected_measured = Decimal(str(measured_value)).normalize()
        expected_threshold = Decimal(str(threshold)).normalize()
        direction_terms = (
            ("above", "exceeds", "greater than", "over", "depases", "peste", ">")
            if direction == "above"
            else ("below", "under", "less than", "sub", "mai putin", "<")
        )
        for alias in metric_aliases:
            normalized_alias = cls._normalize_grounding_text(alias)
            for match in re.finditer(
                rf"(?<!\w){re.escape(normalized_alias)}(?!\w)",
                normalized_response,
            ):
                nearby_text = normalized_response[
                    max(0, match.start() - 40) : min(
                        len(normalized_response), match.end() + 104
                    )
                ]
                nearby_numbers = extract_context_numbers(nearby_text)
                if (
                    expected_measured in nearby_numbers
                    and expected_threshold in nearby_numbers
                    and any(term in nearby_text for term in direction_terms)
                ):
                    return True
        return False

    @classmethod
    def _ensure_response_grounding(
        cls,
        response: str,
        location: str,
        readings: tuple[tuple[str, tuple[str, ...], float, str, float, str], ...],
    ) -> str:
        """Append only missing station, measurement, and configured-breach evidence."""
        if not response or response.startswith(("❌", "Generic")):
            return response

        normalized_response = cls._normalize_grounding_text(response)
        facts: list[str] = []
        normalized_location = cls._normalize_grounding_text(location)
        if normalized_location not in normalized_response:
            facts.append(f"Station: {location}.")

        missing_readings = [
            f"{label}: {value:g} {unit}"
            for label, aliases, value, unit, _, _ in readings
            if not cls._response_mentions_metric_value(response, aliases, value)
        ]
        if missing_readings:
            facts.append("Verified current readings: " + "; ".join(missing_readings) + ".")

        missing_breaches = []
        for label, aliases, value, unit, threshold, direction in readings:
            breached = value > threshold if direction == "above" else value < threshold
            if breached and not cls._response_mentions_threshold_breach(
                response, aliases, value, threshold, direction
            ):
                relation = "exceeds" if direction == "above" else "is below"
                missing_breaches.append(
                    f"{label}: {value:g} {unit} {relation} configured threshold "
                    f"{threshold:g} {unit}"
                )
        if missing_breaches:
            facts.append(
                "Verified configured threshold breaches: "
                + "; ".join(missing_breaches)
                + "."
            )

        if not facts:
            return response
        return response.rstrip() + "\n\n**Verified Operational Facts:**\n" + "\n".join(facts)

    def render(
        self,
        location: str,
        temperature: float,
        air_quality: float,
        soil_moisture: float,
        translations: dict[str, str],
        noise_level: float = 0.0,
        traffic_load: float = 0.0,
    ) -> None:
        """Render the cloud advisory component with defensive data verification blocks."""
        prompt_template = self._load_prompt_template()
        if not prompt_template:
            return

        safe_location = sanitize_untrusted_input(location, max_length=200)
        session_key: Final[str] = f"persistent_ai_response_{safe_location.replace(' ', '_')}"
        if session_key not in st.session_state:
            st.session_state[session_key] = None

        active_lang: Final[str] = st.session_state.get("lang", "RO")
        historical_alerts_log: Final[str] = self._context_reader.get_latest_alerts_context(
            limit_lines=5
        )
        safe_historical_alerts = sanitize_untrusted_input(historical_alerts_log)

        thresholds = self._context_reader.fetch_operational_thresholds(safe_location)

        evidence = [
            TelemetryEvidence(
                location=safe_location,
                sensor="temperature",
                value=temperature,
                unit="°C",
            ),
            TelemetryEvidence(
                location=safe_location,
                sensor="air_quality",
                value=air_quality,
                unit="µg/m³",
            ),
            TelemetryEvidence(
                location=safe_location,
                sensor="noise",
                value=noise_level,
                unit="dB",
            ),
            TelemetryEvidence(
                location=safe_location,
                sensor="traffic",
                value=traffic_load,
                unit="%",
            ),
            TelemetryEvidence(
                location=safe_location,
                sensor="soil_moisture",
                value=soil_moisture,
                unit="%",
            ),
            TelemetryEvidence(
                location=safe_location,
                sensor="temperature",
                value=thresholds.temperature_limit,
                unit="threshold °C",
            ),
            TelemetryEvidence(
                location=safe_location,
                sensor="noise",
                value=thresholds.noise_limit,
                unit="threshold dB",
            ),
            TelemetryEvidence(
                location=safe_location,
                sensor="traffic",
                value=thresholds.traffic_limit,
                unit="threshold %",
            ),
            TelemetryEvidence(
                location=safe_location,
                sensor="air_quality",
                value=thresholds.air_quality_limit,
                unit="threshold µg/m³",
            ),
            TelemetryEvidence(
                location=safe_location,
                sensor="soil_moisture",
                value=thresholds.soil_moisture_limit,
                unit="threshold %",
            ),
        ]
        retrieved_records = retrieve_telemetry(
            evidence,
            "temperature noise traffic air quality soil moisture",
            safe_location,
        )
        recent_trend = self._context_reader.fetch_recent_telemetry_trend(safe_location)
        retrieved_context = (
            f"{format_retrieved_context(retrieved_records)}\n\n"
            f"{sanitize_untrusted_input(recent_trend)}"
        )

        telemetry_context_extension: Final[str] = (
            f"\n\n[AUTHORIZED OPERATIONAL THRESHOLDS]\n"
            f"Reference thresholds for {location}:\n"
            f"- Temperature Threshold: {thresholds.temperature_limit:.1f} °C\n"
            f"- Noise Level Threshold: {thresholds.noise_limit:.1f} dB\n"
            f"- Traffic Load Threshold: {thresholds.traffic_limit:.1f} %\n"
            f"- Air Quality PM2.5 Threshold: "
            f"{thresholds.air_quality_limit:.1f} µg/m³\n"
            f"- Soil Moisture Minimum Threshold: "
            f"{thresholds.soil_moisture_limit:.1f} %"
        )

        rendered_prompt = (
            prompt_template.replace("{{locatie}}", safe_location)
            .replace("{{temperature}}", f"{temperature:.1f}")
            .replace("{{air_quality}}", f"{air_quality:.1f}")
            .replace("{{soil_moisture}}", f"{soil_moisture:.1f}")
            .replace("{{noise_level}}", f"{noise_level:.1f}")
            .replace("{{traffic_load}}", f"{traffic_load:.1f}")
            .replace("{{limba_activa}}", active_lang)
            .replace("{{jurnal_alerte_recente}}", safe_historical_alerts)
        )
        rendered_prompt = (
            f"{rendered_prompt.rstrip()}\n\n{telemetry_context_extension.strip()}"
            f"\n\n[RETRIEVED CONTEXT]\n{retrieved_context}\n[/RETRIEVED CONTEXT]"
        )

        if st.button(
            translations.get("generate_rec", "Generate Recommendations"),
            type="secondary",
            key=f"ai_generate_btn_{safe_location.replace(' ', '_')}",
            width="stretch",
        ):
            with st.spinner(translations.get("loading_ai", "Analyzing sensor telemetry...")):
                try:
                    response_text = _cached_ai_completion(
                        rendered_prompt,
                        safe_location,
                        temperature,
                        air_quality,
                        soil_moisture,
                        noise_level,
                        traffic_load,
                        active_lang,
                        self._ai_cache_key,
                        self._ai_provider,
                    )
                    if response_text:
                        cleaned_response = self._clean_reasoning_artifacts(response_text)
                        station_readings = (
                            (
                                "Temperature",
                                ("temperature", "temperatura"),
                                temperature,
                                "°C",
                                thresholds.temperature_limit,
                                "above",
                            ),
                            (
                                "Air quality PM2.5",
                                ("air quality", "calitatea aerului", "pm2.5", "pm25"),
                                air_quality,
                                "µg/m³",
                                thresholds.air_quality_limit,
                                "above",
                            ),
                            (
                                "Noise level",
                                ("noise", "noise level", "zgomot", "poluare fonica"),
                                noise_level,
                                "dB",
                                thresholds.noise_limit,
                                "above",
                            ),
                            (
                                "Traffic load",
                                ("traffic", "traffic load", "trafic", "aglomeratie"),
                                traffic_load,
                                "%",
                                thresholds.traffic_limit,
                                "above",
                            ),
                            (
                                "Soil moisture",
                                ("soil moisture", "moisture", "umiditate", "sol"),
                                soil_moisture,
                                "%",
                                thresholds.soil_moisture_limit,
                                "below",
                            ),
                        )
                        st.session_state[session_key] = self._ensure_response_grounding(
                            cleaned_response,
                            safe_location,
                            station_readings,
                        )
                    else:
                        st.warning("⚠️ Remote cloud gateway engine returned null.")
                except Exception as exc:
                    st.error(f"❌ Critical connection error: {exc}")

        if st.session_state[session_key]:
            st.markdown("<br>", unsafe_allow_html=True)
            response_payload: str = str(st.session_state[session_key])

            if response_payload.startswith("❌") or response_payload.startswith("Generic"):
                st.error(response_payload)
            else:
                with st.chat_message("assistant"):
                    st.markdown(response_payload)


class GlobalSidebarComponent:
    """Orchestrates unified control layers within the sidebar space."""

    def __init__(self, context_reader: ContextReaderInterface | None = None) -> None:
        """Initialize the layout framework component."""
        self._context_reader: Final[ContextReaderInterface] = (
            context_reader or TelemetryContextReader()
        )
        self._supported_languages: Final[list[str]] = [
            "RO",
            "EN",
            "IT",
            "ES",
            "HU",
        ]

    def _render_operator_card(self, label: str) -> None:
        """Inject an isolated HTML block displaying the authenticated operator profile."""
        user_full_name: Final[str] = os.environ.get("OPERATOR_FULL_NAME", "Cojocaru Maria Gabriela")

        card_style: Final[str] = (
            "background-color: #111525; padding: 15px; border-radius: 8px; "
            "border: 1px solid #000080; margin-bottom: 10px; width: 100%;"
        )
        label_style: Final[str] = (
            "color: #888; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;"
        )
        name_style: Final[str] = (
            "color: white; font-size: 1.05rem; font-weight: bold; "
            "margin-top: 4px; line-height: 1.2;"
        )
        badge_style: Final[str] = (
            "display: inline-block; background-color: #000080; color: white; "
            "font-size: 0.7rem; font-weight: bold; padding: 3px 8px; "
            "border-radius: 4px; margin-top: 8px; letter-spacing: 0.5px;"
        )

        html_payload: Final[str] = (
            f'<div style="{card_style}">'
            f'<span style="{label_style}">👤 {label}</span>'
            f'<div style="{name_style}">{user_full_name}</div>'
            f'<div style="{badge_style}">🚀 Lead Cloud Engineer</div>'
            f"</div>"
        )
        st.markdown(html_payload, unsafe_allow_html=True)

    def render_full_sidebar(self, translations: dict[str, str]) -> str:
        """Compile and project localization controls onto the sidebar view canvas."""
        enforce_session_auth()

        sensor_names: Final[list[str]] = self._context_reader.fetch_sensor_names()

        if "global_sensor_selectbox_widget" not in st.session_state:
            current_index = 0
        else:
            try:
                current_index = sensor_names.index(
                    st.session_state["global_sensor_selectbox_widget"]
                )
            except ValueError:
                current_index = 0

        with st.sidebar:

            def global_lang_callback() -> None:
                st.session_state["lang"] = st.session_state["global_language_widget"]

            st.selectbox(
                translations.get("language_select", "🌐 Change language / Schimbă limba"),
                options=self._supported_languages,
                index=self._supported_languages.index(st.session_state.get("lang", "RO")),
                key="global_language_widget",
                on_change=global_lang_callback,
            )
            st.sidebar.divider()

            self._render_operator_card(translations.get("operator_label", "System Operator"))

            if st.session_state.get("authenticated", False):
                if st.button(
                    translations.get("logout_btn", "🚫 Disconnect"),
                    type="secondary",
                    width="stretch",
                    key="global_logout_btn",
                ):
                    clear_authenticated_session(st.session_state, st.query_params)
                    st.switch_page("main.py")
                st.sidebar.divider()

            selected_sensor: str = st.selectbox(
                f"📍 {translations.get('select_station_adv', 'Select Zone')}",
                options=sensor_names,
                index=current_index,
                key="global_sensor_selectbox_widget",
            )

        return selected_sensor


def render_ai_assistant(
    location: str,
    temperature: float,
    air_quality: float,
    soil_moisture: float,
    translations: dict[str, str],
    noise_level: float = 0.0,
    traffic_load: float = 0.0,
) -> None:
    """Legacy interface orchestration wrapper mapping metrics into clean OOP."""
    reader = TelemetryContextReader()
    component = UrbanAiAssistantComponent(context_reader=reader)
    component.render(
        location=location,
        temperature=temperature,
        air_quality=air_quality,
        soil_moisture=soil_moisture,
        translations=translations,
        noise_level=noise_level,
        traffic_load=traffic_load,
    )


def render_full_global_sidebar(translations: dict[str, str]) -> Any:
    """Legacy UI architecture synchronization wrapper passing control arrays."""
    reader = TelemetryContextReader()
    sidebar_component = GlobalSidebarComponent(context_reader=reader)
    return sidebar_component.render_full_sidebar(translations=translations)
