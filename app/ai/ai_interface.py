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
from pydantic import ValidationError

# Path resilience injection framework for distributed virtual testing environments
PROJECT_ROOT: Final[str] = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.ai.prompt_registry import (  # noqa: E402
    PromptRegistryError,
    PromptRegistryLoader,
    PromptTemplateConfig,
)
from app.ai.rag_pipeline import (  # noqa: E402
    TelemetryEvidence,
    _canonical_unit,
    _extract_number_unit_mentions,
    _find_metric_spans,
    _split_clauses,
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


@dataclass(frozen=True, slots=True)
class AiGenerationTelemetry:
    """Timing and token usage reported for one model generation."""

    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    system_prompt_only: bool = False
    error: dict[str, Any] | str | None = None


@dataclass(frozen=True, slots=True)
class AiCompletionResult:
    """Cached recommendation text and its originating provider telemetry."""

    response_text: str
    telemetry: AiGenerationTelemetry | None = None


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


class SopKnowledgeRetriever:
    """Rank local SOP sections against telemetry-focused queries using TF-IDF."""

    def __init__(self, sop_path: str | Path | None = None) -> None:
        """Load and index the operational SOP passages."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        source_path = (
            Path(sop_path) if sop_path else Path(PROJECT_ROOT) / "docs" / "cluj_urban_sop.md"
        )
        document = source_path.read_text(encoding="utf-8")
        self._passages: list[str] = [
            section.strip()
            for section in re.split(r"(?m)(?=^##\s+)", document)
            if section.lstrip().startswith("## ")
        ]
        if not self._passages:
            raise ValueError(f"No SOP sections were found in '{source_path}'.")

        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            lowercase=True,
            ngram_range=(3, 5),
            strip_accents="unicode",
        )
        self._passage_vectors = self._vectorizer.fit_transform(self._passages)
        self._cosine_similarity = cosine_similarity

    def retrieve(self, query: str, top_k: int = 2) -> list[str]:
        """Return the highest-ranked matching SOP passages in stable order."""
        if top_k <= 0 or not query.strip():
            return []

        query_vector = self._vectorizer.transform([query])
        if query_vector.nnz == 0:
            return []

        scores = self._cosine_similarity(query_vector, self._passage_vectors).ravel()
        ranked_indexes = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        return [self._passages[index] for index in ranked_indexes if scores[index] > 0][:top_k]


@st.cache_resource(show_spinner=False)
def _cached_sop_knowledge_retriever() -> SopKnowledgeRetriever:
    """Reuse the fitted SOP index across Streamlit reruns."""
    return SopKnowledgeRetriever()


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
    system_prompt: str = "",
    system_prompt_only: bool = False,
) -> AiCompletionResult:
    """Cache recommendation text and its telemetry as one consistent result."""
    from app.ai.multi_provider import call_with_prompt_options

    telemetry_generator = getattr(_provider, "generate_completion_with_telemetry", None)
    if callable(telemetry_generator):
        response_text, telemetry = call_with_prompt_options(
            telemetry_generator,
            rendered_prompt,
            system_prompt=system_prompt,
            system_prompt_only=system_prompt_only,
        )
        telemetry_system_prompt_only = telemetry.get("system_prompt_only")
        return AiCompletionResult(
            response_text=response_text,
            telemetry=AiGenerationTelemetry(
                latency_ms=telemetry.get("latency_ms"),
                prompt_tokens=telemetry.get("prompt_tokens"),
                completion_tokens=telemetry.get("completion_tokens"),
                total_tokens=telemetry.get("total_tokens"),
                system_prompt_only=(
                    telemetry_system_prompt_only
                    if isinstance(telemetry_system_prompt_only, bool)
                    else system_prompt_only
                ),
                error=telemetry.get("error"),
            ),
        )

    return AiCompletionResult(
        response_text=call_with_prompt_options(
            _provider.generate_completion,
            rendered_prompt,
            system_prompt=system_prompt,
            system_prompt_only=system_prompt_only,
        )
    )


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
            return f"{heading} No recent station history; no trend can be established."

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
            return f"{heading} History unavailable; no trend can be established."

        readings: list[tuple[float, ...]] = []
        for row in reversed(rows):
            try:
                values = tuple(float(value) for value in row)
            except (TypeError, ValueError):
                continue
            if len(values) == 5 and all(math.isfinite(value) for value in values):
                readings.append(values)

        if len(readings) < 2:
            return f"{heading} Fewer than two valid readings; no trend can be established."

        metrics = (
            ("T", 0, "°C"),
            ("Noise", 1, "dB"),
            ("Traffic", 2, "%"),
            ("PM2.5", 3, "µg/m³"),
            ("Soil", 4, "%"),
        )
        trend_items = [f"n={len(readings)}"]
        for label, index, unit in metrics:
            start_value = readings[0][index]
            end_value = readings[-1][index]
            delta = round(end_value - start_value, 1)
            direction = "stable" if delta == 0 else "rising" if delta > 0 else "falling"
            trend_items.append(f"{label} {direction} Δ{delta:+g}{unit}")
        return f"{heading} " + "; ".join(trend_items)


class UrbanAiAssistantComponent:
    """Renders the intelligent AI advisory dashboard card with absolute state persistence."""

    def __init__(
        self,
        context_reader: ContextReaderInterface | None = None,
        ai_provider: AiProviderInterface | None = None,
    ) -> None:
        """Initialize the AI component anchoring deep template path variables."""
        from app.ai.multi_provider import FallbackAiProviderRouter

        self._context_reader: Final[ContextReaderInterface] = (
            context_reader or TelemetryContextReader()
        )
        self._ai_provider: Final[AiProviderInterface] = ai_provider or FallbackAiProviderRouter()
        self._sop_retriever: Final[SopKnowledgeRetriever] = _cached_sop_knowledge_retriever()
        configured_model = getattr(self._ai_provider, "configured_model", None)
        if isinstance(configured_model, str):
            self._ai_cache_key: Final[str] = (
                f"{type(self._ai_provider).__module__}."
                f"{type(self._ai_provider).__qualname__}:"
                f"{configured_model}"
            )
        else:
            self._ai_cache_key = (
                f"{type(self._ai_provider).__module__}."
                f"{type(self._ai_provider).__qualname__}:{id(self._ai_provider)}"
            )
        self._prompt_template_path: Final[Path] = PromptRegistryLoader.DEFAULT_PROMPT_PATH
        self._think_block_pattern: Final[re.Pattern] = re.compile(
            r"<think>.*?</think>", re.DOTALL | re.IGNORECASE
        )
        self._think_tag_pattern: Final[re.Pattern] = re.compile(r"</?think>", re.IGNORECASE)

    def _load_prompt_config(self) -> PromptTemplateConfig | None:
        """Load the validated prompt configuration and surface file/schema errors."""
        try:
            return PromptRegistryLoader.load(self._prompt_template_path)
        except (OSError, PromptRegistryError, ValidationError) as exc:
            st.error(
                f"❌ Prompt configuration '{self._prompt_template_path}' could not be loaded: {exc}"
            )
            return None

    def _load_prompt_template(self) -> str | None:
        """Preserve compatibility for callers that request only the user template."""
        prompt_config = self._load_prompt_config()
        return prompt_config.user_template if prompt_config else None

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
    def _target_metric_subclauses(
        cls,
        response: str,
        metric_aliases: tuple[str, ...],
    ) -> list[str]:
        """Extract clause slices owned exclusively by the target metric aliases."""
        normalized_aliases = tuple(
            sorted(
                {
                    normalized
                    for alias in metric_aliases
                    if (normalized := cls._normalize_grounding_text(alias).strip())
                },
                key=len,
                reverse=True,
            )
        )
        if not normalized_aliases:
            return []

        subclauses: list[str] = []
        for clause in _split_clauses(response):
            normalized_clause = cls._normalize_grounding_text(clause)
            target_spans: list[tuple[int, int, str]] = []
            for alias in normalized_aliases:
                for match in re.finditer(
                    rf"(?<!\w){re.escape(alias)}(?!\w)",
                    normalized_clause,
                ):
                    target_spans.append((match.start(), match.end(), "__target__"))
            if not target_spans:
                continue

            other_spans = [
                (start, end, metric)
                for start, end, metric in _find_metric_spans(normalized_clause)
                if not any(
                    not (end <= t_start or start >= t_end) for t_start, t_end, _ in target_spans
                )
            ]
            combined = sorted(
                (*target_spans, *other_spans),
                key=lambda item: (item[0], -(item[1] - item[0])),
            )
            deduped: list[tuple[int, int, str]] = []
            last_end = -1
            for start, end, tag in combined:
                if start >= last_end:
                    deduped.append((start, end, tag))
                    last_end = end

            for idx, (start, _end, tag) in enumerate(deduped):
                if tag != "__target__":
                    continue
                left = 0 if idx == 0 else start
                right = len(normalized_clause)
                for next_start, _next_end, next_tag in deduped[idx + 1 :]:
                    if next_tag != "__target__":
                        right = next_start
                        break
                subclauses.append(normalized_clause[left:right])
        return subclauses

    @classmethod
    def _response_mentions_metric_value(
        cls,
        response: str,
        metric_aliases: tuple[str, ...],
        value: float,
        unit: str = "",
    ) -> bool:
        expected_value = Decimal(str(value)).normalize()
        expected_unit = _canonical_unit(unit) if unit else None
        for subclause in cls._target_metric_subclauses(response, metric_aliases):
            for _, _, number, found_unit in _extract_number_unit_mentions(subclause):
                if number != expected_value:
                    continue
                if expected_unit is not None and found_unit != expected_unit:
                    continue
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
        unit: str = "",
    ) -> bool:
        expected_measured = Decimal(str(measured_value)).normalize()
        expected_threshold = Decimal(str(threshold)).normalize()
        expected_unit = _canonical_unit(unit) if unit else None
        direction_terms = (
            ("above", "exceeds", "greater than", "over", "depases", "peste", ">")
            if direction == "above"
            else ("below", "under", "less than", "sub", "mai putin", "<")
        )
        for subclause in cls._target_metric_subclauses(response, metric_aliases):
            if not any(term in subclause for term in direction_terms):
                continue
            mentions = _extract_number_unit_mentions(subclause)
            has_measured = any(
                number == expected_measured
                and (expected_unit is None or found_unit == expected_unit)
                for _, _, number, found_unit in mentions
            )
            has_threshold = any(
                number == expected_threshold
                and (expected_unit is None or found_unit == expected_unit)
                for _, _, number, found_unit in mentions
            )
            if has_measured and has_threshold:
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
        from app.ai.multi_provider import is_provider_failure

        if not response or is_provider_failure(response):
            return response

        normalized_response = cls._normalize_grounding_text(response)
        facts: list[str] = []
        normalized_location = cls._normalize_grounding_text(location)
        if normalized_location not in normalized_response:
            facts.append(f"Station: {location}.")

        missing_readings = [
            f"{label}: {value:g} {unit}"
            for label, aliases, value, unit, _, _ in readings
            if not cls._response_mentions_metric_value(response, aliases, value, unit=unit)
        ]
        if missing_readings:
            facts.append("Verified current readings: " + "; ".join(missing_readings) + ".")

        missing_breaches = []
        for label, aliases, value, unit, threshold, direction in readings:
            breached = value > threshold if direction == "above" else value < threshold
            if breached and not cls._response_mentions_threshold_breach(
                response,
                aliases,
                value,
                threshold,
                direction,
                unit=unit,
            ):
                relation = "exceeds" if direction == "above" else "is below"
                missing_breaches.append(
                    f"{label}: {value:g} {unit} {relation} configured threshold "
                    f"{threshold:g} {unit}"
                )
        if missing_breaches:
            facts.append(
                "Verified configured threshold breaches: " + "; ".join(missing_breaches) + "."
            )

        if not facts:
            return response
        return response.rstrip() + "\n\n**Verified Operational Facts:**\n" + "\n".join(facts)

    @staticmethod
    def _format_telemetry_badge(telemetry: AiGenerationTelemetry) -> str | None:
        if telemetry.latency_ms is None:
            return None
        token_values = (
            telemetry.prompt_tokens,
            telemetry.completion_tokens,
            telemetry.total_tokens,
        )
        token_summary = (
            " · tokens " + "/".join(str(value) for value in token_values)
            if all(value is not None for value in token_values)
            else " · token usage unavailable"
        )
        return f"LLM telemetry · {telemetry.latency_ms:.0f} ms{token_summary}"

    @staticmethod
    def _build_sop_query(
        temperature: float,
        air_quality: float,
        noise_level: float,
        traffic_load: float,
        soil_moisture: float,
        thresholds: SensorThresholdMatrix,
    ) -> str:
        metrics = (
            (
                temperature > thresholds.temperature_limit,
                "temperature temperatura heatwave heat wave val de caldura canicula",
            ),
            (
                air_quality > thresholds.air_quality_limit,
                "PM2.5 air quality calitatea aerului poluare particule fine",
            ),
            (
                noise_level > thresholds.noise_limit,
                "noise zgomot urban poluare fonica",
            ),
            (
                traffic_load > thresholds.traffic_limit,
                "traffic trafic congestion aglomeratie trafic intens",
            ),
            (
                soil_moisture < thresholds.soil_moisture_limit,
                "soil moisture umiditatea solului seceta drought",
            ),
        )
        breached_metrics = [terms for breached, terms in metrics if breached]
        selected_metrics = breached_metrics or [terms for _, terms in metrics]
        return " ".join(selected_metrics)

    @staticmethod
    def _compact_sop_passage(passage: str) -> str:
        """Keep the trigger and first verification step from one authorized SOP."""
        lines = passage.strip().splitlines()
        title = lines[0].removeprefix("## ").strip() if lines else "Urban SOP"
        trigger_match = re.search(
            r"\*\*Declanșare:\*\*\s*(.*?)(?:\n\n|$)",
            passage,
            flags=re.DOTALL,
        )
        action_match = re.search(
            r"(?m)^\s*1\.\s*(.*?)(?=^\s*\d+\.|\Z)",
            passage,
            flags=re.DOTALL,
        )

        def shorten(value: str, word_limit: int) -> str:
            words = value.split()
            return " ".join(words[:word_limit])

        trigger = shorten(trigger_match.group(1), 6) if trigger_match else ""
        first_step = shorten(action_match.group(1), 7) if action_match else ""
        details = "; ".join(
            part for part in (f"trigger: {trigger}" if trigger else "", first_step) if part
        )
        return f"{title}: {details}" if details else " ".join(lines)[:180]

    def render(
        self,
        location: str,
        temperature: float,
        air_quality: float,
        soil_moisture: float,
        translations: dict[str, str],
        noise_level: float = 0.0,
        traffic_load: float = 0.0,
        system_prompt_only: bool = False,
    ) -> None:
        """Render the cloud advisory component with defensive data verification blocks."""
        prompt_config = self._load_prompt_config()
        if prompt_config is None:
            return

        safe_location = sanitize_untrusted_input(location, max_length=200)
        session_key: Final[str] = f"persistent_ai_response_{safe_location.replace(' ', '_')}"
        telemetry_session_key: Final[str] = f"{session_key}_telemetry"
        if session_key not in st.session_state:
            st.session_state[session_key] = None
        if telemetry_session_key not in st.session_state:
            st.session_state[telemetry_session_key] = None

        active_lang: Final[str] = st.session_state.get("lang", "RO")
        historical_alerts_log: Final[str] = self._context_reader.get_latest_alerts_context(
            limit_lines=5
        )
        safe_historical_alerts = sanitize_untrusted_input(
            historical_alerts_log,
            max_length=220,
        )

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
        sop_query = self._build_sop_query(
            temperature,
            air_quality,
            noise_level,
            traffic_load,
            soil_moisture,
            thresholds,
        )
        sop_passages = self._sop_retriever.retrieve(sop_query, top_k=2)
        sop_context = "\n".join(self._compact_sop_passage(passage) for passage in sop_passages)
        recent_trend = self._context_reader.fetch_recent_telemetry_trend(safe_location)
        retrieved_context = (
            f"{format_retrieved_context(retrieved_records)}\n\n"
            f"{sanitize_untrusted_input(recent_trend)}"
        )
        if sop_context:
            retrieved_context += (
                "\n\n[AUTHORIZED URBAN SOP CONTEXT]\n"
                f"{sanitize_untrusted_input(sop_context, max_length=420)}"
            )

        rendered_prompt = PromptRegistryLoader.render_user_template(
            prompt_config,
            {
                "locatie": safe_location,
                "temperature": f"{temperature:.1f}",
                "air_quality": f"{air_quality:.1f}",
                "soil_moisture": f"{soil_moisture:.1f}",
                "noise_level": f"{noise_level:.1f}",
                "traffic_load": f"{traffic_load:.1f}",
                "limba_activa": active_lang,
                "jurnal_alerte_recente": safe_historical_alerts,
            },
        )
        rendered_prompt = (
            f"{rendered_prompt.rstrip()}\n\n[RETRIEVED CONTEXT]\n"
            f"{retrieved_context}\n[/RETRIEVED CONTEXT]"
        )

        if st.button(
            translations.get("generate_rec", "Generate Recommendations"),
            type="secondary",
            key=f"ai_generate_btn_{safe_location.replace(' ', '_')}",
            width="stretch",
        ):
            st.session_state[telemetry_session_key] = None
            with st.spinner(translations.get("loading_ai", "Analyzing sensor telemetry...")):
                try:
                    completion = _cached_ai_completion(
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
                        system_prompt=PromptRegistryLoader.build_system_prompt(prompt_config),
                        system_prompt_only=system_prompt_only,
                    )
                    response_text = completion.response_text
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
                        st.session_state[telemetry_session_key] = {
                            "telemetry": completion.telemetry,
                            "system_prompt_only": (
                                completion.telemetry.system_prompt_only
                                if completion.telemetry is not None
                                else system_prompt_only
                            ),
                        }
                    else:
                        st.warning("⚠️ Remote cloud gateway engine returned null.")
                except Exception as exc:
                    st.session_state[telemetry_session_key] = None
                    st.error(f"❌ Critical connection error: {exc}")

        if st.session_state[session_key]:
            from app.ai.multi_provider import is_provider_failure

            st.markdown("<br>", unsafe_allow_html=True)
            response_payload: str = str(st.session_state[session_key])
            telemetry_state = st.session_state.get(telemetry_session_key)

            if is_provider_failure(
                response_payload,
                telemetry_state if isinstance(telemetry_state, dict) else None,
            ):
                st.error(response_payload)
            else:
                with st.chat_message("assistant"):
                    st.markdown(response_payload)
                telemetry = (
                    telemetry_state.get("telemetry") if isinstance(telemetry_state, dict) else None
                )
                if isinstance(telemetry, AiGenerationTelemetry):
                    telemetry_badge = self._format_telemetry_badge(telemetry)
                    if telemetry_badge:
                        st.caption(telemetry_badge)


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
                fallback = st.session_state.get("lang", "RO")
                st.session_state["lang"] = st.session_state.get("global_language_widget", fallback)

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
    system_prompt_only: bool = False,
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
        system_prompt_only=system_prompt_only,
    )


def render_full_global_sidebar(translations: dict[str, str]) -> Any:
    """Legacy UI architecture synchronization wrapper passing control arrays."""
    reader = TelemetryContextReader()
    sidebar_component = GlobalSidebarComponent(context_reader=reader)
    return sidebar_component.render_full_sidebar(translations=translations)
