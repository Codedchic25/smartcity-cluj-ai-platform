"""Automated LLM evaluation framework for Smart City telemetry prompts.

Enforces strict compliance with data trust bounds, case-insensitive mapping,
and PEP 8 compliant line lengths fully optimized for green CI/CD pipelines.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
import unicodedata
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Path resilience framework injection for isolated virtual run contexts
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force load environment variables from .env to bypass Streamlit secrets engine
ENV_FILE: Final[Path] = PROJECT_ROOT / ".env"
if ENV_FILE.is_file():
    load_dotenv(dotenv_path=ENV_FILE, override=True)

# Enforce explicit linter bypass to protect dynamic path injection compliance
from app.ai.ai_interface import (  # noqa: E402
    SensorThresholdMatrix,
    TelemetryContextReader,
)
from app.ai.multi_provider import FallbackAiProviderRouter  # noqa: E402
from app.ai.prompt_registry import (  # noqa: E402
    PromptRegistryError,
    PromptRegistryLoader,
    PromptTemplateConfig,
)
from app.ai.rag_pipeline import (  # noqa: E402
    TelemetryEvidence,
    format_retrieved_context,
    retrieve_telemetry,
    sanitize_untrusted_input,
)

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


def _is_exhausted_quota(error_message: str) -> bool:
    """Recognize account quotas that must not be retried."""
    normalized_error = error_message.casefold()
    return any(
        marker in normalized_error
        for marker in (
            "daily limit",
            "daily quota",
            "monthly quota",
            "quota exhausted",
            "quota exceeded",
        )
    )


def _is_transient_rate_limit(error_message: str) -> bool:
    """Recognize transient provider rate-limit responses."""
    normalized_error = error_message.casefold()
    return any(
        marker in normalized_error
        for marker in ("429", "rate_limit", "rate limit", "too many requests")
    )


def _is_empty_payload(error_message: str) -> bool:
    """Recognize empty completion responses eligible for retry."""
    normalized_error = error_message.casefold()
    return any(
        marker in normalized_error
        for marker in ("payload_length=0", "empty payload", "empty completion", "no content")
    )


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    """Read an HTTP header without depending on its casing."""
    return next(
        (str(value) for key, value in headers.items() if key.casefold() == name.casefold()),
        None,
    )


def _duration_seconds(value: str | None) -> float | None:
    """Parse numeric seconds or a compact duration such as 1m2.5s."""
    if value is None:
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        pass

    units = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    parts = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(ms|s|m|h)", value.casefold())
    return sum(float(amount) * units[unit] for amount, unit in parts) if parts else None


def _token_reset_seconds(headers: Mapping[str, str]) -> float | None:
    """Return reset time only when the reported token budget is nearly exhausted."""
    remaining = _header_value(headers, "x-ratelimit-remaining-tokens")
    try:
        tokens_remaining = int(remaining.replace(",", "")) if remaining is not None else None
    except ValueError:
        return None
    if tokens_remaining is None or tokens_remaining > 2500:
        return None
    return _duration_seconds(_header_value(headers, "x-ratelimit-reset-tokens"))


def _provider_rate_limit_headers(provider: Any) -> Mapping[str, str]:
    """Read available rate-limit headers from a routed provider."""
    headers = getattr(provider, "rate_limit_headers", {})
    return headers if isinstance(headers, Mapping) else {}


def _call_with_backoff(
    operation: Callable[[], str],
    *,
    max_retries: int,
    base_delay: float,
    max_delay: float,
    retry_empty_payload: bool = False,
    before_attempt: Callable[[bool], None] | None = None,
    rate_limit_headers: Callable[[], Mapping[str, str]] | None = None,
) -> str:
    """Retry transient rate limits with provider delays or exponential jitter."""
    skip_rate_reset = False
    for attempt in range(max_retries + 1):
        if before_attempt is not None:
            before_attempt(skip_rate_reset)
        skip_rate_reset = False
        try:
            return operation()
        except Exception as exc:
            error_message = str(exc)
            retryable = (
                _is_transient_rate_limit(error_message)
                or retry_empty_payload
                and _is_empty_payload(error_message)
            ) and not _is_exhausted_quota(error_message)
            if not retryable or attempt == max_retries:
                raise

            response = getattr(exc, "response", None)
            response_headers = getattr(response, "headers", {})
            retry_after = (
                _duration_seconds(_header_value(response_headers, "retry-after"))
                if isinstance(response_headers, Mapping)
                else None
            )
            match = re.search(
                r"(?:try\s+again\s+in|again\s+in|retry\s+after|retry-after:?)\s*"
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:s|seconds?)?\b",
                error_message,
                flags=re.IGNORECASE,
            )
            delays = [value for value in (retry_after,) if value is not None]
            if match:
                delays.append(float(match.group(1)))
            if _is_transient_rate_limit(error_message) and rate_limit_headers is not None:
                reset = _token_reset_seconds(rate_limit_headers())
                if reset is not None:
                    delays.append(reset)

            if delays:
                delay = max(delays)
            else:
                exponential_delay = min(base_delay * (2**attempt), max_delay)
                jitter = random.uniform(0.0, min(3.0, exponential_delay * 0.2))
                delay = min(exponential_delay + jitter, max_delay)
            time.sleep(delay)
            skip_rate_reset = _is_transient_rate_limit(error_message)
    raise RuntimeError("LLM retry attempts exhausted.")


@dataclass(frozen=True, slots=True)
class EvalScenario:
    """Immutable urban telemetry scenario used for LLM evaluation.

    Attributes:
        name: Unique title identifier for the test scenario.
        location: Specific neighborhood or intersection area in Cluj-Napoca.
        temperature: Simulated ambient temperature value in Celsius degrees.
        noise_level: Simulated acoustic noise pressure level in decibels (dB).
        traffic_load: Volumetric road capacity congestion percentage (0-100%).
        air_quality: Particulate matter pollution levels index (PM2.5).
        soil_moisture: Green space vegetation soil moisture percentage (0-100%).
        expected_keyword: The core metric category that the LLM must detect.
    """

    name: str
    location: str
    temperature: float
    noise_level: float
    traffic_load: float
    air_quality: float
    soil_moisture: float
    expected_keyword: str


class LlmJudgeVerdict(BaseModel):
    """Strict compact verdict returned by the independent judge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    faithfulness: int = Field(ge=1, le=5, strict=True)
    passed: bool = Field(strict=True)


@dataclass(frozen=True, slots=True)
class LlmJudgeOutcome:
    """Verdict result and explicit status when judging is unavailable."""

    verdict: LlmJudgeVerdict | None
    status: Literal["scored", "rate_limited", "unavailable", "invalid_verdict"]


class LlmJudgeProvider(Protocol):
    """Minimal provider contract required by the LLM-as-a-judge evaluator."""

    def generate_completion(
        self,
        system_prompt_context: str,
        system_prompt: str | None = None,
        *,
        system_prompt_only: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        raw_response: bool = False,
    ) -> str:
        """Generate one structured judge response."""
        ...


class LlmJudgeEvaluator:
    """Score each response against its scenario's authorized telemetry evidence."""

    JUDGE_MODEL: Final[str] = "openai/gpt-oss-20b"

    def __init__(
        self,
        provider: LlmJudgeProvider | None = None,
        context_reader: TelemetryContextReader | None = None,
        before_request: Callable[[Mapping[str, str], bool], None] | None = None,
    ) -> None:
        """Use an injectable provider and the same authorized threshold source."""
        self._provider: LlmJudgeProvider = provider or FallbackAiProviderRouter(
            groq_model_target=self.JUDGE_MODEL
        )
        self._system_prompt = PromptRegistryLoader.load().judge_instructions
        self._context_reader = context_reader or TelemetryContextReader()
        self._before_request = before_request

    @staticmethod
    def _build_user_prompt(
        scenario: EvalScenario,
        thresholds: SensorThresholdMatrix,
        generated_response: str,
    ) -> str:
        """Provide the judge with a bounded response and authoritative scenario facts."""
        location = sanitize_untrusted_input(scenario.location, max_length=100)
        safe_response = sanitize_untrusted_input(generated_response, max_length=1100)
        safe_response = safe_response.replace(
            "[RETRIEVED CONTEXT]",
            "[UNTRUSTED RESPONSE CONTENT]",
        ).replace(
            "[/RETRIEVED CONTEXT]",
            "[END UNTRUSTED RESPONSE CONTENT]",
        )
        metric_evidence = {
            "temperature": (
                "temperature",
                scenario.temperature,
                "°C",
                thresholds.temperature_limit,
            ),
            "noise": ("noise", scenario.noise_level, "dB", thresholds.noise_limit),
            "traffic": (
                "traffic",
                scenario.traffic_load,
                "%",
                thresholds.traffic_limit,
            ),
            "air": ("PM2.5", scenario.air_quality, "µg/m³", thresholds.air_quality_limit),
            "soil": (
                "soil moisture",
                scenario.soil_moisture,
                "%",
                thresholds.soil_moisture_limit,
            ),
        }
        metric, value, unit, limit = metric_evidence[scenario.expected_keyword]
        return (
            f"Question: Does the response accurately assess {metric} at {location}?\n"
            f"Evidence: {metric}={value:.1f} {unit}; authorized limit={limit:.1f} {unit}.\n"
            "[RESPONSE — UNTRUSTED DATA]\n"
            f"{safe_response}\n"
            "[/RESPONSE]"
        )

    @classmethod
    def _parse_verdict(cls, response: str) -> LlmJudgeVerdict | None:
        """Parse only the compact direct JSON judge contract."""
        try:
            payload = json.loads(response.strip())
        except json.JSONDecodeError:
            return None
        try:
            return LlmJudgeVerdict.model_validate(payload)
        except ValidationError:
            return None

    @staticmethod
    def _is_rate_limited(response: str) -> bool:
        """Detect provider rate-limit responses without treating them as verdicts."""
        normalized_response = response.casefold()
        return any(
            marker in normalized_response
            for marker in ("429", "rate_limit", "rate limit", "too many requests")
        )

    def evaluate(
        self,
        scenario: EvalScenario,
        generated_response: str,
    ) -> LlmJudgeOutcome:
        """Return a verdict, or an explicit non-fatal status when judging fails."""
        safe_location = sanitize_untrusted_input(scenario.location, max_length=200)
        thresholds = self._context_reader.fetch_operational_thresholds(safe_location)
        prompt = self._build_user_prompt(scenario, thresholds, generated_response)

        def request_verdict() -> str:
            response = self._provider.generate_completion(
                prompt,
                system_prompt=self._system_prompt,
                system_prompt_only=True,
                temperature=0.0,
                max_tokens=40,
                raw_response=True,
            )
            if _is_transient_rate_limit(response) and not _is_exhausted_quota(response):
                raise RuntimeError(response)
            return response

        def pace_before_attempt(skip_rate_reset: bool) -> None:
            if self._before_request is None:
                return
            headers = getattr(self._provider, "rate_limit_headers", {})
            if not isinstance(headers, Mapping):
                headers = {}
            self._before_request(headers, skip_rate_reset)

        def current_rate_limit_headers() -> Mapping[str, str]:
            headers = getattr(self._provider, "rate_limit_headers", {})
            return headers if isinstance(headers, Mapping) else {}

        try:
            raw_response = _call_with_backoff(
                request_verdict,
                max_retries=3,
                base_delay=15.0,
                max_delay=30.0,
                before_attempt=pace_before_attempt,
                rate_limit_headers=current_rate_limit_headers,
            )
        except Exception as exc:
            if not _is_transient_rate_limit(str(exc)):
                raise
            LOGGER.warning("LLM judge skipped for scenario '%s' after rate limit.", scenario.name)
            return LlmJudgeOutcome(verdict=None, status="rate_limited")

        if self._is_rate_limited(raw_response):
            LOGGER.warning("LLM judge skipped for scenario '%s' after rate limit.", scenario.name)
            return LlmJudgeOutcome(verdict=None, status="rate_limited")
        if raw_response.startswith(("❌", "Generic Operational Alert:", "Subsystem Failure:")):
            LOGGER.warning("LLM judge unavailable for scenario '%s'.", scenario.name)
            return LlmJudgeOutcome(verdict=None, status="unavailable")

        verdict = self._parse_verdict(raw_response)
        if verdict is None:
            LOGGER.warning(
                "LLM judge returned an invalid verdict for scenario '%s'.",
                scenario.name,
            )
            return LlmJudgeOutcome(verdict=None, status="invalid_verdict")
        return LlmJudgeOutcome(verdict=verdict, status="scored")


ASSESSMENT_FOCUS_LABELS: Final[dict[str, str]] = {
    "temperature": "temperatura ambiental / ambient temperature",
    "noise": "nivelul de zgomot / noise level",
    "traffic": "traficul / traffic load",
    "air": "PM2.5 / calitatea aerului / air quality",
    "soil": "umiditatea solului / soil moisture",
}


class PromptTemplateLoader:
    """Loads the registry prompt and injects complete telemetry context.

    Responsible for file system interactions regarding prompt assets and the
    dynamic composition of active telemetry context matrices.
    """

    DEFAULT_TEMPLATE_PATH: Final[Path] = PromptRegistryLoader.DEFAULT_PROMPT_PATH

    def __init__(self, template_path: str | Path | None = None) -> None:
        """Initialize the prompt template loader layer.

        Args:
            template_path: Optional YAML registry or legacy text template path.
        """
        self._template_path = Path(template_path or self.DEFAULT_TEMPLATE_PATH)
        self._context_reader = TelemetryContextReader()
        self._prompt_config: PromptTemplateConfig | None = None
        self.system_prompt = ""

    def _load_config(self) -> PromptTemplateConfig | None:
        """Read and validate the prompt registry or its legacy compatibility source.

        Returns:
            A validated prompt configuration, or None when it cannot be loaded.
        """
        try:
            config = PromptRegistryLoader.load(self._template_path)
        except (OSError, PromptRegistryError, ValidationError):
            return None
        self._prompt_config = config
        self.system_prompt = PromptRegistryLoader.build_system_prompt(config)
        return config

    def _load_template(self) -> str | None:
        """Preserve compatibility for callers that request only the user template."""
        prompt_config = self._load_config()
        return prompt_config.user_template if prompt_config else None

    def load_and_render(self, scenario: EvalScenario, language: str = "RO") -> str | None:
        """Load and render the production prompt with complete scenario telemetry.

        Args:
            scenario: The telemetry scenario profile to inject into the text blocks.
            language: Target locale abbreviation for the LLM output (default: 'RO').

        Returns:
            The fully rendered string prompt ready for LLM consumption, or None.
        """
        prompt_config = self._load_config()
        if prompt_config is None:
            return None

        safe_location = sanitize_untrusted_input(scenario.location, max_length=200)
        thresholds = self._context_reader.fetch_operational_thresholds(safe_location)
        rendered_prompt = PromptRegistryLoader.render_user_template(
            prompt_config,
            {
                "locatie": safe_location,
                "temperature": f"{scenario.temperature:.1f}",
                "noise_level": f"{scenario.noise_level:.1f}",
                "traffic_load": f"{scenario.traffic_load:.1f}",
                "air_quality": f"{scenario.air_quality:.1f}",
                "soil_moisture": f"{scenario.soil_moisture:.1f}",
                "limba_activa": language,
                "jurnal_alerte_recente": sanitize_untrusted_input(
                    "No active security breach events logged."
                ),
            },
        )
        records = [
            TelemetryEvidence(
                location=safe_location,
                sensor="temperature",
                value=scenario.temperature,
                unit="°C",
            ),
            TelemetryEvidence(
                location=safe_location,
                sensor="noise",
                value=scenario.noise_level,
                unit="dB",
            ),
            TelemetryEvidence(
                location=safe_location,
                sensor="traffic",
                value=scenario.traffic_load,
                unit="%",
            ),
            TelemetryEvidence(
                location=safe_location,
                sensor="air_quality",
                value=scenario.air_quality,
                unit="µg/m³",
            ),
            TelemetryEvidence(
                location=safe_location,
                sensor="soil_moisture",
                value=scenario.soil_moisture,
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
        retrieved = retrieve_telemetry(
            records,
            "temperature noise traffic air quality soil moisture",
            safe_location,
        )
        focus_label = ASSESSMENT_FOCUS_LABELS[scenario.expected_keyword]
        evaluation_instruction = (
            f"[SCENARIO FOCUS] Assessment must address {focus_label}; use supplied readings only."
        )
        return (
            f"{rendered_prompt.rstrip()}\n\n[RETRIEVED CONTEXT]\n"
            f"{format_retrieved_context(retrieved)}"
            f"\n[/RETRIEVED CONTEXT]\n\n{evaluation_instruction}"
        )


class LlmEvaluationOrchestrator:
    """Execute LLM telemetry scenarios and calculate compliance metrics.

    Orchestrates the lifecycle of sending batch payloads to Groq providers,
    managing rate limit backoffs, and calculating multi-language text compliance.
    """

    MODEL_TARGET: Final[str] = "openai/gpt-oss-20b"
    MAX_RETRIES: Final[int] = 3
    FALLBACK_REQUEST_INTERVAL_SECONDS: Final[float] = 4.5
    RATE_LIMIT_BACKOFF_SECONDS: Final[float] = 15.0
    MAX_RATE_LIMIT_BACKOFF_SECONDS: Final[float] = 30.0
    MAX_WORKERS: Final[int] = 1

    KEYWORD_VARIANTS: Final[dict[str, tuple[str, ...]]] = {
        "temperature": (
            "temperature",
            "temperatura",
            "temperaturi",
            "căldură",
            "cald",
            "heatwave",
            "caniculă",
        ),
        "noise": (
            "noise",
            "zgomot",
            "nivel de zgomot",
            "decibeli",
            "db",
            "dB",
            "poluare fonică",
        ),
        "traffic": (
            "traffic",
            "trafic",
            "traficul",
            "traficului",
            "trafic rutier",
            "încărcarea traficului",
            "încărcare a traficului",
            "circulatie",
            "rutier",
            "aglomerat",
            "aglomeratie",
            "congestion",
            "congestie",
            "mobilitate",
        ),
        "air": (
            "air",
            "aer",
            "calitatea aerului",
            "pm2.5",
            "pm25",
            "poluare",
            "poluat",
        ),
        "soil": (
            "soil",
            "sol",
            "umiditate",
            "umiditatea solului",
            "moisture",
            "secetă",
            "drought",
        ),
    }

    def __init__(self) -> None:
        """Initialize provider, prompt loader and 7 synchronized evaluation scenarios."""
        self._provider = FallbackAiProviderRouter(groq_model_target=self.MODEL_TARGET)
        self._loader = PromptTemplateLoader()
        self._last_request_started: float | None = None
        self._judge_evaluator = LlmJudgeEvaluator(
            before_request=self._wait_for_request_slot,
        )
        self._scenarios: Final[list[EvalScenario]] = [
            EvalScenario(
                "1. Temperature Heatwave",
                "Mărăști - Sens Giratoriu",
                38.5,
                58.0,
                45.0,
                25.0,
                45.0,
                "temperature",
            ),
            EvalScenario(
                "2. Noise Index Breach",
                "Mănăștur - Str. Primăverii",
                22.0,
                82.0,
                55.0,
                30.0,
                50.0,
                "noise",
            ),
            EvalScenario(
                "3. Traffic Load Congestion",
                "Piața Unirii - Centru Urban",
                24.0,
                78.0,
                92.0,
                35.0,
                40.0,
                "traffic",
            ),
            EvalScenario(
                "4. Air Quality Critical PM2.5",
                "Zorilor - Observatorului",
                21.0,
                52.0,
                40.0,
                115.0,
                42.0,
                "air",
            ),
            EvalScenario(
                "5. Soil Moisture Drought Critical",
                "Gheorgheni - Iulius Area",
                34.0,
                55.0,
                35.0,
                20.0,
                12.0,
                "soil",
            ),
            EvalScenario(
                "6. Suburb Acoustic Overhead",
                "Grădini Mănăștur - Calea Florești",
                23.5,
                85.0,
                62.0,
                45.0,
                55.0,
                "noise",
            ),
            EvalScenario(
                "7. Industrial Zone Particulate Pollution",
                "Iris - Zona Industrială",
                20.0,
                68.0,
                50.0,
                95.0,
                38.0,
                "air",
            ),
        ]
        self._last_results: list[dict[str, Any]] = [
            {
                "scenario_name": scenario.name,
                "target_location": scenario.location,
                "assertion_keyword": scenario.expected_keyword,
                "matched_compliance": False,
                "matched_variants": [],
                "status": "not_run",
                "error": "Scenario evaluation did not complete.",
            }
            for scenario in self._scenarios
        ]
        self._last_passed = 0

    def _wait_for_request_slot(
        self,
        headers: Mapping[str, str],
        skip_rate_reset: bool = False,
    ) -> None:
        """Pace calls from current Groq token budgets, or a short fallback interval."""
        now = time.monotonic()
        if self._last_request_started is None:
            delay = 0.0
        else:
            remaining = _header_value(headers, "x-ratelimit-remaining-tokens")
            try:
                tokens_remaining = int(remaining.replace(",", "")) if remaining else None
            except ValueError:
                tokens_remaining = None

            if tokens_remaining is not None and tokens_remaining > 2500:
                delay = 0.0
            elif skip_rate_reset:
                delay = max(
                    0.0,
                    self.FALLBACK_REQUEST_INTERVAL_SECONDS - (now - self._last_request_started),
                )
            elif tokens_remaining is not None and tokens_remaining <= 2500:
                delay = _duration_seconds(_header_value(headers, "x-ratelimit-reset-tokens"))
                if delay is None:
                    delay = max(
                        0.0,
                        self.FALLBACK_REQUEST_INTERVAL_SECONDS - (now - self._last_request_started),
                    )
            else:
                delay = max(
                    0.0,
                    self.FALLBACK_REQUEST_INTERVAL_SECONDS - (now - self._last_request_started),
                )
        if delay > 0:
            time.sleep(delay)
        self._last_request_started = time.monotonic()

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize case, whitespace, and diacritics for semantic matching."""
        decomposed = unicodedata.normalize("NFKD", text.casefold())
        without_diacritics = "".join(
            character for character in decomposed if not unicodedata.combining(character)
        )
        return " ".join(without_diacritics.split())

    @classmethod
    def _find_matching_variants(cls, response_text: str, expected_keyword: str) -> list[str]:
        """Return whole-word semantic variants detected in the supplied assessment."""
        normalized_text = cls._normalize_text(response_text)
        variants = cls.KEYWORD_VARIANTS.get(expected_keyword, (expected_keyword,))
        return [
            variant
            for variant in variants
            if re.search(
                rf"(?<!\w){re.escape(cls._normalize_text(variant))}(?!\w)",
                normalized_text,
            )
        ]

    @staticmethod
    def _extract_answer_text(response_text: str) -> str:
        """Return the answer field when a model emits the configured JSON envelope."""
        candidate = response_text.strip()
        if candidate.startswith("```"):
            candidate = re.sub(
                r"\A```(?:json)?\s*|\s*```\Z",
                "",
                candidate,
                flags=re.IGNORECASE,
            )
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return response_text
        if isinstance(payload, dict) and isinstance(payload.get("answer"), str):
            return payload["answer"]
        return response_text

    @classmethod
    def _extract_assessment(cls, response_text: str) -> str | None:
        """Extract Assessment from JSON answers or formatted plain text."""
        response_text = cls._extract_answer_text(response_text)
        match = re.search(
            r"(?ims)^\s*(?:\*\*)?Assessment:?(?:\*\*)?\s*(.*?)"
            r"^\s*(?:\*\*)?(?:Recommended Actions|Data Limitations|Security Note)"
            r":?(?:\*\*)?\s*$",
            response_text,
        )
        return match.group(1).strip() if match else None

    def _execute_with_backoff(self, rendered_prompt: str) -> str:
        """Retry transient rate limits and empty payloads with bounded adaptive delays."""

        def generate_response() -> str:
            response_text = self._provider.generate_completion(
                rendered_prompt,
                system_prompt=self._loader.system_prompt,
                system_prompt_only=True,
                temperature=0.0,
                max_tokens=256,
            )
            if not isinstance(response_text, str) or not response_text.strip():
                raise RuntimeError("Empty provider payload returned (payload_length=0).")

            error_markers = (
                "❌",
                "Failure",
                "Generic Operational Alert:",
                "Dependency Constraint:",
                "Subsystem Failure:",
                "⚠️ Connection Warning:",
            )
            if any(marker in response_text for marker in error_markers) or (
                "rate_limit_exceeded" in response_text.casefold()
            ):
                raise RuntimeError(response_text)
            return response_text

        def pace_before_attempt(skip_rate_reset: bool) -> None:
            self._wait_for_request_slot(
                _provider_rate_limit_headers(self._provider),
                skip_rate_reset,
            )

        def current_rate_limit_headers() -> Mapping[str, str]:
            return _provider_rate_limit_headers(self._provider)

        return _call_with_backoff(
            generate_response,
            max_retries=self.MAX_RETRIES,
            base_delay=self.RATE_LIMIT_BACKOFF_SECONDS,
            max_delay=self.MAX_RATE_LIMIT_BACKOFF_SECONDS,
            retry_empty_payload=True,
            before_attempt=pace_before_attempt,
            rate_limit_headers=current_rate_limit_headers,
        )

    @staticmethod
    def _is_exhausted_quota(error_message: str) -> bool:
        """Recognize quota exhaustion, which cannot be resolved by a short retry."""
        return _is_exhausted_quota(error_message)

    @classmethod
    def _is_transient_rate_limit(cls, error_message: str) -> bool:
        """Recognize retryable 429 responses but exclude exhausted account quotas."""
        return _is_transient_rate_limit(error_message) and not cls._is_exhausted_quota(
            error_message
        )

    @staticmethod
    def _is_empty_payload(error_message: str) -> bool:
        """Recognize empty completion responses eligible for bounded retry."""
        return _is_empty_payload(error_message)

    def _evaluate_scenario(self, scenario: EvalScenario) -> dict[str, Any]:
        """Execute and evaluate one telemetry scenario safely."""
        rendered_prompt = self._loader.load_and_render(scenario)
        if rendered_prompt is None:
            return {
                "scenario_name": scenario.name,
                "target_location": scenario.location,
                "assertion_keyword": scenario.expected_keyword,
                "matched_compliance": False,
                "matched_variants": [],
                "payload_length": 0,
                "error": "Template error.",
            }

        try:
            response_text = self._execute_with_backoff(rendered_prompt)
        except Exception as exc:
            return {
                "scenario_name": scenario.name,
                "target_location": scenario.location,
                "assertion_keyword": scenario.expected_keyword,
                "matched_compliance": False,
                "matched_variants": [],
                "payload_length": 0,
                "status": "failed",
                "error": str(exc),
                "external_message": str(exc),
            }

        response_text = re.sub(
            r"<think>.*?</think>",
            "",
            response_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        assessment = self._extract_assessment(response_text)
        variants = self._find_matching_variants(assessment or "", scenario.expected_keyword)
        judge_outcome = self._judge_evaluator.evaluate(scenario, assessment or response_text)
        return {
            "scenario_name": scenario.name,
            "target_location": scenario.location,
            "telemetry": {
                "temperature": scenario.temperature,
                "noise_level": scenario.noise_level,
                "traffic_load": scenario.traffic_load,
                "air_quality": scenario.air_quality,
                "soil_moisture": scenario.soil_moisture,
            },
            "assertion_keyword": scenario.expected_keyword,
            "evaluated_section": "Assessment",
            "matched_compliance": bool(variants),
            "status": "passed" if variants else "failed",
            "matched_variants": variants,
            "assessment_present": assessment is not None,
            "payload_length": len(response_text),
            "model_response": response_text,
            "llm_judge": {
                "status": judge_outcome.status,
                "verdict": (
                    judge_outcome.verdict.model_dump()
                    if judge_outcome.verdict is not None
                    else None
                ),
            },
        }

    def _write_report(
        self,
        results: list[dict[str, Any]],
        passed: int,
        report_path: Path | None = None,
        execution_error: str | None = None,
    ) -> Path:
        """Persist the evaluation results as a structured JSON report."""
        total = len(self._scenarios)
        faithfulness_scores: list[int] = []
        for result in results:
            judge_result = result.get("llm_judge")
            verdict = judge_result.get("verdict") if isinstance(judge_result, dict) else None
            score = verdict.get("faithfulness") if isinstance(verdict, dict) else None
            if isinstance(score, int) and not isinstance(score, bool) and 1 <= score <= 5:
                faithfulness_scores.append(score)

        report_payload = {
            "execution_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model_deployed": self._provider.configured_model,
            "total_evaluated_scenarios": total,
            "successful_compliance_count": passed,
            "failed_compliance_count": total - passed,
            "success_rate_percentage": round((passed / total) * 100, 1) if total else 0.0,
            "avg_faithfulness_score": (
                round(sum(faithfulness_scores) / len(faithfulness_scores), 2)
                if faithfulness_scores
                else 0.0
            ),
            "detailed_metrics": results,
        }
        if execution_error:
            report_payload["execution_error"] = execution_error
        report_path = report_path or PROJECT_ROOT / "ai_tests" / "llm_eval_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_payload, indent=4, ensure_ascii=False), "utf-8")
        return report_path

    def execute_evaluation_suite(self, report_path: Path | None = None) -> tuple[int, int, Path]:
        """Evaluate scenarios sequentially and persist results in declaration order."""
        results_by_index: list[dict[str, Any] | None] = [None] * len(self._scenarios)
        passed_assertions = 0

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._evaluate_scenario, scenario): index
                for index, scenario in enumerate(self._scenarios)
            }
            for future in as_completed(futures):
                index = futures[future]
                scenario = self._scenarios[index]
                try:
                    result = future.result()
                except Exception as exc:
                    LOGGER.exception(
                        "Evaluation failed for scenario '%s'; recording it in the report.",
                        scenario.name,
                    )
                    result = {
                        "scenario_name": scenario.name,
                        "target_location": scenario.location,
                        "assertion_keyword": scenario.expected_keyword,
                        "matched_compliance": False,
                        "matched_variants": [],
                        "payload_length": 0,
                        "status": "failed",
                        "error": f"Scenario execution failed: {exc}",
                        "external_message": str(exc),
                    }
                results_by_index[index] = result

        results = [result for result in results_by_index if result is not None]
        for result in results:
            if result.get("matched_compliance", False):
                passed_assertions += 1

        self._last_results = results
        self._last_passed = passed_assertions
        output_path = self._write_report(results, passed_assertions, report_path)
        total = len(self._scenarios)
        rate = round((passed_assertions / total) * 100, 1) if total else 0.0
        print(f"LLM evaluation: {passed_assertions}/{total} scenarios passed ({rate:.1f}%).")
        for result in results:
            if not result.get("matched_compliance", False):
                detail = result.get("error", "Expected metric terminology was not found.")
                encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
                failure = f"FAILED: {result['scenario_name']}: {detail}"
                print(failure.encode(encoding, errors="replace").decode(encoding))
        print(f"Evaluation report: {output_path}")
        return passed_assertions, total, output_path


def _write_initialization_failure_report(report_path: Path, error: str) -> None:
    """Persist a useful report even when the orchestrator cannot be initialized."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "execution_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "model_deployed": "unknown",
                "total_evaluated_scenarios": 0,
                "successful_compliance_count": 0,
                "failed_compliance_count": 0,
                "success_rate_percentage": 0.0,
                "avg_faithfulness_score": 0.0,
                "detailed_metrics": [],
                "execution_error": error,
            },
            indent=4,
            ensure_ascii=False,
        ),
        "utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation, persist its report, and return the process status."""
    parser = argparse.ArgumentParser(description="Evaluate the live urban AI scenarios.")
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args(argv)

    canonical_report_path = PROJECT_ROOT / "ai_tests" / "llm_eval_report.json"
    report_path = args.report_path or canonical_report_path
    orchestrator: LlmEvaluationOrchestrator | None = None
    execution_error: str | None = None
    exit_code = 1
    try:
        orchestrator = LlmEvaluationOrchestrator()
        passed, total, _ = orchestrator.execute_evaluation_suite(report_path)
        exit_code = 0 if total > 0 and passed == total else 1
    except Exception as exc:
        execution_error = str(exc)
        LOGGER.exception("LLM evaluation did not complete.")
    finally:
        if orchestrator is None:
            _write_initialization_failure_report(
                report_path,
                execution_error or "Initialization failed.",
            )
            if report_path != canonical_report_path:
                _write_initialization_failure_report(
                    canonical_report_path,
                    execution_error or "Initialization failed.",
                )
        else:
            orchestrator._write_report(
                orchestrator._last_results,
                orchestrator._last_passed,
                report_path,
                execution_error=execution_error,
            )
            if report_path != canonical_report_path:
                orchestrator._write_report(
                    orchestrator._last_results,
                    orchestrator._last_passed,
                    canonical_report_path,
                    execution_error=execution_error,
                )

    if exit_code:
        raise SystemExit(exit_code)
    return exit_code


if __name__ == "__main__":
    main()
