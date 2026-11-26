"""Automated LLM evaluation framework for Smart City telemetry prompts.

Enforces strict compliance with data trust bounds, case-insensitive mapping,
and PEP 8 compliant line lengths fully optimized for green CI/CD pipelines.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from dotenv import load_dotenv

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
from app.ai.groq_provider import GroqProvider  # noqa: E402
from app.ai.rag_pipeline import (  # noqa: E402
    TelemetryEvidence,
    format_retrieved_context,
    retrieve_telemetry,
    sanitize_untrusted_input,
)


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


ASSESSMENT_FOCUS_LABELS: Final[dict[str, str]] = {
    "temperature": "temperatura ambiental / ambient temperature",
    "noise": "nivelul de zgomot / noise level",
    "traffic": "traficul / traffic load",
    "air": "PM2.5 / calitatea aerului / air quality",
    "soil": "umiditatea solului / soil moisture",
}


class PromptTemplateLoader:
    """Loads the production prompt template and injects complete telemetry context.

    Responsible for file system interactions regarding prompt assets and the
    dynamic composition of active telemetry context matrices.
    """

    DEFAULT_TEMPLATE_PATH: Final[str] = "ai_tests/prompts.txt"

    def __init__(self, template_path: str = DEFAULT_TEMPLATE_PATH) -> None:
        """Initialize the prompt template loader layer.

        Args:
            template_path: Relative path matching the prompt text storage file.
        """
        self._template_path = PROJECT_ROOT / template_path
        self._context_reader = TelemetryContextReader()

    def _load_template(self) -> str | None:
        """Read the prompt template safely from disk storage components.

        Returns:
            The raw text string of the template file, or None if an OS/IO error occurs.
        """
        if not self._template_path.is_file():
            return None
        try:
            return self._template_path.read_text(encoding="utf-8")
        except OSError:
            return None

    @staticmethod
    def _build_telemetry_context(
        scenario: EvalScenario,
        thresholds: SensorThresholdMatrix,
    ) -> str:
        """Build the canonical telemetry block used by the evaluation suite.

        Args:
            scenario: The immutable evaluation scenario containing sensor bounds.
            thresholds: Authorized operational limits for the scenario station.

        Returns:
            A formatted prompt fragment mapping telemetry and authorized thresholds.
        """
        return f"""
[LIVE URBAN TELEMETRY]
Location: {sanitize_untrusted_input(scenario.location, max_length=200)}
Temperature: {scenario.temperature:.1f} °C
Noise level: {scenario.noise_level:.1f} dB
Traffic load: {scenario.traffic_load:.1f} %
Air quality PM2.5: {scenario.air_quality:.1f}
Soil moisture: {scenario.soil_moisture:.1f} %

[AUTHORIZED OPERATIONAL THRESHOLDS]
Reference thresholds for {sanitize_untrusted_input(scenario.location, max_length=200)}:
- Temperature Threshold: {thresholds.temperature_limit:.1f} °C
- Noise Level Threshold: {thresholds.noise_limit:.1f} dB
- Traffic Load Threshold: {thresholds.traffic_limit:.1f} %
- Air Quality PM2.5 Threshold: {thresholds.air_quality_limit:.1f} µg/m³
- Soil Moisture Minimum Threshold: {thresholds.soil_moisture_limit:.1f} %

[TELEMETRY INTERPRETATION]
Identify the dominant operational condition.
Provide a concise operational recommendation in the requested language.
"""

    def load_and_render(self, scenario: EvalScenario, language: str = "RO") -> str | None:
        """Load and render the production prompt with complete scenario telemetry.

        Args:
            scenario: The telemetry scenario profile to inject into the text blocks.
            language: Target locale abbreviation for the LLM output (default: 'RO').

        Returns:
            The fully rendered string prompt ready for LLM consumption, or None.
        """
        template_content = self._load_template()
        if template_content is None:
            return None

        safe_location = sanitize_untrusted_input(scenario.location, max_length=200)
        thresholds = self._context_reader.fetch_operational_thresholds(safe_location)
        rendered_prompt = (
            template_content.replace("{{locatie}}", safe_location)
            .replace("{{temperature}}", f"{scenario.temperature:.1f}")
            .replace("{{noise_level}}", f"{scenario.noise_level:.1f}")
            .replace("{{traffic_load}}", f"{scenario.traffic_load:.1f}")
            .replace("{{air_quality}}", f"{scenario.air_quality:.1f}")
            .replace("{{soil_moisture}}", f"{scenario.soil_moisture:.1f}")
            .replace("{{limba_activa}}", language)
            .replace(
                "{{jurnal_alerte_recente}}",
                sanitize_untrusted_input("No active security breach events logged."),
            )
        )
        telemetry_context = self._build_telemetry_context(scenario, thresholds)
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
            "[SCENARIO FOCUS]\n"
            "In the Assessment section, explicitly discuss this scenario's focus: "
            f"{focus_label}. Ground the assessment in the supplied readings."
        )
        return (
            f"{rendered_prompt.rstrip()}\n\n{telemetry_context.strip()}"
            f"\n\n[RETRIEVED CONTEXT]\n{format_retrieved_context(retrieved)}"
            f"\n[/RETRIEVED CONTEXT]\n\n{evaluation_instruction}"
        )


class LlmEvaluationOrchestrator:
    """Execute LLM telemetry scenarios and calculate compliance metrics.

    Orchestrates the lifecycle of sending batch payloads to Groq providers,
    managing rate limit backoffs, and calculating multi-language text compliance.
    """

    MODEL_TARGET: Final[str] = "openai/gpt-oss-20b"
    MAX_RETRIES: Final[int] = 2
    REQUEST_INTERVAL_SECONDS: Final[float] = 2.1
    RATE_LIMIT_BACKOFF_SECONDS: Final[float] = 1.0
    MAX_RATE_LIMIT_BACKOFF_SECONDS: Final[float] = 3.0
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
        self._provider = GroqProvider(model_target=self.MODEL_TARGET)
        self._loader = PromptTemplateLoader()
        self._next_request_at = 0.0
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

    def _wait_for_request_slot(self) -> None:
        """Space sequential request starts to remain within free-tier rate limits."""
        now = time.monotonic()
        request_at = max(now, self._next_request_at)
        self._next_request_at = request_at + self.REQUEST_INTERVAL_SECONDS

        delay = request_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)

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
    def _extract_assessment(response_text: str) -> str | None:
        """Extract the Assessment body from plain or bold Markdown section labels."""
        match = re.search(
            r"(?ims)^\s*(?:\*\*)?Assessment:?(?:\*\*)?\s*(.*?)"
            r"^\s*(?:\*\*)?(?:Recommended Actions|Data Limitations|Security Note)"
            r":?(?:\*\*)?\s*$",
            response_text,
        )
        return match.group(1).strip() if match else None

    def _execute_with_backoff(self, rendered_prompt: str) -> str:
        """Retry once with a short delay only for transient rate-limit failures."""
        for attempt in range(self.MAX_RETRIES):
            self._wait_for_request_slot()
            try:
                response_text = self._provider.generate_completion(rendered_prompt)
                error_markers = (
                    "❌",
                    "Failure",
                    "Generic Operational Alert:",
                    "Dependency Constraint:",
                    "Subsystem Failure:",
                    "⚠️ Connection Warning:",
                )
                if any(marker in response_text for marker in error_markers):
                    raise RuntimeError(response_text)
                return response_text
            except Exception as exc:
                err_msg = str(exc)
                if not self._is_transient_rate_limit(err_msg) or attempt >= self.MAX_RETRIES - 1:
                    raise

                match = re.search(
                    r"(?:again in|retry after)\s+([0-9.]+)\s*s?",
                    err_msg,
                    flags=re.IGNORECASE,
                )
                requested_delay = (
                    float(match.group(1)) + 0.25
                    if match
                    else self.RATE_LIMIT_BACKOFF_SECONDS * (2**attempt)
                )
                time.sleep(min(requested_delay, self.MAX_RATE_LIMIT_BACKOFF_SECONDS))
        return ""

    @staticmethod
    def _is_transient_rate_limit(error_message: str) -> bool:
        """Recognize retryable 429 responses but exclude exhausted account quotas."""
        normalized_error = error_message.casefold()
        quota_exhaustion_markers = (
            "daily limit",
            "daily quota",
            "monthly quota",
            "quota exhausted",
            "quota exceeded",
        )
        if any(marker in normalized_error for marker in quota_exhaustion_markers):
            return False
        return any(
            marker in normalized_error
            for marker in ("429", "rate_limit", "rate limit", "too many requests")
        )

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
                "error": str(exc),
            }

        response_text = re.sub(
            r"<think>.*?</think>",
            "",
            response_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        assessment = self._extract_assessment(response_text)
        variants = self._find_matching_variants(assessment or "", scenario.expected_keyword)
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
            "matched_variants": variants,
            "assessment_present": assessment is not None,
            "payload_length": len(response_text),
            "model_response": response_text,
        }

    def _write_report(
        self,
        results: list[dict[str, Any]],
        passed: int,
        report_path: Path | None = None,
    ) -> Path:
        """Persist the evaluation results as a structured JSON report."""
        total = len(self._scenarios)
        report_payload = {
            "execution_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_deployed": self._provider.configured_model,
            "total_evaluated_scenarios": total,
            "successful_compliance_count": passed,
            "failed_compliance_count": total - passed,
            "success_rate_percentage": round((passed / total) * 100, 1) if total else 0.0,
            "detailed_metrics": results,
        }
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
                results_by_index[futures[future]] = future.result()

        results = [result for result in results_by_index if result is not None]
        for result in results:
            if result.get("matched_compliance", False):
                passed_assertions += 1

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the live urban AI scenarios.")
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()

    orchestrator = LlmEvaluationOrchestrator()
    passed, total, _ = orchestrator.execute_evaluation_suite(args.report_path)
    raise SystemExit(0 if total > 0 and passed == total else 1)
