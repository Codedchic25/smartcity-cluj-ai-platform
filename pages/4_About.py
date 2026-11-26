"""Smart City Cluj Platform — Architecture and System Quality Analytics Board.

Provides architectural overview mapping layout specifications and extracts
automated testing metrics from local structural JSON reports via Pydantic contracts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import streamlit as st
from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents.__getitem__(1)
AI_TESTS_DIR: Final[Path] = PROJECT_ROOT / "ai_tests"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.ai_interface import render_full_global_sidebar  # noqa: E402
from app.ai.rag_pipeline import write_guardrail_report  # noqa: E402
from src.schemas import enforce_session_auth  # noqa: E402
from translations import TranslationProvider  # noqa: E402

BadgeColor = Literal["green", "orange", "red"]


class PytestSummary(BaseModel):
    """Validated metric contract for automated Pytest JSON execution reports.

    Attributes:
        status: Human-readable execution status label.
        passed: Count of passed unit and integration assertions.
        total: Total number of executed test assertions.
        color: Semantic badge color indicator.
    """

    status: str = Field(default="Unknown")
    passed: int = Field(default=0, ge=0)
    total: int = Field(default=0, ge=0)
    color: BadgeColor = Field(default="orange")


class GuardrailSummary(BaseModel):
    """Validated metric contract for native Python RAG guardrail reports.

    Attributes:
        status: Human-readable compliance status label.
        successes: Count of passing security and faithfulness guards.
        total: Total number of evaluated prompt scenarios.
        color: Semantic badge color indicator.
    """

    status: str = Field(default="Not Run")
    successes: int = Field(default=0, ge=0)
    total: int = Field(default=0, ge=0)
    color: BadgeColor = Field(default="orange")


class LlmEvalSummary(BaseModel):
    """Validated metric contract for custom LLM context evaluation reports.

    Attributes:
        status: Verification status label for the evaluated model.
        rate: Percentage of scenarios passing factual grounding checks.
        model: Identifier of the deployed LLM inference model.
        color: Semantic badge color indicator.
    """

    status: str = Field(default="Not Run")
    passed: int = Field(default=0, ge=0)
    total: int = Field(default=0, ge=0)
    rate: float = Field(default=0.0, ge=0.0, le=100.0)
    model: str = Field(default="gpt-oss-20b")
    color: BadgeColor = Field(default="orange")


PytestSummary.model_rebuild(_types_namespace={"BadgeColor": BadgeColor})
GuardrailSummary.model_rebuild(_types_namespace={"BadgeColor": BadgeColor})
LlmEvalSummary.model_rebuild(_types_namespace={"BadgeColor": BadgeColor})


class QualityMetricsReader:
    """Parses local test artifact ledgers into validated Pydantic health models."""

    @staticmethod
    def read_pytest_metrics(report_path: Path | None = None) -> PytestSummary:
        """Extract execution metrics from the automated Pytest JSON report.

        Returns:
            PytestSummary: Validated summary of Pytest pass/fail counts.
        """
        report_path = report_path or AI_TESTS_DIR / "pytest_report.json"
        if not report_path.is_file():
            return PytestSummary()

        try:
            with open(report_path, encoding="utf-8") as file:
                data = json.load(file)
            summary = data.get("summary", {})
            passed = int(summary.get("passed", 0))
            total = int(summary.get("total", 0))

            has_failed = int(summary.get("failed", 0)) > 0 or int(summary.get("error", 0)) > 0
            status = "Passed" if total > 0 and passed == total and not has_failed else "Failed"
            color: BadgeColor = "green" if status == "Passed" else "red"

            return PytestSummary(status=status, passed=passed, total=total, color=color)
        except (json.JSONDecodeError, OSError, TypeError, ValidationError, ValueError):
            return PytestSummary()

    @staticmethod
    def read_guardrail_metrics(report_path: Path | None = None) -> GuardrailSummary:
        """Extract compliance metrics from the native Python guardrail report.

        Returns:
            GuardrailSummary: Validated summary of native guardrail evaluations.
        """
        report_path = report_path or AI_TESTS_DIR / "rag_guardrails_report.json"
        if not report_path.is_file():
            return GuardrailSummary()

        try:
            with open(report_path, encoding="utf-8") as file:
                data = json.load(file)
            results = data.get("results", ())
            stats = data.get("stats")
            if isinstance(results, dict) and isinstance(results.get("stats"), dict):
                stats = results["stats"]

            if isinstance(stats, dict):
                successes = int(stats.get("successes", 0))
                failures = int(stats.get("failures", 0))
                errors = int(stats.get("errors", 0))
                total = successes + failures + errors
            elif isinstance(results, list):
                total = len(results)
                successes = sum(
                    1
                    for result in results
                    if isinstance(result, dict) and result.get("success") is True
                )
            else:
                return GuardrailSummary()

            if total == 0:
                return GuardrailSummary()

            status = "Passed" if successes == total else "Review Required"
            color: BadgeColor = "green" if status == "Passed" else "red"

            return GuardrailSummary(status=status, successes=successes, total=total, color=color)
        except (json.JSONDecodeError, OSError, TypeError, ValidationError, ValueError):
            return GuardrailSummary()

    @staticmethod
    def read_llm_eval_metrics(report_path: Path | None = None) -> LlmEvalSummary:
        """Extract model evaluation details from the customized telemetry context script.

        Returns:
            LlmEvalSummary: Validated compliance rate and deployed model name.
        """
        report_path = report_path or AI_TESTS_DIR / "llm_eval_report.json"
        if not report_path.is_file():
            return LlmEvalSummary()

        try:
            with open(report_path, encoding="utf-8") as file:
                data = json.load(file)
            passed = int(data.get("successful_compliance_count", 0))
            total = int(data.get("total_evaluated_scenarios", 0))
            rate = float(data.get("success_rate_percentage", 0.0))
            model = str(data.get("model_deployed", "openai/gpt-oss-20b"))

            status = (
                "Passed"
                if total > 0 and passed == total and rate == 100.0
                else "Failed"
                if total > 0
                else "Not Run"
            )
            color: BadgeColor = "green" if status == "Passed" else "red" if total else "orange"

            return LlmEvalSummary(
                status=status,
                passed=passed,
                total=total,
                rate=rate,
                model=model,
                color=color,
            )
        except (json.JSONDecodeError, OSError, TypeError, ValidationError, ValueError):
            return LlmEvalSummary()


QualitySummary = PytestSummary | GuardrailSummary | LlmEvalSummary


class CheckExecution:
    """Pair a command result with its validated quality summary."""

    def __init__(self, command: dict[str, Any], summary: QualitySummary) -> None:
        """Store a check's command metadata and Pydantic summary."""
        self.command = command
        self.summary = summary


class LiveCheckStrategy(Protocol):
    """Contract for an independently executable quality check."""

    def execute(self) -> CheckExecution:
        """Run one check and return its command result and validated summary."""
        ...


class PytestCheckStrategy:
    """Run the repository's pytest suite in an isolated Python subprocess."""

    def __init__(self, report_path: Path) -> None:
        """Set the JSON report location for this pytest invocation."""
        self._report_path = report_path

    def execute(self) -> CheckExecution:
        """Run pytest and return its measured result."""
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--disable-warnings",
            "--json-report-file",
            str(self._report_path),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                check=False,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return CheckExecution(
                command={
                    "name": "Pytest",
                    "returncode": 1,
                    "passed": False,
                    "error": str(exc),
                    "output": "",
                },
                summary=PytestSummary(status="Failed", color="red"),
            )

        output = f"{completed.stdout or ''}{completed.stderr or ''}"[-6000:]
        summary = QualityMetricsReader.read_pytest_metrics(self._report_path)
        passed = completed.returncode == 0 and summary.status == "Passed"
        if not passed:
            summary = PytestSummary(
                status="Failed",
                passed=summary.passed,
                total=summary.total,
                color="red",
            )

        return CheckExecution(
            command={
                "name": "Pytest",
                "returncode": completed.returncode,
                "passed": passed,
                "error": "" if passed else f"Pytest exited with code {completed.returncode}.",
                "output": output,
            },
            summary=summary,
        )


class GuardrailCheckStrategy:
    """Run the native RAG and grounding assertions without calling an LLM."""

    def __init__(self, report_path: Path) -> None:
        """Set the JSON report location for native guardrail results."""
        self._report_path = report_path

    def execute(self) -> CheckExecution:
        """Run the in-process guardrail suite and return its measured result."""
        try:
            report = write_guardrail_report(self._report_path)
            summary = QualityMetricsReader.read_guardrail_metrics(self._report_path)
            passed = summary.status == "Passed" and summary.successes == summary.total == 7
            error = (
                ""
                if passed
                else f"{summary.successes}/{summary.total} native guardrails passed."
            )
            return CheckExecution(
                command={
                    "name": "RAG & guardrails",
                    "returncode": 0 if passed else 1,
                    "passed": passed,
                    "error": error,
                    "output": json.dumps(report.get("results", []), ensure_ascii=False),
                },
                summary=summary,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return CheckExecution(
                command={
                    "name": "RAG & guardrails",
                    "returncode": 1,
                    "passed": False,
                    "error": str(exc),
                    "output": "",
                },
                summary=GuardrailSummary(status="Failed", color="red"),
            )


class LiveLlmCheckStrategy:
    """Run the explicitly requested live Groq scenario evaluation."""

    def __init__(self, report_path: Path) -> None:
        """Set the JSON report location for live model scenarios."""
        self._report_path = report_path

    def execute(self) -> CheckExecution:
        """Run the API-backed scenarios and return their measured result."""
        try:
            from ai_tests.run_llm_eval import LlmEvaluationOrchestrator

            passed_count, total, _ = LlmEvaluationOrchestrator().execute_evaluation_suite(
                self._report_path
            )
            summary = QualityMetricsReader.read_llm_eval_metrics(self._report_path)
            passed = total == 7 and passed_count == total and summary.rate == 100.0
            return CheckExecution(
                command={
                    "name": "Live Groq LLM evaluation",
                    "returncode": 0 if passed else 1,
                    "passed": passed,
                    "error": "" if passed else f"{passed_count}/{total} scenarios passed.",
                    "output": f"Live Groq evaluation: {passed_count}/{total} scenarios passed.",
                },
                summary=summary,
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return CheckExecution(
                command={
                    "name": "Live Groq LLM evaluation",
                    "returncode": 1,
                    "passed": False,
                    "error": str(exc),
                    "output": "",
                },
                summary=LlmEvalSummary(
                    status="Failed",
                    model="openai/gpt-oss-20b",
                    color="red",
                ),
            )


class LiveQaExecutor:
    """Execute selected quality-check strategies concurrently."""

    def __init__(self, strategies: dict[str, LiveCheckStrategy]) -> None:
        """Store the selected independent check strategies."""
        self._strategies = strategies

    def execute(self) -> dict[str, CheckExecution]:
        """Submit every selected check to a worker and collect its result."""
        with ThreadPoolExecutor(
            max_workers=len(self._strategies),
            thread_name_prefix="live-qa",
        ) as executor:
            futures = {
                name: executor.submit(strategy.execute)
                for name, strategy in self._strategies.items()
            }
            return {name: future.result() for name, future in futures.items()}


class AboutQualityPage:
    """Orchestrates the System Quality Control Center and Architecture Overview UI."""

    def __init__(self) -> None:
        """Initialize the quality board reader dependencies."""
        self._reader: Final[QualityMetricsReader] = QualityMetricsReader()

    def _execute_live_qa_suite(self, include_live_llm: bool = False) -> None:
        """Run fast local checks and optionally include the live Groq evaluation."""
        AI_TESTS_DIR.mkdir(parents=True, exist_ok=True)

        strategies: dict[str, LiveCheckStrategy] = {
            "pytest": PytestCheckStrategy(AI_TESTS_DIR / "pytest_report.json"),
            "guardrails": GuardrailCheckStrategy(AI_TESTS_DIR / "rag_guardrails_report.json"),
        }
        llm_report = AI_TESTS_DIR / "llm_eval_report.json"
        if include_live_llm:
            strategies["llm"] = LiveLlmCheckStrategy(llm_report)

        with st.spinner("Running parallel local quality checks..."):
            executions = LiveQaExecutor(strategies).execute()
            command_results = {
                name: execution.command for name, execution in executions.items()
            }
            summaries = {
                name: execution.summary.model_dump()
                for name, execution in executions.items()
            }

            if include_live_llm:
                llm_data = LlmEvalSummary.model_validate(summaries["llm"])
            elif llm_report.is_file():
                llm_data = QualityMetricsReader.read_llm_eval_metrics(llm_report)
                command_results["llm"] = {
                    "name": "LLM evaluation (cached)",
                    "returncode": 0,
                    "passed": True,
                    "skipped": True,
                    "error": "",
                    "output": (
                        "Reused the most recent LLM evaluation report; "
                        "live evaluation skipped."
                    ),
                }
            else:
                llm_data = LlmEvalSummary(
                    status="Cached / Skipped (Fast Mode)",
                    model="openai/gpt-oss-20b",
                )
                command_results["llm"] = {
                    "name": "LLM evaluation",
                    "returncode": 0,
                    "passed": True,
                    "skipped": True,
                    "error": "",
                    "output": "Cached / Skipped (Fast Mode): no previous evaluation report exists.",
                }
            summaries["llm"] = llm_data.model_dump()

        required_checks = ("pytest", "guardrails", "llm") if include_live_llm else (
            "pytest",
            "guardrails",
        )
        all_passed = all(command_results[name]["passed"] for name in required_checks)
        st.session_state["live_qa_execution"] = {
            "passed": all_passed,
            "include_live_llm": include_live_llm,
            "commands": command_results,
            "summaries": summaries,
        }
        st.rerun()

    @staticmethod
    def _ensure_guardrail_report() -> None:
        """Create the native guardrail report for clean deployments."""
        report_path = AI_TESTS_DIR / "rag_guardrails_report.json"
        if not report_path.is_file():
            write_guardrail_report(report_path)

    def _render_quality_control_center(self) -> None:
        """Render the centralized unified automated quality control grid frame."""
        try:
            self._ensure_guardrail_report()
        except OSError as exc:
            st.error(f"Unable to create the RAG guardrail report: {exc}")

        st.markdown("### 🛡️ Systemic Quality & AI Compliance Control Center")
        st.write(
            "Real-time engineering telemetry parsed cleanly from automated "
            "integration suites execution gates metrics."
        )
        st.markdown("<br>", unsafe_allow_html=True)

        controls = st.columns((2, 1))
        with controls[0]:
            include_live_llm: Final[bool] = st.checkbox(
                "Include Live Groq LLM Eval (uses API tokens)",
                key="include_live_groq_llm_eval",
            )
        with controls[1]:
            trigger_tests: Final[bool] = st.button(
                "⚡ Run Live Architecture Verification",
                key="btn_trigger_live_qa_suite",
                width="stretch",
                type="primary",
            )
        if trigger_tests:
            self._execute_live_qa_suite(include_live_llm=include_live_llm)

        st.divider()

        live_execution = st.session_state.get("live_qa_execution")
        if isinstance(live_execution, dict):
            pytest_data = PytestSummary.model_validate(live_execution["summaries"]["pytest"])
            guardrail_summary = GuardrailSummary.model_validate(
                live_execution["summaries"]["guardrails"]
            )
            llm_data = LlmEvalSummary.model_validate(live_execution["summaries"]["llm"])

            if live_execution["passed"]:
                if live_execution.get("include_live_llm", False):
                    st.success(
                        "All checks passed: Pytest, RAG guardrails 7/7, "
                        f"and live Groq evaluation {llm_data.rate:.1f}%."
                    )
                else:
                    st.success(
                        "Fast local checks passed: Pytest and RAG guardrails. "
                        f"LLM evaluation: {llm_data.status}."
                    )
            else:
                st.error(
                    "Live verification did not meet all required results. "
                    "Review each command's exit status and output below."
                )
            with st.expander("Live verification command output"):
                for command_result in live_execution["commands"].values():
                    outcome = (
                        "Skipped"
                        if command_result.get("skipped", False)
                        else "Passed"
                        if command_result["passed"]
                        else "Failed"
                    )
                    st.markdown(f"**{command_result['name']}: {outcome}**")
                    if command_result["error"]:
                        st.caption(command_result["error"])
                    if command_result["output"]:
                        st.code(command_result["output"])
        else:
            pytest_data = self._reader.read_pytest_metrics()
            guardrail_summary = self._reader.read_guardrail_metrics()
            llm_data = self._reader.read_llm_eval_metrics()

        q_col1, q_col2, q_col3 = st.columns(3)

        with q_col1:
            st.markdown("**Backend Regression Tests**")
            st.markdown(f"**Status:** :{pytest_data.color}[{pytest_data.status}]")
            st.metric(
                label="Pytest Assertions Passed",
                value=f"{pytest_data.passed} / {pytest_data.total}",
            )

        with q_col2:
            st.markdown("**RAG & Guardrails**")
            st.markdown(f"**Status:** :{guardrail_summary.color}[{guardrail_summary.status}]")
            st.metric(
                label="Native Guardrails Passed",
                value=f"{guardrail_summary.successes} / {guardrail_summary.total}",
            )

        with q_col3:
            st.markdown("**Context Evaluation Guard**")
            st.markdown(f"**Status:** :{llm_data.color}[{llm_data.status}]")
            st.metric(label="LLM Scenario Compliance", value=f"{llm_data.rate:.1f}%")
            st.caption(
                f"Scenarios: {llm_data.passed}/{llm_data.total} · Engine: `{llm_data.model}`"
            )

        st.divider()

    def _render_architecture_blueprint(self) -> None:
        """Render the architectural summary profile of the Smart City platform."""
        st.markdown("### 🏛️ Engineering Portfolio Profile Blueprint")
        st.markdown(
            "- **Primary Architecture Language:** Python 3.12 (Strict PEP 8 & Pydantic v2)\n"
            "- **Interactive Workspace Presentation:** Multi-Page Streamlit Framework\n"
            "- **Relational Storage Core Persistence:** Localized ACID SQLite + Alembic\n"
            "- **Generative Language Model Orchestrator:** Groq Cloud API Gateway Interface\n"
            "- **AI Evaluation & Guardrails:** Native Python RAG Validation & Pytest Reports\n"
            "- **Package and Dependency Manager Execution:** Astral `uv` Environment Ecosystem"
        )

    def display(self) -> None:
        """Execute structural execution flow mapping for the localized About page."""
        st.set_page_config(
            page_title="Smart City Cluj - About & Quality",
            layout="wide",
            page_icon="ℹ️",
        )
        enforce_session_auth()

        active_lang: Final[str] = st.session_state.get("lang", "RO")
        provider: Final[TranslationProvider] = TranslationProvider(default_language=active_lang)
        translations: Final[dict[str, str]] = provider.get_all_translations()

        _ = render_full_global_sidebar(translations)

        st.markdown(
            "<h1>ℹ️ System Architecture & Engineering Specifications</h1>",
            unsafe_allow_html=True,
        )
        st.write("Blueprints covering decoupled designs and automated verification.")
        st.divider()

        self._render_quality_control_center()
        self._render_architecture_blueprint()


if __name__ == "__main__":
    page = AboutQualityPage()
    page.display()
