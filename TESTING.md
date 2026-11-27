# 🧪 Automated Testing Guide — Smart City Cluj-Napoca

This document describes the test suite architecture and execution instructions for the Urban IoT platform, ensuring code quality verification prior to deployment in the CI/CD pipeline.

## 🛠️ Testing Stack

In accordance with the unified specifications in `pyproject.toml`, the test suite uses:
- **`pytest`** (>= 8.0) — Primary testing and quality assertion framework.
- **`pytest-json-report`** (>= 1.5) — Automated metric export plugin for unit tests.
- **`pytest-asyncio`** (>= 1.4) — Concurrent execution engine for asynchronous loops.
- **`unittest.mock`** — Isolation patches for simulating network queries.
- **Python RAG Runner** (`ai_tests/run_guardrail_eval.py`) — Verifies hybrid filtering, input sanitization, and numerical grounding.

## 🚀 Execution Instructions

All test commands must be executed from the project root using the `uv` package manager:

```bash
# Run the full offline automated test suite and generate pytest_report.json
uv run pytest

# Run native RAG and guardrail checks (7 checks)
uv run python ai_tests/run_guardrail_eval.py

# Run live Groq evaluation (requires an API key)
uv run python ai_tests/run_llm_eval.py
```

## GitHub Actions: Core CI & LLMOps

Automated validation is split across two workflows:

- **Core CI** (`.github/workflows/ci.yml`) triggers on push and pull requests targeting `main`.
  It synchronizes dependencies with `uv`, runs Ruff, populates a temporary empty database, and
  executes Pytest. The `ai_tests/pytest_report.json` report is uploaded as an artifact; the pipeline
  never commits or pushes reports back to the repository.
- **LLMOps** (`.github/workflows/mlops.yml`) can be triggered manually, runs weekly, and is
  triggered by changes inside `app/ai/` or `ai_tests/`. Python RAG checks run without credentials;
  the live Groq evaluation is skipped when no API key is configured. The `ai_tests/llm_eval_report.json`
  and `ai_tests/rag_guardrails_report.json` reports are uploaded as artifacts without committing them
  to the repository.

## 🏗️ Test Structure & Logic (Offline Verification Ledger)

Automated tests are designed to run in complete isolation, guaranteeing that application logic is valid before simulated telemetry is persisted to the `DATABASE_PATH` database.

### 1. Translation Matrix Integrity (Registry Sync)
The `test_translation_matrix_integrity` function automatically verifies that all 5 supported languages (`RO`, `EN`, `IT`, `ES`, `HU`) contain the exact same localization keys. It accesses the public services exposed by the `TranslationProvider` class, preventing UI crashes caused by runtime `KeyError` exceptions in the browser.

### 2. Startup Bootstrap & Missing-Sensor Handling
`ai_tests/test_app_bootstrap.py` verifies that `SmartCityApplicationEngine.initialize()` creates
the schema and populates a fresh database with the 7 stations, their thresholds, and 84 demo
readings. A second call verifies that custom thresholds are preserved, while UI tests for
Dashboard, Settings, and Analytics confirm that a warning is displayed when no active sensor
exists, without attribute access exceptions.

### 3. Safe & Idempotent Seeder
`ai_tests/test_seed_db.py` covers initial database seeding, preservation of existing settings
and telemetry on subsequent runs, intentional resets via `--force`, and database error propagation
from the library function.

### 4. Session & Offline Integrity
`ai_tests/test_alerts_offline.py` verifies translation key synchronization, reading
`SESSION_SECURITY_TOKEN` from environment variables when Streamlit secrets are unavailable, and
the signed token lifecycle (restoration, expiration, tampering, and logout). It includes eight
parametrized cases with metric values; these currently validate repository instantiation rather
than `stable/critical` classification. Tests in this suite do not call the Groq API.

In total, the baseline offline suite comprises **18 tests**: 11 in `test_alerts_offline.py`, 4 in
`test_app_bootstrap.py`, and 3 in `test_seed_db.py` (alongside the 9 guardrail tests in `test_ai_guardrails.py`). Run locally:

```powershell
uv run pytest
```

Core CI runs without Groq credentials; external LLM evaluations are isolated in the LLMOps
workflow and gated by secret availability.

---

## 🤖 Programmatic LLMOps Evaluation Suite (`run_llm_eval.py`)

To evaluate the non-deterministic generative component without disrupting classical backend unit tests, the platform uses a dedicated evaluator built around Cluj-Napoca urban scenarios.

### Semantic Validation Matrix (7 Scenarios)
The script executes seven Cluj-Napoca urban scenarios, generating with
`qwen/qwen3.8-27b` (with automatic failover to `gemini-3.5-flash`) and compact JSON judging.
The evaluation checks for focal terminology inside the `Assessment` section, normalizes
diacritics, and reports actual provider errors. Requests use adaptive header-driven pacing (x-ratelimit-remaining-tokens / x-ratelimit-reset-tokens, 1.5s fallback) and a compact JSON judge contract (faithfulness: int, passed: bool) and bounded
429/empty-payload retries. The process exits non-zero if any scenario fails; the report is
written to `ai_tests/llm_eval_report.json`, dynamically recording both **Scenario Compliance**
(`success_rate_percentage` from deterministic focal-term matching) and **Average Faithfulness**
(`avg_faithfulness_score`, the arithmetic mean of `1–5` judge scores parsed on `pages/4_About.py` via `LlmEvalSummary`).

---

## 🛡️ Python RAG & Guardrails Evaluation
`ai_tests/run_guardrail_eval.py` executes seven deterministic checks covering metadata and
semantic term filtering, input sanitization, telemetry boundary limits, and numerical grounding
of responses. The current report is written to `ai_tests/rag_guardrails_report.json`.

The LLM evaluator in `ai_tests/run_llm_eval.py` separately executes seven live scenarios via Groq
and records the measured score in `ai_tests/llm_eval_report.json`. This score depends on live
model responses and API availability; prior results are never reused as a passing score.

## Live Verification from the About Page

The **Run Live Architecture Verification** button runs Pytest, the Groq evaluator, and the RAG
checks directly inside the application's Python runtime. Current results are saved to
`st.session_state` and the JSON reports, after which the page reruns to display updated metrics.
Configure one or more Groq keys in Streamlit Secrets for live evaluation; local RAG checks require
no API key.

## 📊 Regression Suite Coverage (99 Tests)

The automated Pytest suite (ai_tests/) validates **99 passing tests** (plus automated documentation sync checks) alongside the **7 deterministic Native Guardrails** (ai_tests/run_guardrail_eval.py):

- **Telemetry & Prompt Propagation:** Verifies default (system_prompt_only=False) and explicit (True) propagation across GroqProvider, GeminiOpenAiProvider, FallbackAiProviderRouter, and Streamlit caching (ai_interface.py).
- **Gemini Fallback Ordering:** Confirms configured_model is always attempted first, followed by deduplicated fallback candidates without dropping any configured model.
- **Structured Router Error Classification:** Tests recovery across typed exceptions (APIStatusError, TimeoutError, ConnectionError), HTTP status codes (429, 500, 502, 503), structured telemetry errors, and ensures numeric sensor readings (e.g., 503 µg/m³) are not misclassified as HTTP errors.
- **Sensor–Value–Unit RAG Grounding:** Validates strict binding between sensor context, numerical values, and measurement units while preserving all 7 Native Guardrails.
