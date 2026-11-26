# 📝 Changelog: Smart City Cluj IoT Core

All notable changes to this project are documented in this file.
This repository adheres to Semantic Versioning (`MAJOR.MINOR.PATCH`).

---

## [Unreleased]

### ✨ Added
- **Startup SQLite Bootstrap:** After schema creation, `main.py` invokes the idempotent
  seeder to populate new databases with the 7 Cluj-Napoca monitoring stations, default
  thresholds, and demo telemetry without overwriting existing operational records.
- **Station Selection Safeguards:** Dashboard, Settings, and Analytics display a
  warning when no active sensor is selected instead of raising `AttributeError`.
- **Bootstrap Regression Suite:** `ai_tests/test_app_bootstrap.py` verifies empty
  database initialization, settings persistence across restarts, and page behavior when
  selection is missing. The current report is generated on every Pytest run.
- **Multilingual LLM Alignment:** `GroqProvider` generates response bodies in the
  active UI language (`RO`, `EN`, `IT`, `ES`, `HU`) and formats the Assessment,
  Recommended Actions, and Data Limitations sections with bold Markdown labels on
  separate lines.
- **Dashboard Cold-Start Fallback:** During normal startup, `main.py` idempotently
  populates an empty database; if Dashboard finds no stations, it displays manual
  seeding instructions, and when fewer than two readings exist, it falls back to
  synthetic history.

### 🔧 Improved
- **`system_prompt_only` & Telemetry Propagation:** Propagated `system_prompt_only: bool = False`
  end-to-end across `UrbanAiAssistantComponent`, `_cached_ai_completion`,
  `FallbackAiProviderRouter`, `GroqProvider`, and `GeminiOpenAiProvider` down to the
  API payload and session telemetry, preserving default (`False`) behavior for
  existing callers.
- **Configured Model Priority in Gemini Fallback:** Updated
  `GeminiOpenAiProvider._candidate_models` so `configured_model` is always attempted
  first, followed by all deduplicated models in `FALLBACK_MODELS` without dropping any
  fallback candidate.
- **Robust Router Error Classification:** Replaced fragile prefix matching in
  `FallbackAiProviderRouter` and `ai_interface.py` with `is_provider_failure`,
  evaluating typed exceptions (`APIStatusError`, `TimeoutError`, `ConnectionError`),
  HTTP status codes (`429`, `500`, `502`, `503`, `504`), structured telemetry errors,
  and normalized failure messages without misclassifying numeric sensor readings
  (e.g., `503 µg/m³`).
- **Strict Sensor–Value–Unit RAG Grounding:** Enhanced `rag_pipeline.py` and
  `ai_interface.py` to bind numerical values strictly to their corresponding sensor
  identifier, location, and measurement unit while preserving all 7 deterministic
  Native Guardrails.
- **Live LLM Evaluator Free-Tier Optimization:** Switched the LLM judge in `ai_tests/run_llm_eval.py` to the fast non-reasoning `llama-3.1-8b-instant` model to eliminate `<think>` token overhead, and added `max_scenarios` / `--scenarios` sampling (`3` scenarios in `pages/4_About.py` UI, full `7` scenarios in CLI/CI).
- **Gemini Correction Retry Decision:** Evaluated and documented skipping generic
  iterative correction retries on Gemini to avoid doubling input token cost and adding
  latency, relying instead on multi-model fallback and deterministic guardrails.
- **Expanded Regression Suite:** Increased the automated Pytest suite from **72** to
  **97 passing tests** (`ai_tests/test_ai_guardrails.py`).

---

## [v2.3.0] — 2026-10-06

### ✨ Added
- **Context-Aware AI Assistant with Direct Translations:** Implemented an alignment
  mechanism connecting the intelligent assistant with the application's native
  dictionaries by directly querying the `_REGISTRY` store in `translations.py`. This
  ensures seamless localization across RO, EN, IT, ES, and HU while eliminating
  `AttributeError` exceptions.
- **Programmatic Python Evaluation Framework (`run_llm_eval.py`):** Built an isolated,
  native Python automated evaluation suite that benchmarks the cognitive assistant
  across 7 critical urban scenarios in Cluj-Napoca, exporting structured metrics to
  `llm_eval_report.json`.
- **Cryptographic Authentication in Isolated Environments:** Hardened
  `AuthenticationService` with a strict `RuntimeError` guard that blocks platform
  startup if credentials are missing from `.env`. Password verification combines
  cryptographic hashing with constant-time comparison via `hmac.compare_digest` to
  mitigate brute-force and timing side-channel attacks.
- **Orchestration with Ruff & UV:** Standardized code quality across the entire
  repository to PEP 8 conventions by integrating the `ruff` linter orchestrated via the
  `uv` package manager.

### 🐛 Fixed
- **Test Suite Restoration:** Separated test modules cleanly under `ai_tests/`.
  Refactored `test_alerts_offline.py` to remove duplicate LLM stubs and query atomic
  methods exposed directly by `CityRepository` in `main.py`, restoring clean terminal
  test execution.
- **Database Path Alignment:** Fixed the `DatabaseConfig` constructor in `seed_db.py`
  by appending `.resolve()`. This eliminated the unused variable reported by Ruff
  (F841) and ensured the seeding script writes to the exact SQLite database file read
  by Streamlit, preventing duplicate database files on disk.
- **Windows File Lock Bypass (EBUSY):** Documented the PowerShell procedure for
  terminating orphaned Python processes (`Stop-Process -Name "python" -Force`),
  resolving OS file-lock conflicts when deleting or replacing `app.db`.
- **PM2.5 Measurement Unit Alignment:** Updated explanatory strings and metrics across
  `1_Dashboard.py` and `3_Analytics.py`, replacing the incorrect `ppm` unit with the
  accurate standard **`µg/m³`** in alignment with the translation dictionary.
