# 🎓 Technical Study & Architecture Guide — Smart City Cluj-Napoca

## 0. Executive Summary (30-Second Elevator Pitch)

**Smart City Cluj-Napoca** is a modular urban intelligence, IoT telemetry, and AI-assisted analytics platform built in **Python 3.12** and **Streamlit**. The application combines deterministic data processing (**Pydantic v2** validation, **SQLite** persistence, **Pandas/NumPy** statistical analytics, cooldown-throttled audit logging) with Groq-first operational recommendations and Gemini fallback, cryptographic session security (**Argon2id password hashing + HMAC-SHA256 session-token signing**), and native Python verification checks for RAG and LLM workflows.

### End-to-End Execution Pipeline

    [Login: Argon2id password verification + hmac.compare_digest for username comparison]
           │
           ▼
    [HMAC-SHA256 Signed Session (24h TTL) in session_state & URL query_param]
           │
           ▼
    [Idempotent Bootstrap: CREATE TABLE IF NOT EXISTS + seed_db (only if empty)]
           │
           ▼
    [Pydantic v2 Validation (Sensor physical bounds, GPS, operational thresholds)]
           │
           ├──► [Deterministic Analytics: Pearson Correlations, Linear Regression, CSV/XLSX Export]
           │
           └──► [Context Builder: Telemetry + Thresholds + Station Trend + Latest 5 alerts]
                       │
                       ▼
                [FallbackAiProviderRouter: Groq -> Gemini + Grounding -> UI Recommendations]

---

## 1. System Architecture & Entry Points

### Startup & Multipage Navigation
The web entry point is `main.py`, launched locally via `uv run streamlit run main.py` or in production on **Streamlit Community Cloud**.

During initialization, `main.py`:
1. Loads environment variables from `.env` (if present locally) and resolves configuration via Pydantic `BaseSettings`.
2. Instantiates `CityRepository`, `AuthenticationService`, and `SmartCityApplicationEngine`.
3. Creates the `sensors`, `city_stats`, and `settings` tables if missing and executes the idempotent seeder to automatically populate the **7 Cluj-Napoca monitoring stations** only when the database is empty.
4. Enforces the authentication gateway; authenticated operators gain access to the 4 modules inside `pages/`:
   * **`1_Dashboard.py`:** Real-time KPIs, interactive geospatial map, trend charts, security alert feed, and the AI assistant.
   * **`2_Settings.py`:** Per-station alert threshold configuration (atomic `UPSERT` operations) and data export (`CSV` / `XLSX`).
   * **`3_Analytics.py`:** Pearson correlation matrix, urban balance radar profile, joint distributions (box + violin), and `NumPy` linear regression.
   * **`4_About.py`:** Architectural specifications and the **Systemic Quality & AI Compliance Control Center**.

Each page validates session integrity via `enforce_session_auth()` and implements defensive *null-guards*: if the active station cannot be resolved from an empty database, the page renders a controlled `st.warning` without raising `AttributeError` exceptions.
Normal `main.py` startup idempotently seeds an empty database. If the Dashboard is
opened with no stations, it shows instructions to run `uv run python seed_db.py`;
when a station has fewer than two stored readings, it uses synthetic history.

### Separation of Concerns
* **`main.py` & `pages/`:** Presentation layer (UI) and session orchestration.
* **`src/`:** Pydantic v2 data contracts (`schemas.py`), session token management (`session_auth.py`), and the local alerting engine (`utils/local_alerts.py`).
* **`app/ai/`:** AI interface (`ai_interface.py`), RAG/grounding pipeline (`rag_pipeline.py`), Groq client (`groq_provider.py`), and Groq-first Gemini fallback (`multi_provider.py`).
* **`app/database/` & `migrations/`:** SQLAlchemy ORM models, async session setup, and Alembic migrations prepared for external database scaling.
* **`ai_tests/`:** Pytest regression suite and native Python evaluators for RAG guardrails and LLM scenarios.

> **💡 Architectural Decision (Trade-off):** Maintaining a modular architecture (decoupling authentication, alerting, and LLM client logic from Streamlit UI code) enables fast unit testing with `Pytest` without spinning up a web server and simplifies future replacement of the UI or AI provider.

---

## 2. Security & Session Management

Authentication is engineered across two independent layers (Defense-in-Depth):

1. **Credential Verification (`AuthenticationService`):**
   * Reads the configured username (`PLATFORM_ADMIN_USER`) and password hash (`PLATFORM_ADMIN_PASS`) from secure configuration.
   * Verifies the submitted password against the configured Argon2id encoded password hash using PasswordHasher.verify().
   * Zero plaintext passwords or hardcoded hashes exist anywhere in the source code.

2. **Cryptographically Signed Session Tokens (`src/session_auth.py`):**
   * Upon authentication, the system issues a token composed of a JSON payload (`sub` + `exp`) signed with **HMAC-SHA256** using `SESSION_SECURITY_TOKEN` and a **24-hour TTL**.
   * The token is stored in both `st.session_state` and the `?session_token=...` URL query parameter. Across browser refreshes (`F5`) or multipage navigation, signature validity and expiration are automatically re-verified.
   * Expired or tampered tokens are immediately rejected, and clicking *Logout* clears both session memory and the URL query parameter.

> **💡 Architectural Decision (Trade-off):** Because Streamlit re-executes scripts on every UI interaction and can lose in-memory state on browser refresh, carrying a 24-hour HMAC-SHA256 signed token in `query_params` provides true session persistence without exposing administrator credentials.

---

## 3. Data Layer, Validation & Idempotent Seeding

### Runtime Data Contracts (`Pydantic v2`)
* **`SensorNode` & `TelemetryReading`:** Enforce strict physical bounds (e.g., temperature between `-50°C` and `65°C`, traffic load `0–100%`, valid GPS coordinates) before telemetry reaches charts or the LLM prompt.
* **`SensorThreshold`:** Validates administrator-configured thresholds in `2_Settings.py` prior to persisting them in SQLite.
* **QA Report Summaries (`PytestSummary`, `GuardrailSummary`, `LlmEvalSummary`):** Defensively parse JSON test reports for display on `4_About.py`.

### Safe & Idempotent Seeding (`seed_db.py` / `app/database/seeder.py`)
* In its initial design, the seeding script unconditionally reset tables, risking overwriting custom operator thresholds.
* **Implemented Solution:** On default execution (`uv run python seed_db.py`) or `main.py` startup, the seeder checks whether `sensors`, `settings`, and `city_stats` are completely empty. If data already exists, execution is skipped (*safe skip*), preserving operator customizations.
* A full reset back to demo values (7 stations × 12 readings = 84 telemetry points) occurs only intentionally via the CLI **`--force`** flag.

> **💡 Architectural Decision (Trade-off):** Using SQLite with idempotent auto-bootstrap eliminates the need for an external database server in ephemeral environments (Docker / Streamlit Community Cloud), while the SQLAlchemy + Alembic models in `app/database/` keep the migration path ready for PostgreSQL (`asyncpg`).

---

## 4. Multi-Provider AI Integration & Deterministic Alert Engine

### AI Resilience & Multi-Provider Fallback (`app/ai/multi_provider.py`)
1. **Context Construction (`ai_interface.py`):** `PromptRegistryLoader` validates and caches only `app/ai/prompts.yaml` through the immutable Pydantic v2 `PromptTemplateConfig`. It sends system instructions, output format, and anti-hallucination rules separately from the rendered user template containing the selected station, current telemetry, configured thresholds, active UI language (`RO`, `EN`, `IT`, `ES`, `HU`), recent station trend, retrieved SOP passages, and latest events from `security_alerts.log`.
2. **Provider Routing (`FallbackAiProviderRouter`):** `GroqProvider` prioritizes `qwen/qwen3.8-27b` (with `openai/gpt-oss-20b` and `openai/gpt-oss-120b` fallbacks) and rotates `GROQ_API_KEY*` credentials with SDK retries disabled. If Groq exhausts its keys/models or returns a 429/TPM error, `GeminiOpenAiProvider` fails over to the native `google-genai` SDK (`GEMINI_API_KEY`) using `gemini-3.5-flash` followed by `gemini-3.5-flash-lite`, `gemini-3.6-flash`, and `gemini-3.8-flash` with strict Pydantic `response_schema` decoding and numeric grounding validation.
3. **Recommendation Cache:** `UrbanAiAssistantComponent` uses `@st.cache_data(ttl=300, show_spinner=False)` keyed by the rendered prompt, telemetry values, and active language. The request remains exclusive to the recommendation button; matching inputs reuse the cached answer for five minutes and reduce token consumption.
4. **Response Sanitization:** Strips internal reasoning blocks (`<think>...</think>`) before rendering recommendations and stores the result in `st.session_state` per station.
5. **Language & Markdown Alignment:** Aligns response body text with the active language (`RO`, `EN`, `IT`, `ES`, or `HU`) and places the bold English Assessment, Recommended Actions, and Data Limitations labels on separate lines as Markdown sections.
6. **Recent Telemetry Trend:** `TelemetryContextReader` retrieves up to five latest rows for the selected station, orders them chronologically for comparison, and adds a compact `[RECENT TELEMETRY TREND]` summary with per-metric deltas and rising/stable/falling direction. With fewer than two valid readings, the prompt receives a safe no-trend fallback. The prompt explicitly treats this block as historical evidence only.
7. **Deterministic Grounding:** After the model response, the assistant checks for the selected station, current metric values, and configured threshold breaches. It appends only omitted facts from the same telemetry and station settings, requiring no extra provider request or tokens. Unit tests cover station-specific trends and missing-fact completion in `ai_tests/test_ai_guardrails.py`.
8. **Hybrid SOP Retrieval:** `SopKnowledgeRetriever` indexes `docs/cluj_urban_sop.md` with `TfidfVectorizer`, ranks procedures with `cosine_similarity`, and adds at most two passages as `[AUTHORIZED URBAN SOP CONTEXT]`. SOP guidance informs recommendations but never overrides thresholds or operator authority.
9. **Generation Telemetry:** The router reports end-to-end latency and token usage from the successful provider. `UrbanAiAssistantComponent` caches telemetry with the response and shows a compact caption beneath it, including on cache hits without issuing another API call.

### Live Evaluator Rate Limits and Grounding

`ai_tests/run_llm_eval.py` uses the same validated YAML prompt registry and separate system/user
prompts, then runs its seven scenarios through the Groq-to-Gemini router with one worker and adaptive header-driven pacing (x-ratelimit-remaining-tokens / x-ratelimit-reset-tokens, 1.5s fallback)
between request attempts, preserving declared scenario order. Transient HTTP 429 errors and
empty payloads allow up to two adaptive retries with 15/30-second backoff; exhausted quotas
are not retried. Each scenario loads its authorized `SensorThresholdMatrix` and supplies
`[AUTHORIZED OPERATIONAL THRESHOLDS]` in both the prompt and retrieved evidence.
`LlmJudgeEvaluator` returns a Pydantic `LlmJudgeVerdict` for faithfulness, actionability, and
hallucination detection; rate-limited judge calls are skipped cleanly. Each scenario thus produces two complementary dynamic metrics without hardcoded values:
- **Deterministic Scenario Compliance (`matched_compliance`):** Checks normalized whole-word presence of the expected metric inside the `Assessment` section, computing `success_rate_percentage` dynamically (`passed / total * 100`).
- **Semantic Faithfulness (`avg_faithfulness_score`):** Computes the arithmetic mean of all valid `1–5` judge scores (`sum(scores) / len(scores)`), which `LlmEvalSummary` parses dynamically on `pages/4_About.py` (defaulting to `0.0` only when legacy reports lack judge scores).

### Fast Architecture Verification (`pages/4_About.py`)

The default verification uses a `ThreadPoolExecutor` to run the Pytest suite
(`sys.executable -m pytest -q --disable-warnings`) and native in-process RAG
guardrails in parallel. This mode does not call external AI providers; it reuses the latest LLM
evaluation report or displays **Cached / Skipped (Fast Mode)** when no report
exists. Selecting **Include Live LLM Eval (uses API tokens)** opts into the
external seven-scenario evaluation through the Groq-to-Gemini router.

### Local Alerting Engine (`src/utils/local_alerts.py`)
* **`AlertManager`** operates **100% offline**, completely independent of external AI provider availability.
* Evaluates threshold breaches (upper-bound limits for temperature, noise, traffic, PM2.5; lower-bound limit for soil moisture — drought detection).
* Enforces a **configurable cooldown (10 seconds)** to prevent log flooding (*alert fatigue*) and sanitizes `\r` / `\n` characters prior to writing to `security_alerts.log` to mitigate **Log Injection** attacks.
