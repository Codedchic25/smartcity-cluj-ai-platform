# 🎓 Technical Study & Architecture Guide — Smart City Cluj-Napoca

## 0. Executive Summary (30-Second Elevator Pitch)

**Smart City Cluj-Napoca** is a modular urban intelligence, IoT telemetry, and AI-assisted analytics platform built in **Python 3.12** and **Streamlit**. The application combines deterministic data processing (**Pydantic v2** validation, **SQLite** persistence, **Pandas/NumPy** statistical analytics, cooldown-throttled audit logging) with operational recommendations generated via the **Groq Cloud API**, featuring API key rotation, multi-model fallback, cryptographic session security (**SHA-256 + HMAC-SHA256**), and native Python verification checks for RAG and LLM workflows.

### End-to-End Execution Pipeline

    [Login: SHA-256 + hmac.compare_digest]
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
           └──► [Context Builder: Telemetry + Thresholds + Latest 5 alerts from security_alerts.log]
                       │
                       ▼
                [GroqProvider: Multi-Model Fallback + Sanitization (<think>) -> UI Recommendations]

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
* **`app/ai/`:** AI interface abstraction (`ai_interface.py`), RAG/grounding pipeline (`rag_pipeline.py`), and resilient Groq client (`groq_provider.py`).
* **`app/database/` & `migrations/`:** SQLAlchemy ORM models, async session setup, and Alembic migrations prepared for external database scaling.
* **`ai_tests/`:** Pytest regression suite and native Python evaluators for RAG guardrails and LLM scenarios.

> **💡 Architectural Decision (Trade-off):** Maintaining a modular architecture (decoupling authentication, alerting, and LLM client logic from Streamlit UI code) enables fast unit testing with `Pytest` without spinning up a web server and simplifies future replacement of the UI or AI provider.

---

## 2. Security & Session Management

Authentication is engineered across two independent layers (Defense-in-Depth):

1. **Credential Verification (`AuthenticationService`):**
   * Reads the configured username (`PLATFORM_ADMIN_USER`) and password hash (`PLATFORM_ADMIN_PASS`) from secure configuration.
   * Converts the submitted form password into a **SHA-256** hexadecimal digest at runtime and compares it against the stored hash using **`hmac.compare_digest()`** (constant-time comparison to mitigate *timing side-channel* attacks).
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

## 4. AI / Groq Integration & Deterministic Alert Engine

### AI Resilience & Multi-Model Fallback (`app/ai/groq_provider.py`)
1. **Context Construction (`ai_interface.py`):** Loads the prompt template from `ai_tests/prompts.txt` and dynamically injects the selected station, current telemetry, configured thresholds, active UI language (`RO`, `EN`, `IT`, `ES`, `HU`), and the latest 5 events from `security_alerts.log`.
2. **Key Rotation & Fallback (`GroqProvider`):** Collects `GROQ_API_KEY*` entries from `st.secrets` and environment variables, supporting lists and CSV strings. Rotates keys on HTTP 401, 429, and 5xx errors, and falls back across backup models if the primary model is unavailable.
3. **Response Sanitization:** Strips internal reasoning blocks (`<think>...</think>`) before rendering recommendations and caches the result in `st.session_state` per station.
4. **Language & Markdown Alignment:** Aligns response body text with the active language (`RO`, `EN`, `IT`, `ES`, or `HU`) and places the bold English Assessment, Recommended Actions, and Data Limitations labels on separate lines as Markdown sections.

### Local Alerting Engine (`src/utils/local_alerts.py`)
* **`AlertManager`** operates **100% offline**, completely independent of Groq API availability.
* Evaluates threshold breaches (upper-bound limits for temperature, noise, traffic, PM2.5; lower-bound limit for soil moisture — drought detection).
* Enforces a **configurable cooldown (10 seconds)** to prevent log flooding (*alert fatigue*) and sanitizes `\r` / `\n` characters prior to writing to `security_alerts.log` to mitigate **Log Injection** attacks.
