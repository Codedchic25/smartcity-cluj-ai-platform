# 🛠️ Technical Troubleshooting Guide: Smart City Cluj IoT Core

This document centralizes common operational errors, infrastructure bottlenecks, and formal platform recovery procedures across local and containerized environments.

---

## 1. Persistence Layer Locks (SQLite Engine)

### 🚨 Error: `sqlite3.OperationalError: database is locked` or Windows `EBUSY` lock
* **Cause:** An active Streamlit or Python process may still hold an open connection to the database.
* **Resolution:** Stop the Streamlit session using the database (for example, by pressing `Ctrl+C`
  in the terminal where it was started), then retry the operation. Running the seeder does not
  unlock an active connection and must not be used as a remedy for a lock error.

### 🚨 Error: `No sensors registered inside the local relational database ledger schema`
* **Cause:** Normally, `main.py` creates the schema and runs the idempotent seeder, which
  automatically populates an empty database. If this message still appears, verify that the page
  and the startup process use the same `DATABASE_PATH` value and check the logs for any
  initialization errors.
* **Resolution:** After verifying the path and logs, you can manually populate an empty SQLite database:
  ```powershell
  uv run python seed_db.py
  ```
  By default, the seeder inserts data only if the `sensors`, `settings`, and `city_stats` tables
  are all empty; if data exists in any of them, the operation is skipped to preserve existing
  configuration and telemetry. For an intentional reset, running `uv run python seed_db.py --force`
  clears existing data from these tables and reinstalls the demo values. Back up your database
  before using `--force`.
* **Missing selection:** Dashboard, Settings, and Analytics display `st.warning` and halt execution
  if no station exists or if the current selection no longer matches a station in the database.
* **Containers:** The default command in `Dockerfile` runs the seeder at container startup
  (not during image build). With the safe default behavior, an already populated database
  is preserved.

---

## 2. Authentication Failures and Cryptographic Conflicts

### Issue: Operator or recruiter receives an invalid credentials message

**Possible causes:**

- The configured username does not match the expected account.
- `PLATFORM_ADMIN_PASS` is missing or contains an invalid password hash.
- The local `.env` file is missing or has a malformed assignment.
- A legacy SHA-256 digest or plaintext password is being used instead of
  an Argon2id encoded password hash.
- Streamlit Secrets are missing, incorrectly configured, or have not
  been updated in the deployed application.

**Resolution:**

1. For local use, copy `.env.example` to `.env` and configure its five
   documented keys. Keep `.env` out of version control; `.gitignore`
   excludes it.
2. Check that assignments use `KEY=VALUE` or `KEY = VALUE`. The session
   authentication loader trims whitespace around keys and values.
3. For Streamlit Community Cloud, configure the values under Settings → Secrets.
4. Verify that `PLATFORM_ADMIN_USER` contains the expected username.
5. Verify that `PLATFORM_ADMIN_PASS` contains a valid Argon2id encoded hash.
6. Verify that `SESSION_SECURITY_TOKEN` contains a dedicated signing key.
7. Restart or redeploy the application after changing deployed settings if required.

### Password Verification

The application uses `PasswordHasher.verify()` to verify passwords
against their stored Argon2id hashes.

Do not generate SHA-256 password fingerprints or store plaintext
passwords in `PLATFORM_ADMIN_PASS`.

### Session Token Security

Session tokens are signed separately using HMAC-SHA256.

`SESSION_SECURITY_TOKEN` must be a dedicated, sufficiently random secret.
It must not be reused as a password or published in source code.

If session validation fails, verify that the signing key is correctly
configured and consistent across application instances.

Never expose actual recruiter credentials, password hashes, or signing
keys in public repositories, logs, screenshots, or documentation.
## 3. AI Service Degradation and Network Outages (Groq Cloud API)

### 🚨 Issue: Sidebar or chat windows display error messages or rate-limit warnings
* **Cause:** The external Groq Cloud API is experiencing a temporary network outage or timeout, or all configured API keys have expired.
* **Fail-Safe Mechanism:** The application rotates through `GROQ_API_KEY`, `GROQ_API_KEY_1`, `GROQ_API_KEY_2` (or the comma-separated `GROQ_API_KEYS` list) upon receiving HTTP 401, 429, or 5xx responses. Keys can be configured in `.env` or Streamlit Cloud Secrets. If all keys are exhausted, the UI displays a controlled error message.
* **Impact:** The rest of the platform remains **100% functional and locally isolated**. SQLite telemetry collection, NumPy Pearson correlation matrix rendering, Plotly chart generation, and cooldown-throttled alert logging in `security_alerts.log` continue running uninterrupted and completely offline.

The Groq client is configured with `max_retries=0`, so the SDK does not wait
through automatic retries after HTTP 429. `GroqProvider` handles the response
through its key rotation and model fallback. Recommendation generation still
requires an explicit button click; identical prompt, telemetry, and language
inputs reuse a cached result for up to 300 seconds, reducing repeated token use.

### 🚨 Issue: `Received an empty message context response` or `0.0%` LLM Scenario Compliance on reasoning models
* **Cause:** Reasoning-capable models (such as `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen3`, or `deepseek-r1`) allocate completion tokens to internal chain-of-thought reasoning before emitting the final JSON payload. If `max_tokens` is set too low (e.g., `800`), the model exhausts its token budget during internal reasoning and returns an empty `message.content` string (`""`), causing UI recommendations to appear blank and scenario evaluations to fail.
* **Resolution:**
  1. Ensure `GroqProvider` (`app/ai/groq_provider.py`) dynamically allocates an expanded token budget (`max_tokens=2048`) for reasoning models while preserving `max_tokens=800` for standard instruction models (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`).
  2. Treat empty `message.content` responses as a retryable model failure so the gateway automatically falls back to the next candidate model in `FALLBACK_MODELS` instead of returning early.
  3. Isolate the per-attempt message history (`list(base_messages)`) so grounding validation retries do not leak correction prompts into subsequent fallback models.

---

## 4. Automated Testing and Reporting Desynchronizations

### 🚨 Issue: JSON metric files in `ai_tests/` are missing or incomplete
* **Cause:** Tests were executed via isolated file invocations instead of using the project's centralized global configuration.
* **Resolution:** Run the full unit test suite using the unified command:
  ```bash
  uv run pytest
  ```
  This automatically reads the flags from `pyproject.toml` and generates the unified `ai_tests/pytest_report.json` file on disk, which is seamlessly loaded by the quality control dashboard on the `4_About.py` page.

  For dedicated RAG security guardrails and LLM evaluation checks, run:
  ```bash
  uv run python ai_tests/run_guardrail_eval.py
  uv run python ai_tests/run_llm_eval.py
  ```

The **"Run Live Architecture Verification"** button runs Pytest and native
in-process guardrails concurrently by default and does not call Groq. It reuses
the latest LLM evaluation report or displays **Cached / Skipped (Fast Mode)**
when no report exists. To use API tokens for a fresh LLM evaluation, select
**Include Live LLM Eval (uses API tokens)** next to the button before
starting verification. The live evaluator tries Groq first and may fall back to Gemini, using a fast 3-scenario sample (`max_scenarios=3`) with the non-reasoning `llama-3.1-8b-instant` judge in the UI (or `--scenarios N` in CLI) to avoid free-tier HTTP 429 TPM rate limits. If `python-dotenv` logs a parse warning at startup, ensure `.env` entries contain no spaces around `=` (e.g., `GROQ_API_KEY_1=gsk_...`).
