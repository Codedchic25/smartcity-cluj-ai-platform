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

### 🚨 Issue: Operator receives `Invalid credentials token` despite entering the correct password
* **Cause:** The local `.env` file stores a password in plain text or using a legacy algorithm (e.g., an MD5 hash), whereas the security engine in `main.py` requires a modern hexadecimal digest.
* **Resolution:**
  1. Generate the proper hash for your password using the **SHA-256** algorithm.
  2. Open the `.env` file and ensure the `PLATFORM_ADMIN_PASS` variable is populated exclusively with the resulting 64-character hexadecimal fingerprint.
  3. The system validates credentials using `hmac.compare_digest` in constant time to mitigate timing side-channel attacks.

---

## 3. AI Service Degradation and Network Outages (Groq Cloud API)

### 🚨 Issue: Sidebar or chat windows display error messages or rate-limit warnings
* **Cause:** The external Groq Cloud API is experiencing a temporary network outage or timeout, or all configured API keys have expired.
* **Fail-Safe Mechanism:** The application rotates through `GROQ_API_KEY`, `GROQ_API_KEY_1`, `GROQ_API_KEY_2` (or the comma-separated `GROQ_API_KEYS` list) upon receiving HTTP 401, 429, or 5xx responses. Keys can be configured in `.env` or Streamlit Cloud Secrets. If all keys are exhausted, the UI displays a controlled error message.
* **Impact:** The rest of the platform remains **100% functional and locally isolated**. SQLite telemetry collection, NumPy Pearson correlation matrix rendering, Plotly chart generation, and cooldown-throttled alert logging in `security_alerts.log` continue running uninterrupted and completely offline.

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

The **"Run Live Architecture Verification"** button invokes the Python evaluators for LLM scenarios and RAG/guardrail checks. Live reports are saved inside `ai_tests/`, and the metrics are updated in the current Streamlit session.
