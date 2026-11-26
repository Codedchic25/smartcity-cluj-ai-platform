# 🏙️ Interactive Walkthrough & Validation Guide (Urban Walkthrough)

This document guides evaluators, review boards, or recruiters step by step through every module of the **Smart City Cluj-Napoca IoT Core** platform, demonstrating end-to-end functionality and key engineering decisions.

---

## 🛠️ Step 1: Bootstrap & Deterministic Data Initialization

Upon launching `main.py`, the SQLite schema is created and the idempotent seeder automatically
populates an empty database with the 7 demo monitoring stations in Cluj-Napoca, along with their
thresholds and associated telemetry. Existing data is never overwritten, including across fresh
Streamlit Community Cloud runtimes.

1. Open a terminal in the project root: `smartcity-cluj-iot-platform`.
2. For an empty database, seeding occurs automatically when the application starts. The standalone script remains
   available for manual population while preserving existing records:
   ```bash
   uv run python seed_db.py
   ```
   To intentionally replace existing stations, thresholds, and telemetry, run
   `uv run python seed_db.py --force` only after creating a backup.
3. To verify the structural integrity of the generated SQLite tables (`city_stats`, `sensors`, `settings`), run the local diagnostic utility:
   ```bash
   uv run python check_db.py
   ```

---

## 🔒 Step 2: Secure Operator Authentication

1. Start the local Streamlit server using high-speed orchestration:
   ```bash
   uv run streamlit run main.py
   ```
2. Open the local URL displayed in the terminal (`http://localhost:8501`).
3. The interface is immediately gated by the secure login screen. Administrative credentials are supplied through secure configuration. Password verification uses Argon2id, while username comparison uses `hmac.compare_digest`. No default plaintext administrative password is provided.

---

## 📡 Step 3: Live Monitoring & Alerting Logic (Dashboard)

1. Navigate to **`1_Dashboard.py`**. You will see symmetrical live telemetry KPI columns alongside the native open-source Folium geospatial map. Air quality metrics are rendered using the standard scientific unit **`µg/m³`**.
2. Switch the platform language via the global sidebar selector. The global callback immediately propagates the localization bundle exposed by the `TranslationProvider` class across all supported languages (RO, EN, IT, ES, HU), updating the UI cleanly without `AttributeError` regressions.
3. To simulate a critical environmental breach, the state generator pushes a reading that exceeds immutable thresholds (e.g., Air Quality PM2.5 > 80.0 or Temperature > 32.0°C).
4. **Anti-Flood Shield Validation:** A red alert banner appears on screen. The system writes a sanitized entry—hardened against log-injection attacks—into `security_alerts.log`, governed by the 10-second cooldown barrier enforced by the `AlertManager` class.

---

## 🔬 Step 4: Predictive Analytics & Applied AI (Analytics)

1. Navigate to **`3_Analytics.py`**.
2. **Classical ML Forecasting:** Select an urban parameter. A Linear Regression model or NumPy `polyfit` polynomial estimator queries historical SQLite records, computes the trend slope, and renders the upcoming predictive intervals on the Plotly chart without stale visual caching.
3. **Operational Recommendation (Llama Engine):** At the bottom of the page, the contextual AI assistant is ready for queries. Upon submission, the pipeline retrieves dense context from the latest 5 log entries, injects it into the prompt template from `app/ai/prompts.yaml`, and invokes the production `openai/gpt-oss-20b` model to generate a tailored administrative recommendation. The output—stripped of internal `<think>` tags via regex—is securely persisted in `st.session_state` to survive UI auto-refresh cycles.

---

## 🧪 Step 5: DevOps & MLOps Validation (Test Suite)

To verify application resilience in CI/CD pipelines prior to final deployment, manually execute the automated test suites to generate JSON metric reports inside `ai_tests/`:

1. **Unit & Regression Testing (Pytest):**
   ```bash
   uv run pytest
   ```
   A fresh suite report is generated on every run. Tests cover empty-database bootstrapping,
   preservation of existing records, missing-sensor warnings across all three pages, translation
   integrity, session state checks, and safe seeder behavior. Results are written to
   `ai_tests/pytest_report.json`.
2. **Programmatic Python Evaluation:**
   ```bash
   uv run python ai_tests/run_llm_eval.py
   ```
   This command executes the 5 Cluj-Napoca operational scenarios, verifying semantic compliance of the AI assistant and writing the output report to `ai_tests/llm_eval_report.json`.
