# 📁 Operational Gallery & Visual Assets (UI/UX)

This document lists the module structure and production-verified screenshots that validate the functionality of the Smart City Cluj-Napoca platform.

### 🔒 Operational Security & Central Control Panel
* **01. Secure Operator Authentication:** Access checkpoint powered by **Argon2id** password verification and constant-time username comparison (`hmac.compare_digest`), protecting the `main.py` interface using credentials supplied through secure configuration.
* **02. Centralized Control Dashboard:** Symmetrical KPI display for monitored parameters (Temperature, Air Quality, Soil Moisture, Noise, Traffic) inside the Dashboard.
* **03. Industrial Audit Ledger:** Real-time logging of transactional telemetry directly into the `city_stats` SQLite table and alert recording with a 10-second cooldown inside `security_alerts.log`.
* **04. Multi-Language Synchronization:** Dynamic localization via callbacks bound to `st.session_state["lang"]` consuming the public services provided by the `TranslationProvider` class (RO, EN, IT, ES, HU).
* **05. Open-Source Geospatial Engine:** Native geospatial mapping of the 5 unique IoT stations across Cluj-Napoca on interactive maps powered by the **Folium** component.
* **06. System Administration Panel:** Dedicated `Settings` section featuring atomic sliders for dynamically configuring and updating node alert thresholds in the database.

### 🔬 Applied AI & Machine Learning Forecasts (Llama Engine)

* **07. LLM Reasoning Trace:** Operational administrative recommendations generated through the active `openai/gpt-oss-20b` instance via `GroqProvider`.
* **08. Pearson Statistical Correlation Matrix:** Analytical heatmap on the `3_Analytics.py` page identifying linear dependencies across urban indicators.
* **09. Time-Series Trend Charts:** Predictive Plotly subplots inside the Dashboard rendered with stable, modern formatting.
* **10. Linear Temperature Prediction:** 5-step forward extrapolation automatically computed using a `LinearRegression` model or NumPy polynomials.
* **11. Road Traffic Predictive Deviation:** Congestion index forecasting powered by numerical computations using NumPy `polyfit`.
* **12. Contextual AI Guardrails:** End-to-end assistant hardening enforced through immutable instructions in `prompts.txt`, paired with regex-based stripping of internal `<think>...</think>` reasoning tags.
