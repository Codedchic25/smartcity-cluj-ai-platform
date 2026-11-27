# 🗺️ Technical Development Roadmap: Smart City Cluj IoT Core

This document outlines the architectural evolution phases for the Cluj-Napoca urban control
platform, transitioning from a local educational prototype toward a high-availability distributed
microservices ecosystem.

---

## 📅 Phase 1: Decoupling, Cloud Migration & API Services (Q3 2026) — *In Progress*

- **Always-On Cloud Hosting & Live Quality Control Center (*Completed*)**: The platform has been
  migrated onto Streamlit Community Cloud with Argon2id + HMAC-SHA256 session security, Pydantic v2
  runtime contracts, multi-provider AI failover (`qwen/qwen3.8-27b` -> `gemini-3.5-flash`), and an
  interactive **Systemic Quality & AI Compliance Control Center** on the `About` page.
- **FastAPI Backend Migration**: Complete decoupling of business logic from the Streamlit
  presentation layer by introducing an asynchronous FastAPI backend with protected REST
  endpoints, leveraging strict dependency management via `uv` and `pyproject.toml`.
- **Production PostgreSQL**: Replacing the local SQLite database (`app.db`) with a clustered
  PostgreSQL instance optimized for high-concurrency queries and advanced geospatial indexing
  via the `PostGIS` extension.

## 📅 Phase 2: Edge Computing & Advanced Security (Q4 2026)

- **MQTT Broker Integration**: Replacing the simulated ingestion pipeline in `seed_db.py` with
  a production MQTT broker (e.g., Eclipse Mosquitto) capable of ingesting encrypted live
  telemetry streams directly from physical field sensors.
- **Semantic Firewall Expansion**: Extending the existing deterministic Python guardrails (TF-IDF SOP
  grounding, numeric sensor validation, and `LlmJudgeEvaluator`) with programmable guardrail rails
  (such as `NeMo Guardrails`) across the `qwen/qwen3.8-27b -> gemini-3.5-flash` router to block complex
  context-escape attempts.

## 📅 Phase 3: Scalability & Automated MLOps (Q1 2027)

- **Docker & Kubernetes Orchestration**: Full containerization of microservices by optimizing
  `Dockerfile` builds and orchestrating workloads with Kubernetes to guarantee High Availability
  and automated horizontal scaling based on sensor network load.
- **Predictive Retraining Pipeline**: Automating the Scikit-Learn pipeline to retrain Linear
  Regression (`LinearRegression`) models nightly on newly ingested telemetry across Cluj-Napoca
  neighborhoods, expanding forecasting capabilities with `statsmodels` and `prophet`.

## 📅 Phase 4: Distributed Monitoring & Multi-Region Sync (Q2 2027)

* **Architectural Question**: Once the platform migrates to PostgreSQL/PostGIS and FastAPI across
  a multi-region infrastructure, how will the local `security_alerts.log` audit ledger be
  managed to prevent data consistency loss?
* **Defense & Direction**: Single-node local file logging will be replaced by a centralized log
  aggregation agent (such as Vector or Fluent Bit). Audit events will stream asynchronously to
  a dedicated Elasticsearch/Grafana Loki cluster while maintaining a local disk fallback buffer
  (5 MB rotating log files) on each compute node. This ensures security guardrails, the 10-second
  cooldown mechanism, and audit rules remain resilient against temporary cloud network outages.
