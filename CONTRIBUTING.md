# 🤝 Contributing Guide for Smart City Cluj IoT

Thank you for contributing to the urban platform! To maintain architectural integrity and a clean commit history, please follow this set of administrative and technical guidelines.

---

## 🚀 Local Development & Deployment Workflow

All environment operations must be performed exclusively using the high-speed `uv` package manager. Do not install dependencies manually without registering them in the project configuration.

1. **Clone and initialize the environment:**
   ```bash
   git clone https://github.com/Codedchic25/smartcity-cluj-iot-platform.git
   cd smartcity-cluj-iot-platform
   uv sync
   ```

2. **Pre-Commit/Push Pipeline Verification:**
   Before pushing changes to the `main` branch (which automatically triggers the CI/CD pipeline), ensure that all unit tests, linters, and security evaluations pass cleanly:
   ```bash
   # Commands for validating quality guardrails and generating JSON reports
   uv run ruff check .
   uv run pytest
   uv run python ai_tests/run_guardrail_eval.py
   uv run python ai_tests/run_llm_eval.py
   ```

---

## 🛠️ Code Quality Standards (Ruff-Style)

The project enforces strict static analysis. Any code violating these rules will be rejected during integration by the CI/CD pipeline:
* All Python source files must be linted and formatted using `ruff`. Running `uv run ruff format .` and `uv run ruff check . --fix` must return a clean status with zero redefinition errors (`F811`) or duplicate fields (`PIE794`).
* Do not leave residual `print()` statements in backend layers; use the native logging system integrated into `main.py` (`LOGGER.info`).
* Preserve strict inequalities when evaluating telemetry thresholds to prevent alert oscillation on the dashboard.

---

## 🧪 Test Suite, Security & MLOps Integration

* **Backend Regression Tests:** Whenever you add new features or update localization dictionaries, run the full test suite using `uv run pytest`. This executes all automated tests and automatically generates the structured metric report `ai_tests/pytest_report.json`.
* **Semantic Validation:** Changes to prompts in `ai_tests/prompts.txt` require re-running the Python evaluation matrix (`run_llm_eval.py`) to verify alignment across critical urban profiles directly against the active production model `openai/gpt-oss-20b`, outputting `ai_tests/llm_eval_report.json`.
* **RAG & Guardrails:** Run `uv run python ai_tests/run_guardrail_eval.py` for the seven native filtering, sanitization, and numerical grounding checks. The report is written to `ai_tests/rag_guardrails_report.json`.
* **Live LLM Evaluation:** Run `uv run python ai_tests/run_llm_eval.py` with a configured Groq API key. Results are recorded in `ai_tests/llm_eval_report.json`; do not assume prior runs guarantee a current passing score.

---

## 🔐 Security & Isolation

* **No Secrets in Git:** Never commit private API keys or plaintext passwords into source code. All sensitive environment variables must remain isolated on disk inside the local `.env` file.
* **Constant-Time Cryptography:** Administrative authentication verifies the **SHA-256** digest from `.env` using `hmac.compare_digest` to prevent timing side-channel attacks and eliminate hardcoded credentials (backdoors).
* **Local Audit System:** All alerting components must write through the 10-second cooldown shield into `security_alerts.log`, explicitly stripping `\n` and `\r` characters to mitigate Log Injection / Log Forgery attacks.
