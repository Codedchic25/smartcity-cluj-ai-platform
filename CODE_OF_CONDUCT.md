# 🛡️ TECHNICAL & CIVIC CODE OF CONDUCT: SMART CITY CLUJ IoT

## 1. Our Institutional Commitment
In the interest of fostering an open, academic, high-integrity, and engineering-rigorous development environment, we pledge to make participation in the **Smart City Cluj IoT** project a harassment-free experience for all contributors, regardless of experience level, educational background, or technical role.

## 2. Technical Integrity & Governance Standards
Because this system simulates critical urban infrastructure decisions (city alerts, ecological indicators, and climate forecasts), our standards mandate:
* **Absolute Determinism**: Any modification to decision algorithms, localization matrices, or hardcoded thresholds in Python must be validated through the automated test suite using `pytest` (executed via `uv run pytest`) and the `run_llm_eval.py` evaluation engine.
* **Civic Data Security**: Protecting SQLite database integrity by relying exclusively on native transactional methods inside the `CityRepository` class and defensive context managers to prevent database file locks (`EBUSY` errors or locked resources).

## 3. Unacceptable Behavior
* Intentionally injecting corrupted sensor parameters or modifying source code to bypass local security guardrails.
* Publicly sharing plaintext credentials or private API keys (`GROQ_API_KEY`) used during development inside files tracked by version control. Administrative passwords must use Argon2id encoded hashes and must never be committed to version control.
