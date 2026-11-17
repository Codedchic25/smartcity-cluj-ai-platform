# 🎓 Advanced Technical Defense Guide (Q&A Matrix) — Smart City Cluj IoT Core

This document centralizes architectural questions, engineering decisions, and troubleshooting scenarios that may arise during technical reviews of the project, providing clear defenses grounded directly in the platform's production source code.

---

## 🏗️ Section 1: Architecture Decisions & Streamlit Concurrency

### Q1: Across the visualization pages, you execute direct SQL queries via the `CityRepository` class. Why did you choose this approach inside page functions?
* **Defense**: Streamlit operates as a repetitive top-to-bottom script execution model (auto-refresh loop) triggered on every user interaction with the UI. Opening and closing repository instances inside page functions guarantees strict isolation.
* **Technical Justification**: To interface the local SQLite database with the dynamic rendering loop, we used synchronous connections protected by strict timeouts (`timeout=5.0`) inside the `CityRepository` class in `main.py`. This enables rapid data retrieval for Pandas DataFrames and immediate session closure, preventing blocking `database is locked` errors when concurrent processes write in parallel.

### Q2: Why did you choose Streamlit's native Multipage architecture (`pages/` directory) instead of a simple selector inside a single monolithic file?
* **Defense**: A single-file monolithic approach (where screens are merely Python functions gated behind `if/else` conditional blocks) leads to accelerated performance degradation and severe Memory Bloat.
* **Technical Justification**: By leveraging the native multipage structure (`pages/`), Streamlit loads into RAM exclusively the source code of the active page currently viewed by the operator. Files are structurally isolated as independent modules, minimizing the runtime memory footprint and simplifying maintenance while preserving global state via `st.session_state`.

---

## 🚨 Section 2: Alert Logic & Localized Resource Optimization

### Q3: In the application code, you evaluate telemetry using strict inequalities (`>`). What is the mathematical impact of this decision on the system?
* **Defense**: Non-strict inequalities introduce a high risk of alert oscillation in IoT systems whenever a malfunctioning sensor repeatedly transmits the exact boundary value.
* **Technical Justification**: By standardizing on strict inequalities (`temperature > 32.0`, `air_quality > 80.0`, `soil_moisture < 35.0`), we aligned the codebase with the formal specification in the documentation. If a sensor reports the exact threshold value, the system treats it as the upper bound of the safe zone and does NOT trigger an alert, completely eliminating operator alert fatigue.

---

## 🧠 Section 3: AI Security & MLOps Validation (`openai/gpt-oss-20b`)

### Q4: How does the AI processing pipeline enforce protection against Prompt Injection attacks?
* **Defense**: Rather than relying on brittle string-filtering heuristics, the project enforces strict data isolation inside a clearly delimited prompt template.
* **Technical Justification**: Incoming sensor telemetry is injected into clearly separated sections using structured delimiters inside `ai_tests/prompts.txt`. This prevents the LLM from confusing telemetry payloads with system instructions. Furthermore, setting a low temperature (`0.2`) in `groq_provider.py` constrains the model to remain deterministic and deliver concise technical recommendations, mitigating context-escape risks. A regex sanitization pass also strips internal `<think>...</think>` reasoning tags to ensure clean output.

### Q5: Inside the test suite, you integrated a programmatic evaluation script named `run_llm_eval.py`. What does it validate, and how does it ensure Quality Assurance (QA)?
* **Defense**: LLM outputs are probabilistic and can drift over time in cloud environments. The project treats the generative component like traditional software that must undergo automated testing.
* **Technical Justification**: The `run_llm_eval.py` script evaluates **7 urban scenarios in Cluj-Napoca**, querying `GroqProvider` with the `openai/gpt-oss-20b` model. It checks for whole-word focal term matches, normalizes diacritic differences, and scopes evaluation strictly to the `Assessment` section. The output JSON report records the actual outcome of each run; a passing score on a previous run does not guarantee success on subsequent evaluations.

---

## 🌐 Section 4: Session Management, I18n & Reactivity

### Q6: Across the files in `pages/`, you configured dedicated callbacks for the language selectors (e.g., `on_change=global_lang_callback`). How do you justify this design from a global synchronization perspective?
* **Defense**: By default, widget states inside subpages reset upon page navigation. A standard widget binding would revert the platform language back to the default setting on every screen transition.
* **Technical Justification**: The callback synchronizes the widget selection directly with the persistent `st.session_state["lang"]` key and propagates the active localization dictionary from `TranslationProvider` across all pages, ensuring seamless, stateful multi-language switching without `KeyError` or `AttributeError` regressions.
