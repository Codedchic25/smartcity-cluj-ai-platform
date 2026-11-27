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

## 🧠 Section 3: AI Security & MLOps Validation (`qwen/qwen3.8-27b`)

### Q4: How does the AI processing pipeline enforce protection against Prompt Injection attacks?
* **Defense**: Rather than relying on brittle string-filtering heuristics, the project enforces strict data isolation inside a clearly delimited prompt template.
* **Technical Justification**: Incoming sensor telemetry is injected into clearly separated sections using the structured prompt template in `app/ai/prompts.yaml`. This prevents the LLM from confusing telemetry payloads with system instructions. Furthermore, setting a low temperature (`0.2`) in `groq_provider.py` constrains the model to remain deterministic and deliver concise technical recommendations, mitigating context-escape risks. A regex sanitization pass also strips internal `<think>...</think>` reasoning tags to ensure clean output.

### Q5: Inside the test suite, you integrated a programmatic evaluation script named `run_llm_eval.py`. What does it validate, and how do it and the About dashboard separate compliance from faithfulness without hardcoding?
* **Defense**: LLM outputs are probabilistic and can drift over time in cloud environments. Treating the generative layer like testable software requires both deterministic keyword checks and independent semantic auditing.
* **Technical Justification**: `run_llm_eval.py` evaluates **7 urban scenarios in Cluj-Napoca** via `qwen/qwen3.8-27b` (failing over to `gemini-3.5-flash`) and produces two dynamic metrics in `ai_tests/llm_eval_report.json`:
  1. **Deterministic Scenario Compliance (`success_rate_percentage`):** Verifies normalized whole-word presence of the focal telemetry term inside the `Assessment` section and computes the pass rate dynamically from live scenario outcomes.
  2. **Semantic Faithfulness (`avg_faithfulness_score`):** `LlmJudgeEvaluator` audits each output strictly against the scenario's sensor readings and authorized `SensorThresholdMatrix`, returning a Pydantic-validated `1–5` score (`LlmJudgeVerdict`). `_write_report()` calculates the arithmetic mean across valid verdicts, and `LlmEvalSummary` renders it dynamically on `pages/4_About.py` without hardcoded numbers.

### Q6 (AI Resilience): How does the router prevent HTTP 429 rate limits or reasoning-model token truncation from breaking live recommendations?
* **Defense**: Relying on a single API key or small token budgets causes silent failures when reasoning models emit internal `<think>` chains before structured JSON.
* **Technical Justification**: `FallbackAiProviderRouter` disables blind SDK retries (`max_retries=0`), rotates across configured `GROQ_API_KEY*` credentials and Groq candidate models (`qwen/qwen3.8-27b`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b`), and fails over seamlessly to Google Gemini (`gemini-3.5-flash` -> `gemini-3.8-flash`) using the native `google-genai` SDK with strict Pydantic `response_schema` decoding and dedicated completion token budgets (`2048–2500` tokens).

---

## 🌐 Section 4: Session Management, I18n & Reactivity

### Q6: Across the files in `pages/`, you configured dedicated callbacks for the language selectors (e.g., `on_change=global_lang_callback`). How do you justify this design from a global synchronization perspective?
* **Defense**: By default, widget states inside subpages reset upon page navigation. A standard widget binding would revert the platform language back to the default setting on every screen transition.
* **Technical Justification**: The callback synchronizes the widget selection directly with the persistent `st.session_state["lang"]` key and propagates the active localization dictionary from `TranslationProvider` across all pages, ensuring seamless, stateful multi-language switching without `KeyError` or `AttributeError` regressions.
