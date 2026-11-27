# 🛡️ Project Defense Guide (Interview Defense) — Smart City Cluj-Napoca

This guide centralizes software design decisions, architectural trade-offs, and defensive
guardrails implemented across the platform. It is structured to answer Senior/Architect-level
questions during technical interviews or academic evaluations.

---

## 🗄️ 1. Persistence & Connection Management

### Question: Why did you choose SQLite for a Smart City application simulating continuous IoT data streams? Doesn't SQLite suffer from concurrent write bottlenecks?
* **Defense**: The choice was driven by requirements for determinism, local isolation, and full
portability in a technical portfolio project (zero-config, zero-infrastructure overhead for any
evaluator or recruiter cloning the repository locally or in GitHub Codespaces).
* **Technical Justification**: To mitigate native SQLite locking limitations under concurrent writes
(`EBUSY` / `database is locked` errors), we implemented multiple defensive layers:
  1. Synchronous connections inside `CityRepository` specify an explicit `timeout=5.0` upon
  opening, preventing immediate transaction crashes.
  2. We configured an atomic UPSERT mechanism (`INSERT ... ON CONFLICT(sensor_id) DO UPDATE SET`)
  inside `main.py` that executes threshold updates within a single safe transaction.
  3. The test suite in `ai_tests/` runs in complete isolation via `pytest`, validating business
  logic and thresholds against temporary databases before touching disk state.
  4. **Nominal Row Mapping Fix:** SQLite queries rely exclusively on nominal `sqlite3.Row`
  objects or explicit column mappings. This decouples the UI from column ordering in the database,
  eliminating positional indexing crashes.

---

## 🚨 2. Service Isolation & Local Anti-Flood Logic

### Question: What happens if a sensor gets stuck above a critical threshold? How did you prevent blocking the main thread and log flooding?
* **Defense**: External cloud messaging dependencies introduce network vulnerabilities, high
I/O-bound latency, and unpredictable operational overhead. Therefore, we decoupled the platform
from third-party alerting services and standardized on fast local audit logging inside
`security_alerts.log`.
* **Technical Justification**: To prevent disk exhaustion from repetitive writes when a faulty
sensor continuously emits anomalies, the application enforces a strict temporal barrier. During
telemetry simulations, history writes across the 5 sensor metrics are governed by a strict
10-second anti-spam cooldown inside `AlertProcessor` / `AlertManager`. Streamlit UI rendering
operates independently of the write transaction flow: the server sequentially reads the current
state from SQLite and renders UI components with minimal latency.

---

## 🧠 3. Deterministic Evaluation of Context-Aware Generative AI

### Question: LLM outputs are non-deterministic. How do you guarantee that the AI assistant's recommendations are safe, multilingual, and grounded in operational telemetry?
* **Defense**: Across this platform, the Artificial Intelligence component is treated as a standard
software module subject to continuous metric-driven testing rather than a probabilistic black box.
We engineered the assistant as a Context-Aware urban analyst hardened against Prompt Injection
attacks and numerical hallucinations.
* **Technical Justification**:
  1. In `app/ai/ai_interface.py`, the assistant dynamically resolves active language context by
  querying the localization dictionaries exposed via `get_all_translations()` in `translations.py`
  (RO, EN, IT, ES, HU).
  2. `FallbackAiProviderRouter` tries the `qwen/qwen3.8-27b` Groq model first (with `openai/gpt-oss-20b` and `openai/gpt-oss-120b` fallbacks), rotating
  `GROQ_API_KEYS` and Groq models on provider failures. If Groq keys/models fail or a 429/TPM
  limit is returned, it switches to `GeminiOpenAiProvider`, which uses the `openai` SDK with
  Google's OpenAI-compatible Gemini endpoint, always trying `configured_model` first followed by
  deduplicated fallback candidates (`gemini-3.5-flash`, `gemini-3.5-flash-lite`,
  `gemini-3.6-flash`, `gemini-3.8-flash`) without dropping configured models. Router
  failures are classified structurally via `is_provider_failure` (typed API/network errors,
  HTTP `>= 400` codes, structured telemetry errors, and normalized failure messages) without
  misclassifying numeric sensor telemetry. Both providers propagate `system_prompt_only` into
  API payloads and cached telemetry, enforce strict sensor-value-unit RAG grounding, and
  omit iterative Gemini correction retries to avoid doubling token cost and latency.
  3. **Recommendation Cache:** `UrbanAiAssistantComponent` applies
  `@st.cache_data(ttl=300, show_spinner=False)` to AI generation, keyed by the rendered prompt,
  telemetry inputs, and active language. Generation is still triggered only by the recommendation
  button; repeated matching inputs reuse the result for five minutes and reduce token consumption.
  4. The native Python evaluation script (`ai_tests/run_llm_eval.py`) executes **seven urban
  scenarios in Cluj-Napoca**. It verifies whole-word focal terms inside the `Assessment` section,
  normalizes diacritics, and reports actual provider errors; it uses one worker, spaces request
  attempts by 8.5 seconds, and preserves scenario order. Transient HTTP 429 and empty-payload
  errors receive up to two adaptive retries with 15/30-second backoff; exhausted quotas fail
  without retry. After each response, the Pydantic `LlmJudgeVerdict` scores faithfulness and
  actionability and flags hallucinations against the scenario's telemetry and thresholds. A
  judge-side 429 skips that verdict without failing the scenario; the report records the mean
  faithfulness score and the About card displays it, defaulting to zero for older reports. The
  prompt loads the station's authorized `SensorThresholdMatrix` and places
  `[AUTHORIZED OPERATIONAL THRESHOLDS]` in retrieved evidence. The final score is computed
  strictly from live run outcomes. The assistant and evaluator share the cached,
  Pydantic-v2-validated `app/ai/prompts.yaml` registry; system instructions and output
  guardrails are sent separately from the rendered telemetry user prompt.   The evaluator and application now load prompts exclusively from
  `app/ai/prompts.yaml`.
  5. A regex sanitization pass strips internal reasoning tags (`<think>...</think>`), paired with
  Pydantic schema validation and numerical grounding checks (`rag_pipeline.py` and
  `run_guardrail_eval.py`) before rendering responses in Streamlit.
  6. The About page's default architecture verification runs Pytest and in-process native
  guardrails concurrently through `ThreadPoolExecutor`, without external provider requests. A
  prior LLM score is reused when available; live Groq-to-Gemini scenarios run only after selecting
  the **Include Live LLM Eval (uses API tokens)** checkbox (running a fast 3-scenario sample
  via `max_scenarios=3` in the UI, while CLI/CI runs all 7 scenarios via `--scenarios`).
  7. **Historical RAG trend:** `TelemetryContextReader` selects up to five recent `city_stats`
  rows for the chosen station and summarizes endpoint deltas as rising, stable, or falling.
  Fewer than two valid readings produce an explicit no-trend fallback; the prompt treats every
  trend as historical, not as the current state or a causal explanation.
  8. **Deterministic response grounding:** After generation, `UrbanAiAssistantComponent` adds
  only omitted station identity, current metric values, and configured threshold breaches.
  These facts come from the already available telemetry and settings, so this completion step
  does not make another API call or consume additional tokens. Focused SQLite and grounding
  tests live in `ai_tests/test_ai_guardrails.py`.
  9. **Hybrid SOP Retrieval:** `SopKnowledgeRetriever` builds a lightweight TF-IDF index over
  `docs/cluj_urban_sop.md`, ranks passages with cosine similarity, and injects at most two
  relevant procedures as `[AUTHORIZED URBAN SOP CONTEXT]`. The SOPs keep interventions under
  authorized human control and do not supply invented thresholds.
  10. **LLM Observability:** The router measures end-to-end generation time and returns token
  counts from the successful provider. `UrbanAiAssistantComponent` caches these metrics with the
  answer and displays a small caption under it, so a cache hit does not make another request.

---

## 🔐 4. Layered Security Strategy (Defense in Depth)

### Question: How did you secure API keys and protect the application at runtime against cyberattacks?
* **Defense**: The project enforces a multi-layered security model (Defense in Depth) to minimize
both local and cloud attack surfaces.
* **Technical Justification**:
  1. **Layer 1 (Secret Isolation):** All private credentials and API keys (`GROQ_API_KEY`,
  `GROQ_API_KEYS`, `GEMINI_API_KEY`) are decoupled from source code and stored in the local `.env` file or
  encrypted cloud secrets, excluded from Git via `.gitignore` and `.dockerignore`.
  2. **Layer 2 (Constant-Time Cryptography):** To eliminate plaintext password exposure,
  Runtime authentication verifies the configured Argon2id password hash using argon2-cffi. Username comparison uses hmac.compare_digest.
  `hmac.compare_digest` in constant time, mitigating timing side-channel attacks, paired with
  HMAC-SHA256 signed 24-hour session tokens.
  3. **Layer 3 (Access Control & State Isolation):** Multipage views under `pages/` are globally
  gated by Streamlit session state (`st.session_state["authenticated"]`). AI generation outputs
  are persisted per location in session memory to survive Streamlit's auto-refresh loop without
  duplicate widget collisions.

---

## 🛠️ 5. Ecosystem Modernization (`uv` vs. `requirements.txt`)

### Question: I noticed you use the `uv` package manager and define dependencies in `pyproject.toml`. What was the technical rationale?
* **Defense**: Maintaining manual `requirements.txt` files introduces a high risk of *Environment
Drift* (package version mismatches across developer machines, CI runners, and production).
* **Technical Justification**: The single source of truth for dependencies is defined using the
modern PEP 621 standard in `pyproject.toml`, while deterministic dependency resolution is locked
in `uv.lock`. Using `uv` (written in Rust) delivers near-instant installations via global caching
and guarantees reproducible execution via `uv run`. Static analysis and automated formatting are
enforced by `ruff` (0 errors / 0 warnings) in accordance with project configuration rules.
* **Static Method Optimization (N805):** We standardized on Python's native `@staticmethod`
decorator, ensuring full PEP 8 / Ruff compliance regarding first-argument naming without
artificial `self` parameters.
## 🧹 6. Local Quality Control & Cache Management on Windows

### Question: How do you ensure code consistency locally before pushing changes to the central CI/CD pipeline?
* **Defense**: To prevent stale bytecode/linter caches from accumulating and interfering with
static analysis, we built a dedicated local quality automation helper.
* **Technical Justification**: The Windows automation script `scripts/lint.ps1` spawns a background
PowerShell job (`Start-Job`) to purge residual cache directories (`__pycache__`, `.pytest_cache`,
`.ruff_cache`). Meanwhile, the main thread sequentially executes code formatting (`ruff format .`)
and static analysis with auto-fixes (`ruff check . --fix`), ensuring **0 detected errors** prior
to running the unified GitHub Actions workflow (`ci.yml`).

---

## 🌐 7. Cloud Architecture, Cost Optimization & Availability

### Question: Initially, you configured live deployment on Railway, but later migrated the application to Streamlit Community Cloud. What was the architectural reasoning behind this decision?
* **Defense**: The decision was governed by two core Software Engineering principles: **Cost
Optimization** and **Continuous High Availability**. Commercial general-purpose hosting platforms
carry the risk of service suspension once volatile free credits or billing limits are exhausted.
For a portfolio platform accessed asynchronously by academic evaluators and recruiters, such an
outage creates a critical *Single Point of Failure*.
* **Technical Justification**: By migrating to **Streamlit Community Cloud**, we eliminated reliance
on volatile runtime credits in favor of native, zero-cost hosting optimized for Python execution
threads. From a DevOps perspective, the pipeline executes Continuous Deployment (CD) via GitHub
webhooks bound to `main`, without requiring Node.js (`packages.txt` removed), resulting in fast
container rebuilds and reliable uptime for technical reviewers.

---

## 🔒 8. Cloud Secret Management (Security Constraints)

### Question: Since `.env` is excluded via `.gitignore` and never pushed to GitHub, how does your application run live on Streamlit Community Cloud without exposing provider keys or administrative credentials?
* **Defense**: The project strictly adheres to **OWASP Top 10** guidelines and *Twelve-Factor App*
principles regarding configuration and secret isolation. Configuration files are excluded from
public Git repositories and injected securely at runtime by the hosting platform.
* **Technical Justification**: Instead of storing secrets in version control, we utilize
**Streamlit Secrets Management**. Configuration variables are stored encrypted in the cloud
console as TOML entries. At runtime, Streamlit maps them into the protected `st.secrets` object,
which the provider router reads alongside `os.environ` and local `.env` configuration. Groq
supports single and rotating keys; Gemini uses `GEMINI_API_KEY` as the secondary provider
credential. This *Layered Security* model keeps provider keys out of source control.

---

## 🗂️ 9. Safe Payload Parsing from Third-Party AI SDKs

### Question: I noticed that in `groq_provider.py` you extract text using explicit list indexing. Why did you choose this approach, and what runtime errors did it prevent?
* **Defense**: Objects returned by modern LLM SDKs (such as `ChatCompletion` payloads) wrap
candidate completions as a list inside the `.choices` attribute. Accessing `.choices.message`
directly as a singular object is a common pitfall that immediately triggers a `TypeError` crash
at runtime.
* **Technical Justification**: To guarantee runtime resilience, we enforce explicit defensive
extraction of the first choice element via indexed access:
  ```python
  if completion_payload.choices:
      choice_item = completion_payload.choices[0]
      if hasattr(choice_item, "message") and choice_item.message:
          output_content = choice_item.message.content
  ```
  This structure validates collection presence and attribute availability before consuming the
  response string, preventing UI crashes on malformed or empty API payloads.

---

## 🎨 10. Unpacking Streamlit Column Objects as Context Managers

### Question: Why did you avoid passing `st.columns(2)` directly into a `with` context manager block?
* **Defense**: Calling `st.columns(n)` returns a standard Python `list` containing `n` independent
column container objects. Passing the list directly into a `with` statement (e.g., `with ctrl_cols:`)
raises a fatal runtime exception:
`TypeError: 'list' object does not support the context manager protocol`.
* **Technical Justification**: To maintain clean, multi-panel rendering across the Analytics and
Dashboard pages, we apply explicit tuple unpacking at assignment:
  ```python
  col_x, col_y = st.columns(2)
  with col_x:
      # Render independent axis selector
  ```
  This allows each column object to act as an isolated context manager, eliminating runtime layout
  exceptions.
## 🗄️ 11. Scalar Result Resolution & Unpacking in SQLite3

### Question: On the Analytics page, why did direct `int(row)` conversion fail on `fetchone()` results, and how did you enforce type safety?
* **Defense**: Python's native `sqlite3` driver always returns a tuple or `sqlite3.Row` object when
`.fetchone()` is called, even when the SQL query selects a single scalar column (e.g.,
`SELECT id FROM sensors`). Calling `int(row)` directly on a tuple raises a critical
`TypeError: int() argument must be a string... not 'tuple'`.
* **Technical Justification**: To ensure safe transactional loading of historical telemetry, we
explicitly extract the first scalar element via structural indexing before type conversion:
  ```python
  row = conn.execute("SELECT id FROM sensors...").fetchone()
  sensor_id = int(row[0]) if row else 1
  ```
  This defensive guard guarantees that the value passed to subsequent `city_stats` queries is a
  valid integer, preventing statistical page crashes.

---

## 🎨 12. Geometric Axis Precision Control in Plotly Dark Templates

### Question: How did you eliminate distorted floating-point labels on monitoring chart axes?
* **Defense**: Predictive linear regression calculations and simulated sensor drift can produce
unrounded floating-point values (e.g., `32.19999999999574`), which stretch chart margins and
degrade dashboard readability.
* **Technical Justification**: We configured explicit tick-formatting masks directly within the
Plotly multi-panel layout definitions:
  ```python
  fig.update_yaxes(tickformat=".1f", showgrid=True, gridcolor="#2d3748")
  fig_ml.update_xaxes(tickformat=".1f")
  ```
  Enforcing the `.1f` format directive rounds axis tick labels to a single fixed decimal place
  (e.g., `32.2`), keeping the dark-mode UI crisp, compact, and focused on actionable metrics.

---

## 🚀 13. Continuous Integration (CI) Automation & LLMOps Validation

### Question: How do you guarantee that code changes do not break PEP 8 formatting, unit tests, or AI assistant guardrails before merging?
* **Defense**: In the Smart City architecture, DevOps automation is treated as a first-class asset.
We eliminated manual verification bottlenecks by implementing an automated Continuous Integration
(CI) pipeline via GitHub Actions (`.github/workflows/ci.yml` and `.github/workflows/mlops.yml`).
* **Technical Justification**: On every `git push` or pull request targeting `main`, the CI runner
isolates the environment and sequentially enforces strict quality gates governed by `pyproject.toml`:
  1. **Style & Static Analysis Gate (Ruff):** Enforces line-length boundaries (`E501`),
  alphabetical import sorting (`I001`), and clean static analysis rules. Any violation fails the
  build immediately.
  2. **Logic & Regression Gate (Pytest):** The offline test suite runs via `uv run pytest -v`
  (covering bootstrap initialization, seeder idempotency, missing-sensor
  guards, session security, and native Python AI/RAG guardrails in `test_ai_guardrails.py`).
  3. **Semantic & Guardrail AI Gate (LLMOps):** `ai_tests/run_guardrail_eval.py` runs 7 local
  deterministic RAG/grounding checks without credentials, while `ai_tests/run_llm_eval.py`
  evaluates **7 live Cluj-Napoca urban scenarios** using `qwen/qwen3.8-27b -> gemini-3.5-flash` for generation
  and compact JSON judging (`max_tokens=40`, stripping any `<think>` blocks). It uses one worker,
  adaptive header-driven pacing (x-ratelimit-remaining-tokens / x-ratelimit-reset-tokens, 1.5s fallback), and up to three adaptive retries for transient 429/empty payload
  errors, with authorized station thresholds. A Pydantic
  `LlmJudgeVerdict` scores each response and contributes to the reported average faithfulness.

---

## 🌐 14. Continuous Delivery (CD) & Cloud Git Webhooks

### Question: How does approved code transition from local development to live cloud production, and how are secrets protected across this pipeline?
* **Defense**: Deployment is fully automated following Continuous Delivery (CD) principles,
eliminating *Environment Drift* and billing-related service interruptions.
* **Technical Justification**:
  1. **Deployment & Idempotent Bootstrap:** Streamlit Community Cloud executes `main.py` as its
  entry point. On startup, the application initializes the SQLite schema and runs the safe
  idempotent seeder: the 7 Cluj-Napoca stations are inserted only when operational tables are
  empty, preserving any existing custom thresholds or telemetry. Dashboard, Settings, and
  Analytics display controlled warnings if an active sensor cannot be resolved, preventing
  `AttributeError` crashes.
  2. **Zero-Trust Secrets:** To prevent exposing API keys (`GROQ_API_KEY`, `GROQ_API_KEYS`,
  `GEMINI_API_KEY`) or
  credentials in public Git history, secrets are stored encrypted in the cloud management console.
  At runtime, the platform injects them directly into protected process memory as TOML variables
  exposed via `st.secrets`, ensuring continuous uptime for technical evaluators and strict
  compliance with OWASP Top 10 security standards.


---

## 📊 15. Dual-Layer Evaluation Metrics: Deterministic Compliance vs. LLM Faithfulness

### Question: In the Systemic Quality & AI Compliance Control Center, what is the difference between Scenario Compliance and Average Faithfulness, and how are they computed without hardcoding?
* **Defense**: A single pass/fail metric cannot distinguish between a response that merely mentions the right sensor keyword and a response that stays strictly faithful to retrieved telemetry without extrapolating unsupported causes. Therefore, `ai_tests/run_llm_eval.py` separates **deterministic keyword/section compliance** from **semantic faithfulness scoring**.
* **Technical Justification**:
  1. **Deterministic Scenario Compliance (`matched_compliance`):** A local Python evaluator isolates the `Assessment` section of each generated response, normalizes diacritics, and verifies whole-word presence of the scenario's focal telemetry term. The percentage in `success_rate_percentage` is computed dynamically at runtime from live pass counts (`passed / total * 100`).
  2. **Semantic Faithfulness Score (`avg_faithfulness_score`):** Each response is independently audited by `LlmJudgeEvaluator` using the `judge_instructions` contract in `app/ai/prompts.yaml`. Comparing the output strictly against the scenario's telemetry values and authorized `SensorThresholdMatrix`, the judge returns a Pydantic-validated `LlmJudgeVerdict` (`faithfulness: int` on a `1–5` scale, `passed: bool`). A `5/5` score requires strict adherence to supplied readings and units, whereas `4/5` reflects accurate numeric grounding with minor interpretive phrasing.
  3. **Dynamic Report Aggregation (`LlmEvalSummary`):** At the end of each run, `_write_report()` computes the arithmetic mean of all valid integer verdicts (`sum(scores) / len(scores)`) and writes `ai_tests/llm_eval_report.json`. On `pages/4_About.py`, `LlmEvalSummary` parses this artifact dynamically at page load or after an on-demand live verification run, keeping UI metrics synchronized with actual execution outcomes.

---

## Production Incident Post-Mortems & Engineering Lessons (Nov 2026)

1. **Model Deprecation & Live Catalog Verification:**
   - **Symptom:** Legacy model IDs (`gemini-2.5-flash` returning `HTTP 404 NOT_FOUND`; legacy Llama IDs removed from active Groq tiers) caused avoidable retry loops.
   - **Fix:** Audited live provider catalogs (`models.list()`) and standardized on active 2026 models: `qwen/qwen3.8-27b` (with `openai/gpt-oss-20b` and `openai/gpt-oss-120b` fallbacks) on Groq, failing over to `gemini-3.5-flash` (`gemini-3.5-flash-lite`, `gemini-3.6-flash`, `gemini-3.8-flash`) via the native `google-genai` SDK.
2. **Reasoning-Model Token Starvation (`max_tokens` Budgeting):**
   - **Symptom:** Reasoning models (`qwen/qwen3.8-27b`, `openai/gpt-oss-20b`, `gemini-3.5-flash`) consume initial completion tokens on internal `<think>` chains before outputting JSON. Tight budgets (`max_tokens=256` or `800`) truncated payloads mid-JSON.
   - **Fix:** Configured `max_tokens=2048` in `GroqProvider`, `MAX_COMPLETION_TOKENS=2500` in `GeminiOpenAiProvider`, and `max_tokens=1500` in `run_llm_eval.py`, stripping `<think>...</think>` blocks prior to validation.
3. **Structured JSON Contract Enforcement (`output_format` + `response_schema`):**
   - **Symptom:** Omitting `output_format` when `system_prompt_only=True` caused generic JSON mode to emit ad-hoc telemetry keys instead of the `{"answer": "..."}` schema.
   - **Fix:** Unified `system_instructions`, `output_format`, and `anti_hallucination_rules` in `PromptTemplateLoader` and enforced Pydantic `response_schema` constrained decoding in `GeminiOpenAiProvider`.
4. **Zero-Downtime Multi-Key Rotation Under `HTTP 429`:**
   - **Symptom:** Burst evaluation workloads triggered free-tier `HTTP 429` rate limits on single keys.
   - **Fix:** Set `max_retries=0` on underlying SDK clients and delegated retry/rotation to `GroqProvider` (`GROQ_API_KEY_1` <-> `GROQ_API_KEY_2`) and `FallbackAiProviderRouter` (-> Gemini), sustaining 100.0% scenario compliance (7/7) and 99/99 Pytest assertions.
5. **In-Process Test Isolation in Deployed Cloud Runtimes (`st.secrets`):**
   - **Symptom:** Running `pytest` inside a deployed Streamlit Cloud container failed 3 authentication unit tests because live `st.secrets` (`PLATFORM_ADMIN_USER`, `PLATFORM_ADMIN_PASS`) overrode `monkeypatch.setenv(...)`.
   - **Fix:** Added an `autouse=True` fixture in `ai_tests/test_app_bootstrap.py` (`monkeypatch.setitem(st.__dict__, "secrets", {})`) so unit tests remain hermetically isolated both locally and when triggered live from the Cloud UI.
