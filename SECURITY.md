# 🛡️ Security Policy & Hardening Guide: Smart City Cluj IoT Core

This document details the defensive layers implemented across the platform to protect the integrity of Cluj-Napoca IoT telemetry and mitigate API-level cyberattacks.

---

## 1. Defensive Layers & Defense-in-Depth Model

The platform enforces a multi-layered local security architecture:

```text
  Sensor Data (Raw SQLite)
         │
         ▼
[Local Interface Isolation] ─► CityRepository securely queries DATABASE_PATH
         │
         ▼
[Structured Prompt Syntax] ──► Isolates payloads inside delimited tags in the YAML prompt registry
         │
         ▼
[Language Matrix Sync] ──────► Aligns responses with the TranslationProvider component
         │
         ▼
[AI Reasoning Regex Filter] ─► Automatically strips <think> tags returned by Groq API
```

## 2. Credential & Urban Secret Management

- **Private Key Isolation**: All sensitive credentials (`GROQ_API_KEY*`, `GEMINI_API_KEY`, `SESSION_SECURITY_TOKEN`) and administrative accounts are completely decoupled from source code and stored exclusively inside the local `.env` file (or Streamlit Cloud Secrets).
- **Argon2id Password Hashing:** Administrative passwords are stored as Argon2id encoded hashes and are never stored in plaintext.
- **Production Secret Leak Prevention**: The `.env` file is explicitly excluded via `.gitignore` and `.dockerignore`. No production keys or certificates ever enter public version control (GitHub).
- **Container Hardening**: Running the platform inside a Docker container isolates application processes from the Host OS. All non-essential ports are closed, exposing solely the standard communication port `8501` protected by Docker network directives.

## 3. Vulnerability Disclosure

If you discover a security vulnerability in the data evaluation logic or port exposure, please **do not open a public Issue**. Submit a detailed report directly to the technical security contact listed on the personal portfolio.

## 4. Protection Against Path Traversal Attacks on Audit Logs

* **Architectural Question**: Since the `security_alerts.log` audit ledger is dynamically read and rendered in real time on the dashboard UI, how is the system protected against a malicious actor attempting to read arbitrary system files on the server (e.g., `/etc/passwd`)?
* **Implemented Safeguard**: The path to the audit log file is defined immutably at the module level using `Path("security_alerts.log").resolve()`. The code in `main.py` and `ai_interface.py` never accepts external parameters or arguments (from query strings or user text inputs) to alter the file descriptor location. File reads execute strictly against the pre-bound path on disk, structurally blocking any directory traversal payloads (`../`) and securing data rendering in the UI.
