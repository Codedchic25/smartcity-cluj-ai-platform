# 📝 Jurnal de Modificări: Smart City Cluj IoT Core

Toate modificările notabile aduse acestui proiect sunt documentate în acest fișier.
Acest depozit respectă standardele industriale de Versiune Semantică (`MAJOR.MINOR.PATCH`).

---

## [Unreleased]

### ✨ Adăugat
- **Bootstrap SQLite la pornire:** După crearea schemei, `main.py` apelează seederul
  idempotent pentru a popula bazele noi cu cele 7 stații Cluj-Napoca, pragurile și
  telemetria demo. Datele existente din tabelele operaționale nu sunt suprascrise.
- **Protecție pentru selecția stației:** Dashboard, Settings și Analytics afișează un
  avertisment dacă senzorul activ lipsește, în loc să ridice `AttributeError`.
- **Regresii de bootstrap:** `ai_tests/test_app_bootstrap.py` verifică inițializarea
  bazei goale, păstrarea setărilor la repornire și comportamentul paginilor pentru
  selecția absentă. Raportul curent este generat la fiecare rulare Pytest.
- **Aliniere multilingvă LLM:** `GroqProvider` generează corpul răspunsului în limba
  activă (`RO`, `EN`, `IT`, `ES`, `HU`) și formatează secțiunile Assessment,
  Recommended Actions și Data Limitations cu etichete Markdown îngroșate pe linii separate.
- **Fallback pentru pornirea la rece în Dashboard:** La pornirea normală, `main.py`
  populează idempotent o bază goală; dacă Dashboard-ul nu găsește stații, afișează
  instrucțiuni pentru seed manual, iar pentru mai puțin de două citiri folosește
  istoric sintetic.

---

## [v2.3.0] — 2026-10-06

### ✨ Adăugat
- **Asistent AI Context-Aware cu Traduceri Directe:** S-a dezvoltat un mecanism de aliniere
  a asistentului inteligent cu dicționarele native ale aplicației, apelând direct registrul
  privat `_REGISTRY` din `translations.py`. Acest lucru asigură o localizare lingvistică
  fluidă în RO, EN, IT, ES și HU, eliminând erorile de tip `AttributeError`.
- **Framework de Evaluare Programatică în Python (`run_llm_eval.py`):** S-a implementat o suită
  nativă și izolată de testare automată în Python care evaluează automat performanța asistentului
  cognitiv pe baza a 7 scenarii urbane critice din Cluj-Napoca. Rezultatele sunt exportate
  automat într-un raport metric structurat JSON (`llm_eval_report.json`).
- **Autentificare Criptografică în Mediu Izolat:** S-a securizat clasa `AuthenticationService`
  prin adăugarea unei clauze defensive rigide (`RuntimeError`) care blochează pornirea platformei
  dacă acreditările lipsesc din `.env`. Verificarea parolei combină acum amprentarea MD5 cu
  evaluarea în timp constant prin `hmac.compare_digest`, oferind o rezistență totală împotriva
  atacurilor de tip brute-force și timing side-channel.
- **Orchestrare cu Ruff & UV:** Întregul proiect a fost optimizat și aliniat la standardele
  PEP 8 prin integrarea linterului ultra-rapid `ruff` orchestrat nativ prin managerul de
  pachete modern `uv`.

### 🐛 Corectat
- **Restaurarea Suitei de Teste:** S-au separat manual fișierele
  din folderul `ai_tests/`. Codul pentru `test_alerts_offline.py` a fost curățat de dublurile
  LLM și adaptat să interogheze direct metodele atomice expuse de clasa `CityRepository` din
  `  main.py`. Testele din acea suită rulau cu succes în terminal.
- **Aliniere Path-uri Bază de Date:** S-a corectat constructorul clasei `DatabaseConfig` din
  `seed_db.py` prin adăugarea metodei `.resolve()`. Această rectificare elimină variabila
  neutilizată raportată de Ruff (eroarea F841) și garantează că scriptul de populare scrie
  exact în fișierul citit de Streamlit, prevenind duplicarea bazelor de date pe disc.
- **Bypass Blocaj Resurse Windows (EBUSY):** S-a documentat procedura de eliberare forțată a
  proceselor Python fantomă din PowerShell (`Stop-Process -Name "python" -Force`), rezolvând
  blocajele sistemului de operare la ștergerea sau înlocuirea fișierului `app.db`.
- **Aliniere Unități de Măsură PM2.5:** S-au modificat string-urile explicative și metricile
  din paginile `1_Dashboard.py` și `3_Analytics.py`, schimbând unitatea eronată `ppm` cu
  standardul academic corect **`µg/m³`**, în perfectă concordanță cu definiția din dicționar.
