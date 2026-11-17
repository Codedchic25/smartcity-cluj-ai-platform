"""Run native Python RAG and output-guardrail checks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.rag_pipeline import GUARDRAIL_REPORT_PATH, write_guardrail_report  # noqa: E402


def main() -> int:
    """Write guardrail metrics and return nonzero when any check fails."""
    report = write_guardrail_report()
    stats = report["stats"]
    total = stats["successes"] + stats["failures"] + stats["errors"]
    print(f"Native RAG guardrails: {stats['successes']}/{total} checks passed.")
    print(f"Guardrail report: {GUARDRAIL_REPORT_PATH}")
    return 0 if total == 7 and stats["successes"] == 7 else 1


if __name__ == "__main__":
    raise SystemExit(main())
