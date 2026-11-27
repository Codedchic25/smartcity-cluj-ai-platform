"""Automated checks ensuring Markdown documentation stays aligned with code."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_markdown_docs_reflect_latest_architecture_and_test_count() -> None:
    """Verify key architecture updates and test count appear in Markdown docs."""
    architecture = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    testing = (ROOT / "TESTING.md").read_text(encoding="utf-8")

    for keyword in (
        "system_prompt_only",
        "_candidate_models",
        "is_provider_failure",
        "Sensor",
        "Correction Retry",
        "max_scenarios",
    ):
        assert keyword in architecture, f"Missing '{keyword}' in ARCHITECTURE.md"
        assert keyword in changelog, f"Missing '{keyword}' in CHANGELOG.md"

    assert "99" in testing, "TESTING.md must document the 99 passing tests"
    assert "99" in architecture, "ARCHITECTURE.md must document the 99 passing tests"


def test_root_markdown_files_are_in_english() -> None:
    """Ensure root Markdown files contain no leftover Romanian headings."""
    romanian_markers = (
        "Aplica?ie Opera?ional?",
        "Jurnal de Modific?ri",
        "Modific?ri efectuate",
    )
    for md_file in ROOT.glob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        for marker in romanian_markers:
            assert marker not in content, f"{md_file.name} still contains Romanian text: {marker}"
