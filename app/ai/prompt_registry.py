"""Validated loading and rendering for the structured AI prompt registry."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import ClassVar, Final

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


class PromptRegistryError(RuntimeError):
    """Raised when a prompt registry file cannot be parsed as YAML."""


class PromptTemplateConfig(BaseModel):
    """Immutable schema for one versioned prompt template."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(min_length=1)
    system_instructions: str = Field(min_length=1)
    user_template: str = Field(min_length=1)
    output_format: str = Field(min_length=1)
    anti_hallucination_rules: tuple[str, ...] = Field(min_length=1)
    judge_instructions: str = Field(min_length=1)


class PromptRegistryLoader:
    """Load, validate, cache, and render prompt template configurations."""

    DEFAULT_PROMPT_PATH: Final[Path] = PROJECT_ROOT / "app" / "ai" / "prompts.yaml"
    _CACHE: ClassVar[dict[Path, PromptTemplateConfig]] = {}
    _CACHE_LOCK: ClassVar[threading.RLock] = threading.RLock()

    @classmethod
    def load(cls, prompt_path: str | Path | None = None) -> PromptTemplateConfig:
        """Return a validated cached YAML prompt template."""
        requested_path = Path(prompt_path) if prompt_path is not None else cls.DEFAULT_PROMPT_PATH
        if not requested_path.is_absolute():
            requested_path = PROJECT_ROOT / requested_path
        resolved_path = requested_path.resolve()

        if resolved_path.suffix.lower() not in {".yaml", ".yml"}:
            raise PromptRegistryError("Prompt configuration must use a YAML file.")

        with cls._CACHE_LOCK:
            cached_config = cls._CACHE.get(resolved_path)
            if cached_config is not None:
                return cached_config

            if not resolved_path.is_file():
                raise FileNotFoundError(f"Prompt configuration not found: {resolved_path}")
            try:
                document = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise PromptRegistryError(
                    f"Prompt configuration is not valid YAML: {resolved_path}"
                ) from exc
            config = PromptTemplateConfig.model_validate(document)

            cls._CACHE[resolved_path] = config
            return config

    @staticmethod
    def render_user_template(
        config: PromptTemplateConfig,
        values: dict[str, str],
    ) -> str:
        """Replace only the registered double-brace placeholders."""
        rendered_template = config.user_template
        for name, value in values.items():
            rendered_template = rendered_template.replace(f"{{{{{name}}}}}", value)
        return rendered_template

    @staticmethod
    def build_system_prompt(config: PromptTemplateConfig) -> str:
        """Compose system instructions with the output and grounding contracts."""
        anti_hallucination_rules = "\n".join(
            f"- {rule}" for rule in config.anti_hallucination_rules
        )
        return (
            f"{config.system_instructions.strip()}\n\n"
            f"OUTPUT FORMAT\n{config.output_format.strip()}\n\n"
            f"ANTI-HALLUCINATION RULES\n{anti_hallucination_rules}"
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached configurations, primarily for isolated tests and reloads."""
        with cls._CACHE_LOCK:
            cls._CACHE.clear()
