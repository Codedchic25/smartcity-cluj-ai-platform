"""Google Gemini provider and resilient Groq-first completion routing."""

from __future__ import annotations

import inspect
import logging
import re
import time
from collections.abc import Callable, Mapping
from typing import Any, Final, Protocol

from openai import APIStatusError, OpenAI, OpenAIError
from pydantic import ValidationError

try:
    import httpx

    _HTTPX_ERRORS: tuple[type[BaseException], ...] = (httpx.HTTPError,)
except ImportError:
    _HTTPX_ERRORS = ()

try:
    import groq

    _GROQ_ERRORS: tuple[type[BaseException], ...] = (groq.GroqError,)
except ImportError:
    _GROQ_ERRORS = ()

from app.ai.groq_provider import GroqProvider
from app.ai.rag_pipeline import validate_grounded_answer

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

_EXPLICIT_FAILURE_PREFIXES: Final[tuple[str, ...]] = (
    "❌",
    "⚠️",
    "generic operational alert:",
    "subsystem failure:",
    "dependency constraint:",
    "ingestion gateway failure:",
    "gemini gateway failure:",
    "multi-provider fallback failed",
    "error:",
    "api error:",
    "apierror:",
    "rate limit error:",
    "ratelimiterror:",
    "timeout error:",
    "timeouterror:",
    "connection error:",
    "connectionerror:",
    "http error:",
    "httperror:",
    "gateway failure:",
    "service unavailable:",
    "bad gateway:",
)

_FAILURE_MESSAGE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bhttp\s+(?:status\s+)?(?:4\d{2}|5\d{2})\b", re.IGNORECASE),
    re.compile(r"\bstatus(?:_code|\s+code)?\s*[:=]\s*(?:4\d{2}|5\d{2})\b", re.IGNORECASE),
    re.compile(
        r"(?:^|\b)(?:429|500|502|503|504)\s*[:\-]?\s*"
        r"(?:too many requests|rate[_ ]limit(?:_exceeded|\s+exceeded)?|"
        r"internal server error|bad gateway|service unavailable|gateway timeout)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:rate_limit_exceeded|too many requests|quota exceeded|"
        r"daily quota exceeded|monthly quota exceeded)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:request timed out|connection timed out|read timed out|"
        r"gateway timeout|timed out after|timeout expired)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:apitimeouterror|apiconnectionerror|ratelimiterror|"
        r"internalservererror|apistatuserror|apierror|groqerror|openaierror)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:gateway failure|upstream request failed|service unavailable|"
        r"internal server error|bad gateway|returned an empty completion)\b",
        re.IGNORECASE,
    ),
)


def _extract_status_code(error: Any) -> int | None:
    """Extract an integer HTTP status code from an exception or response object."""
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    return (
        status_code if isinstance(status_code, int) and not isinstance(status_code, bool) else None
    )


def _exception_error_metadata(error: Exception) -> dict[str, Any]:
    """Build structured error metadata from a raised exception."""
    status_code = _extract_status_code(error)
    error_name = type(error).__name__
    lowered_message = str(error).casefold()
    if status_code == 429 or "rate_limit" in lowered_message or "rate limit" in lowered_message:
        category = "rate_limit"
    elif (
        isinstance(error, TimeoutError)
        or "timeout" in error_name.casefold()
        or ("timed out" in lowered_message)
    ):
        category = "timeout"
    elif status_code is not None and 500 <= status_code <= 599:
        category = "server_error"
    elif status_code is not None:
        category = "http_error"
    elif isinstance(error, (ValidationError, ValueError)):
        category = "validation"
    else:
        category = "api_error"
    return {
        "category": category,
        "error_type": error_name,
        "status_code": status_code,
        "message": str(error)[:300],
    }


def _is_recoverable_provider_exception(error: Exception) -> bool:
    """Return True for API, HTTP, timeout, network, or runtime provider failures."""
    if isinstance(
        error,
        (
            APIStatusError,
            OpenAIError,
            TimeoutError,
            ConnectionError,
            OSError,
            ValidationError,
            ValueError,
            RuntimeError,
            *_HTTPX_ERRORS,
            *_GROQ_ERRORS,
        ),
    ):
        return True
    if _extract_status_code(error) is not None:
        return True
    return (
        getattr(error, "error_code", None) is not None or getattr(error, "code", None) is not None
    )


def is_provider_failure(
    response: Any,
    telemetry: Mapping[str, Any] | None = None,
    provider: Any = None,
) -> bool:
    """Classify provider failures via structured signals, status codes, or messages."""
    if isinstance(telemetry, Mapping):
        if telemetry.get("failed") is True or telemetry.get("is_error") is True:
            return True
        if telemetry.get("error") or telemetry.get("error_type") or telemetry.get("error_code"):
            return True
        telemetry_status = telemetry.get("status_code")
        if (
            isinstance(telemetry_status, int)
            and not isinstance(telemetry_status, bool)
            and telemetry_status >= 400
        ):
            return True
        nested_telemetry = telemetry.get("telemetry")
        if isinstance(nested_telemetry, Mapping):
            if is_provider_failure("", telemetry=nested_telemetry):
                return True
        elif getattr(nested_telemetry, "error", None):
            return True

    if provider is not None:
        if getattr(provider, "last_error", None):
            return True
        provider_status = getattr(provider, "last_status_code", None)
        if (
            isinstance(provider_status, int)
            and not isinstance(provider_status, bool)
            and provider_status >= 400
        ):
            return True

    if isinstance(response, Exception):
        return True
    if isinstance(response, Mapping):
        return (
            response.get("failed") is True
            or response.get("is_error") is True
            or bool(
                response.get("error") or response.get("error_type") or response.get("error_code")
            )
            or ((_extract_status_code(response) or 0) >= 400)
        )
    obj_status = _extract_status_code(response)
    if obj_status is not None and obj_status >= 400:
        return True
    if getattr(response, "error", None) or getattr(response, "is_error", False) is True:
        return True
    if not isinstance(response, str):
        return True

    stripped = response.strip()
    if not stripped:
        return True
    lowered = stripped.casefold()
    if lowered.startswith(_EXPLICIT_FAILURE_PREFIXES):
        return True
    return any(pattern.search(stripped) for pattern in _FAILURE_MESSAGE_PATTERNS)


def _accepts_keyword_arg(func: Callable[..., Any], name: str) -> bool:
    """Return True when the callable accepts the given keyword argument."""
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or (
            parameter.name == name
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        )
        for parameter in parameters
    )


def call_with_prompt_options(
    func: Callable[..., Any],
    prompt_context: str,
    *,
    system_prompt: str | None = None,
    system_prompt_only: bool = False,
) -> Any:
    """Invoke a provider method while forwarding supported prompt options."""
    kwargs: dict[str, Any] = {}
    if _accepts_keyword_arg(func, "system_prompt") or system_prompt:
        kwargs["system_prompt"] = system_prompt
    if _accepts_keyword_arg(func, "system_prompt_only") or system_prompt_only:
        kwargs["system_prompt_only"] = system_prompt_only
    return func(prompt_context, **kwargs)


class CompletionProvider(Protocol):
    """Completion provider contract shared by primary and fallback clients."""

    @property
    def configured_model(self) -> str:
        """Return the configured model identifier."""
        ...

    def generate_completion(
        self,
        system_prompt_context: str,
        system_prompt: str | None = None,
        *,
        system_prompt_only: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        raw_response: bool = False,
    ) -> str:
        """Generate one response from supplied context."""
        ...


class GeminiOpenAiProvider:
    """Use Google's OpenAI-compatible Gemini endpoint with key/model fallback."""

    BASE_URL: Final[str] = "https://generativelanguage.googleapis.com/v1beta/openai/"
    FALLBACK_MODELS: Final[tuple[str, ...]] = (
        "gemini-2.5-flash-lite",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
    )
    REQUEST_TIMEOUT_SECONDS: Final[float] = 30.0
    MAX_COMPLETION_TOKENS: Final[int] = 800

    def __init__(
        self,
        api_key_env_var: str = "GEMINI_API_KEY",
        model_target: str = "gemini-2.5-flash-lite",
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        """Collect Gemini credentials without exposing them in provider state."""
        self._api_keys = GroqProvider._collect_api_keys(api_key_env_var)
        self._model_target = model_target
        self._client_factory = client_factory or OpenAI
        self._last_token_usage: dict[str, int | None] | None = None
        self._last_error: dict[str, Any] | None = None

    @property
    def last_error(self) -> dict[str, Any] | None:
        """Return structured metadata for the latest failed completion attempt."""
        return dict(self._last_error) if self._last_error is not None else None

    @property
    def configured_model(self) -> str:
        """Return the primary configured Gemini model."""
        return self._model_target

    def _candidate_models(self) -> list[str]:
        """Return unique Gemini models with the configured model tried first."""
        return list(dict.fromkeys((self._model_target, *self.FALLBACK_MODELS)))

    @staticmethod
    def _usage_counts(completion: Any) -> dict[str, int]:
        """Read token counts from an OpenAI-compatible response."""
        usage = getattr(completion, "usage", None)
        if usage is None:
            return {}

        counts: dict[str, int] = {}
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(field) if isinstance(usage, dict) else getattr(usage, field, None)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                counts[field] = value
        if "total_tokens" not in counts and {
            "prompt_tokens",
            "completion_tokens",
        }.issubset(counts):
            counts["total_tokens"] = counts["prompt_tokens"] + counts["completion_tokens"]
        return counts

    def _record_completion_usage(self, completion: Any) -> None:
        """Accumulate usage from each completed Gemini model attempt."""
        counts = self._usage_counts(completion)
        if not counts:
            return
        if self._last_token_usage is None:
            self._last_token_usage = {}
        for field, value in counts.items():
            self._last_token_usage[field] = (self._last_token_usage.get(field, 0) or 0) + value

    @staticmethod
    def _error_description(error: Exception) -> str:
        """Describe API errors without including request credentials or payloads."""
        if isinstance(error, APIStatusError):
            return f"HTTP {error.status_code}"
        if isinstance(error, OpenAIError):
            return type(error).__name__
        if isinstance(error, (ValidationError, ValueError, TypeError)):
            return str(error)[:300]
        return type(error).__name__

    def _request_completion(
        self,
        client: Any,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Request structured JSON, retrying without JSON mode if Gemini rejects it."""
        request = {
            "model": model,
            "messages": messages,
            "temperature": 0.2 if temperature is None else temperature,
            "max_tokens": (self.MAX_COMPLETION_TOKENS if max_tokens is None else max_tokens),
        }
        try:
            return client.chat.completions.create(
                **request,
                response_format={"type": "json_object"},
            )
        except APIStatusError as exc:
            if exc.status_code != 400:
                raise
            LOGGER.info("Gemini rejected JSON response mode; retrying without it.")
            return client.chat.completions.create(**request)

    def generate_completion_with_telemetry(
        self,
        system_prompt_context: str,
        system_prompt: str | None = None,
        system_prompt_only: bool = False,
    ) -> tuple[str, dict[str, int | float | bool | dict[str, Any] | None]]:
        """Return the grounded completion and provider-reported usage."""
        started_at = time.perf_counter()
        response = self.generate_completion(
            system_prompt_context,
            system_prompt=system_prompt,
            system_prompt_only=system_prompt_only,
        )
        telemetry: dict[str, int | float | bool | dict[str, Any] | None] = {
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "prompt_tokens": (self._last_token_usage or {}).get("prompt_tokens"),
            "completion_tokens": (self._last_token_usage or {}).get("completion_tokens"),
            "total_tokens": (self._last_token_usage or {}).get("total_tokens"),
            "system_prompt_only": system_prompt_only,
        }
        if self._last_error is not None:
            telemetry["error"] = dict(self._last_error)
        return response, telemetry

    def generate_completion(
        self,
        system_prompt_context: str,
        system_prompt: str | None = None,
        *,
        system_prompt_only: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        raw_response: bool = False,
    ) -> str:
        """Generate a grounded answer with bounded model and credential fallback."""
        self._last_token_usage = None
        self._last_error = None
        if not self._api_keys:
            self._last_error = {
                "category": "configuration",
                "error_type": "MissingApiKey",
                "status_code": None,
                "message": "GEMINI_API_KEY is not configured.",
            }
            return "Generic Operational Alert: GEMINI_API_KEY is not configured."

        language = GroqProvider._resolve_target_language(system_prompt_context)
        default_system_prompt = (
            "You are an evidence-bound Smart City telemetry adviser. Return one "
            "JSON object with string field 'answer'. Follow the requested format "
            f"and write in {language}. Do not invent values or causes."
        )
        if system_prompt and system_prompt.strip():
            resolved_system_prompt = (
                system_prompt.strip()
                if system_prompt_only
                else f"{system_prompt.strip()}\n\n{default_system_prompt}"
            )
        else:
            resolved_system_prompt = default_system_prompt

        messages = [
            {"role": "system", "content": resolved_system_prompt},
            {"role": "user", "content": system_prompt_context},
        ]
        retrieved_context = GroqProvider._retrieved_context(system_prompt_context)
        last_error: Exception | None = None

        for model in self._candidate_models():
            model_unavailable = False
            for api_key in self._api_keys:
                try:
                    client = self._client_factory(
                        api_key=api_key,
                        base_url=self.BASE_URL,
                        max_retries=0,
                        timeout=self.REQUEST_TIMEOUT_SECONDS,
                    )
                    completion = self._request_completion(
                        client,
                        model,
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    self._record_completion_usage(completion)
                    choices = getattr(completion, "choices", None)
                    message = getattr(choices[0], "message", None) if choices else None
                    content = getattr(message, "content", None)
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("Gemini returned an empty completion.")

                    if raw_response:
                        return content.strip()
                    answer = validate_grounded_answer(content.strip(), retrieved_context)
                    return GroqProvider._format_answer_sections(answer)
                except APIStatusError as exc:
                    last_error = exc
                    if exc.status_code in (400, 404):
                        model_unavailable = True
                        break
                except OpenAIError as exc:
                    last_error = exc
                except (ValidationError, ValueError, TypeError) as exc:
                    last_error = exc
                    model_unavailable = True
                    break

            if model_unavailable:
                LOGGER.warning(
                    "Gemini model '%s' was unavailable or returned invalid output.",
                    model,
                )

        reason = self._error_description(last_error) if last_error else "no completion returned"
        self._last_error = (
            _exception_error_metadata(last_error)
            if last_error is not None
            else {
                "category": "api_error",
                "error_type": "GatewayError",
                "status_code": None,
                "message": reason,
            }
        )
        return f"❌ Gemini gateway failure: {reason}."


class FallbackAiProviderRouter:
    """Route requests to Gemini when Groq exhausts its configured fallback path."""

    def __init__(
        self,
        groq_provider: CompletionProvider | None = None,
        gemini_provider: CompletionProvider | None = None,
        *,
        groq_model_target: str = "openai/gpt-oss-20b",
        gemini_model_target: str = "gemini-2.5-flash-lite",
    ) -> None:
        """Initialize injectable providers for production use and deterministic tests."""
        self._groq: CompletionProvider = groq_provider or GroqProvider(
            model_target=groq_model_target
        )
        self._gemini: CompletionProvider = gemini_provider or GeminiOpenAiProvider(
            model_target=gemini_model_target
        )

    @property
    def configured_model(self) -> str:
        """Describe the configured primary-to-fallback model route."""
        return f"{self._groq.configured_model} -> {self._gemini.configured_model}"

    @property
    def rate_limit_headers(self) -> dict[str, str]:
        """Expose rate-limit headers reported by the primary Groq provider."""
        headers = getattr(self._groq, "rate_limit_headers", {})
        return dict(headers) if isinstance(headers, dict) else {}

    @staticmethod
    def _is_failure(
        response: Any,
        telemetry: Mapping[str, Any] | None = None,
        provider: Any = None,
    ) -> bool:
        """Recognize failed provider outcomes via exceptions, metadata, status, or text."""
        return is_provider_failure(response, telemetry=telemetry, provider=provider)

    @staticmethod
    def _format_exception_failure(provider_label: str, error: Exception) -> str:
        """Format a recoverable provider exception into a bounded failure string."""
        status_code = _extract_status_code(error)
        error_name = type(error).__name__
        if status_code is not None:
            return f"❌ {provider_label} gateway failure: HTTP {status_code} ({error_name})."
        if isinstance(error, TimeoutError) or "timeout" in error_name.casefold():
            return f"❌ {provider_label} gateway failure: request timed out ({error_name})."
        detail = str(error).strip()[:200]
        return (
            f"❌ {provider_label} gateway failure: {error_name}: {detail}."
            if detail
            else f"❌ {provider_label} gateway failure: {error_name}."
        )

    @staticmethod
    def _sum_usage(
        primary: dict[str, int | float | None],
        fallback: dict[str, int | float | None],
    ) -> dict[str, Any]:
        """Sum usage reported by either provider, preserving unavailable values."""
        result: dict[str, Any] = {}
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            values = [
                value for value in (primary.get(field), fallback.get(field)) if value is not None
            ]
            result[field] = sum(values) if values else None
        return result

    def _invoke_completion_provider(
        self,
        provider: CompletionProvider,
        provider_label: str,
        system_prompt_context: str,
        *,
        system_prompt: str | None,
        system_prompt_only: bool,
        request_options: dict[str, Any],
    ) -> tuple[str, bool]:
        """Invoke a completion provider and classify returned or raised failures."""
        try:
            response = provider.generate_completion(
                system_prompt_context,
                system_prompt=system_prompt,
                system_prompt_only=system_prompt_only,
                **request_options,
            )
        except Exception as exc:
            if not _is_recoverable_provider_exception(exc):
                raise
            return self._format_exception_failure(provider_label, exc), True
        return response, self._is_failure(response, provider=provider)

    def _invoke_telemetry_provider(
        self,
        provider: CompletionProvider,
        provider_label: str,
        system_prompt_context: str,
        *,
        system_prompt: str | None,
        system_prompt_only: bool,
    ) -> tuple[str, dict[str, Any], bool]:
        """Invoke a telemetry-aware or legacy provider and classify failures."""
        telemetry: dict[str, Any] = {}
        try:
            generator = getattr(provider, "generate_completion_with_telemetry", None)
            if callable(generator):
                response, raw_telemetry = call_with_prompt_options(
                    generator,
                    system_prompt_context,
                    system_prompt=system_prompt,
                    system_prompt_only=system_prompt_only,
                )
                telemetry = dict(raw_telemetry) if isinstance(raw_telemetry, Mapping) else {}
            else:
                response = call_with_prompt_options(
                    provider.generate_completion,
                    system_prompt_context,
                    system_prompt=system_prompt,
                    system_prompt_only=system_prompt_only,
                )
        except Exception as exc:
            if not _is_recoverable_provider_exception(exc):
                raise
            return (
                self._format_exception_failure(provider_label, exc),
                {"error": _exception_error_metadata(exc)},
                True,
            )
        return (
            response,
            telemetry,
            self._is_failure(response, telemetry=telemetry, provider=provider),
        )

    def generate_completion(
        self,
        system_prompt_context: str,
        system_prompt: str | None = None,
        *,
        system_prompt_only: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        raw_response: bool = False,
    ) -> str:
        """Try Groq first, then Gemini after any primary failure or exception."""
        request_options: dict[str, Any] = {}
        if temperature is not None:
            request_options["temperature"] = temperature
        if max_tokens is not None:
            request_options["max_tokens"] = max_tokens
        if raw_response:
            request_options["raw_response"] = True

        groq_response, groq_failed = self._invoke_completion_provider(
            self._groq,
            "Groq",
            system_prompt_context,
            system_prompt=system_prompt,
            system_prompt_only=system_prompt_only,
            request_options=request_options,
        )
        if not groq_failed:
            return groq_response

        LOGGER.warning("Groq completion failed; trying the Gemini OpenAI-compatible fallback.")
        gemini_response, gemini_failed = self._invoke_completion_provider(
            self._gemini,
            "Gemini",
            system_prompt_context,
            system_prompt=system_prompt,
            system_prompt_only=system_prompt_only,
            request_options=request_options,
        )
        if not gemini_failed:
            return gemini_response

        return (
            "❌ Multi-provider fallback failed. "
            f"Groq: {str(groq_response)[:300]} Gemini: {str(gemini_response)[:300]}"
        )

    def generate_completion_with_telemetry(
        self,
        system_prompt_context: str,
        system_prompt: str | None = None,
        system_prompt_only: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        """Route telemetry-aware calls and report end-to-end latency and token use."""
        started_at = time.perf_counter()
        groq_response, groq_telemetry, groq_failed = self._invoke_telemetry_provider(
            self._groq,
            "Groq",
            system_prompt_context,
            system_prompt=system_prompt,
            system_prompt_only=system_prompt_only,
        )

        if not groq_failed:
            telemetry = dict(groq_telemetry)
        else:
            LOGGER.warning("Groq completion failed; trying the Gemini OpenAI-compatible fallback.")
            gemini_response, gemini_telemetry, gemini_failed = self._invoke_telemetry_provider(
                self._gemini,
                "Gemini",
                system_prompt_context,
                system_prompt=system_prompt,
                system_prompt_only=system_prompt_only,
            )
            telemetry = self._sum_usage(groq_telemetry, gemini_telemetry)
            if gemini_failed:
                raw_gemini_error = gemini_telemetry.get("error") or str(gemini_response)[:300]
                gemini_response = (
                    "❌ Multi-provider fallback failed. "
                    f"Groq: {str(groq_response)[:300]} Gemini: {str(gemini_response)[:300]}"
                )
                telemetry["error"] = {
                    "groq": groq_telemetry.get("error") or str(groq_response)[:300],
                    "gemini": raw_gemini_error,
                }
            groq_response = gemini_response

        telemetry["latency_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        telemetry["system_prompt_only"] = system_prompt_only
        return groq_response, telemetry
