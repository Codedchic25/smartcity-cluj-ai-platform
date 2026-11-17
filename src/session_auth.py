"""Signed browser-session tokens and Streamlit authentication state helpers."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import time
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Final

from src.schemas import get_session_security_token

SESSION_TOKEN_QUERY_PARAM: Final[str] = "session_token"
SESSION_TOKEN_STATE_KEY: Final[str] = "_session_token"
SESSION_TTL_SECONDS: Final[int] = 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class SessionTokenClaims:
    """Verified identity and expiry claims extracted from a signed session token."""

    username: str
    expires_at: int


def _encode_segment(value: bytes) -> str:
    """Encode bytes as an unpadded URL-safe Base64 token segment."""
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_segment(value: str) -> bytes:
    """Decode an unpadded URL-safe Base64 token segment."""
    encoded = value.encode("ascii")
    padded = encoded + b"=" * (-len(encoded) % 4)
    return base64.b64decode(padded, altchars=b"-_", validate=True)


def _signing_key() -> bytes:
    """Resolve the configured HMAC key or raise an explicit configuration error."""
    secret = get_session_security_token()
    if not secret:
        raise RuntimeError("SESSION_SECURITY_TOKEN must be configured to create login sessions.")
    return secret.encode("utf-8")


def _sign_session_token(username: str, expires_at: int) -> str:
    payload = json.dumps(
        {"exp": expires_at, "sub": username},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    encoded_payload = _encode_segment(payload)
    signature = hmac.new(_signing_key(), encoded_payload.encode("ascii"), hashlib.sha256).digest()
    return f"{encoded_payload}.{_encode_segment(signature)}"


def issue_signed_session_token(username: str, *, now: int | None = None) -> tuple[str, int]:
    """Create an HMAC-SHA256 token valid for one configured session lifetime."""
    clean_username = username.strip()
    if not clean_username:
        raise ValueError("A non-empty username is required to create a login session.")

    current_time = int(time.time()) if now is None else now
    expires_at = current_time + SESSION_TTL_SECONDS
    return _sign_session_token(clean_username, expires_at), expires_at


def verify_signed_session_token(
    token: str,
    *,
    now: int | None = None,
) -> SessionTokenClaims | None:
    """Verify a signed token and reject malformed, tampered, or expired sessions."""
    secret = get_session_security_token()
    if not secret:
        return None

    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        provided_signature = _decode_segment(encoded_signature)
        expected_signature = hmac.new(
            secret.encode("utf-8"),
            encoded_payload.encode("ascii"),
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(expected_signature, provided_signature):
            return None

        payload = json.loads(_decode_segment(encoded_payload))
        if not isinstance(payload, dict):
            return None

        username = payload.get("sub")
        expires_at = payload.get("exp")
        current_time = int(time.time()) if now is None else now
        if (
            not isinstance(username, str)
            or not username.strip()
            or isinstance(expires_at, bool)
            or not isinstance(expires_at, int)
            or expires_at <= current_time
        ):
            return None

        return SessionTokenClaims(username=username, expires_at=expires_at)
    except (binascii.Error, UnicodeError, ValueError, TypeError):
        return None


def persist_authenticated_session(
    username: str,
    session_state: MutableMapping[str, Any],
    query_params: MutableMapping[str, Any],
    *,
    now: int | None = None,
) -> None:
    """Mark the user authenticated and persist a signed token in the URL."""
    token, expires_at = issue_signed_session_token(username, now=now)
    session_state["authenticated"] = True
    session_state["operator_name"] = username.strip()
    session_state["session_expires_at"] = expires_at
    session_state[SESSION_TOKEN_STATE_KEY] = token
    query_params[SESSION_TOKEN_QUERY_PARAM] = token


def restore_authenticated_session(
    session_state: MutableMapping[str, Any],
    query_params: MutableMapping[str, Any],
    *,
    now: int | None = None,
) -> bool:
    """Restore a valid authenticated session from state or the signed URL token."""
    current_time = int(time.time()) if now is None else now
    state_expiry = session_state.get("session_expires_at")
    if (
        session_state.get("authenticated") is True
        and isinstance(state_expiry, int)
        and not isinstance(state_expiry, bool)
        and state_expiry > current_time
    ):
        username = session_state.get("operator_name")
        if not isinstance(username, str) or not username.strip():
            clear_authenticated_session(session_state, query_params)
            return False

        token = session_state.get(SESSION_TOKEN_STATE_KEY)
        claims = (
            verify_signed_session_token(token, now=current_time) if isinstance(token, str) else None
        )
        if claims is None or claims.username != username or claims.expires_at != state_expiry:
            token = _sign_session_token(username, state_expiry)
            session_state[SESSION_TOKEN_STATE_KEY] = token

        query_params[SESSION_TOKEN_QUERY_PARAM] = token
        return True

    token = query_params.get(SESSION_TOKEN_QUERY_PARAM, "")
    claims = (
        verify_signed_session_token(token, now=current_time) if isinstance(token, str) else None
    )
    if claims is not None:
        session_state["authenticated"] = True
        session_state["operator_name"] = claims.username
        session_state["session_expires_at"] = claims.expires_at
        session_state[SESSION_TOKEN_STATE_KEY] = token
        query_params[SESSION_TOKEN_QUERY_PARAM] = token
        return True

    clear_authenticated_session(session_state, query_params)
    return False


def clear_authenticated_session(
    session_state: MutableMapping[str, Any],
    query_params: MutableMapping[str, Any],
) -> None:
    """Clear the authenticated state and remove its persistent URL token."""
    session_state["authenticated"] = False
    session_state.pop("operator_name", None)
    session_state.pop("session_expires_at", None)
    session_state.pop(SESSION_TOKEN_STATE_KEY, None)
    if SESSION_TOKEN_QUERY_PARAM in query_params:
        del query_params[SESSION_TOKEN_QUERY_PARAM]
