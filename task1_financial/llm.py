"""Mockable Groq adapter with bounded application-level validation retries."""

import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Protocol, TypeVar

from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq
from pydantic import BaseModel

logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT_SECONDS = 20.0
MAX_ATTEMPTS = 3
TRANSPORT_ATTEMPTS = 4
MAX_RETRY_WAIT_SECONDS = 60.0
Output = TypeVar("Output", bound=BaseModel)


class LLMConfigurationError(ValueError):
    """Required Groq configuration is absent."""


class LLMTransportError(RuntimeError):
    """Safe terminal provider failure after transport retries, without raw bodies."""

    def __init__(
        self,
        category: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
    ) -> None:
        self.category = category
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(f"LLM provider unavailable: {category}")


SAFE_PROVIDER_CODES = frozenset(
    {
        "rate_limit_exceeded",
        "context_length_exceeded",
        "model_not_found",
        "json_validate_failed",
        "tool_use_failed",
        "invalid_request_error",
        "invalid_api_key",
        "authentication_error",
        "tokens_per_minute_limit",
    }
)


def safe_provider_details(exc: Exception) -> dict[str, int | str | None]:
    """Extract status and allowlisted machine code only, never messages/headers."""
    status = getattr(exc, "status_code", None)
    code = getattr(exc, "error_code", None)
    body = getattr(exc, "body", None)
    if code is None and isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            code = error.get("code")
    return {
        "status_code": status if type(status) is int and 100 <= status <= 599 else None,
        "error_code": code
        if isinstance(code, str) and code in SAFE_PROVIDER_CODES
        else None,
    }


def error_category(exc: Exception) -> str:
    """Classify using SDK types/status only; never inspect sensitive error text."""
    if isinstance(exc, LLMTransportError):
        return exc.category
    if isinstance(exc, (APITimeoutError, TimeoutError)):
        return "timeout"
    if isinstance(exc, (APIConnectionError, ConnectionError)):
        return "connection"
    if isinstance(exc, APIStatusError):
        if exc.status_code == 429:
            return "rate_limit"
        if exc.status_code in {408, 409, 498} or exc.status_code >= 500:
            return "connection"
        return "invalid_request"
    if isinstance(exc, ValueError):
        return "validation"
    return "invalid_request"


def retry_delay(exc: Exception, attempt: int) -> float:
    """Exponential 1/2/4 seconds, honoring numeric or HTTP-date Retry-After.

    A delay beyond the wait bound ends this request instead of retrying early.
    """
    delay = float(2**attempt)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {})
    value = headers.get("retry-after")
    if value:
        try:
            wait = float(value)
        except (TypeError, ValueError):
            try:
                wait = (
                    parsedate_to_datetime(value) - datetime.now(timezone.utc)
                ).total_seconds()
            except (TypeError, ValueError, OverflowError):
                wait = 0.0
        if math.isfinite(wait):
            delay = max(delay, wait)
    return delay


class CompletionClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return model text, or raise a provider error."""
        ...


class GroqClient:
    """Use GROQ_API_KEY/GROQ_MODEL; SDK retries disabled to bound calls.

    Never log credentials, raw provider exceptions or response bodies.
    TLS verification remains the SDK default (enabled).
    """

    def __init__(self, *, max_completion_tokens: int = 700) -> None:
        if not 1 <= max_completion_tokens <= 4096:
            raise ValueError("Completion token limit must be between 1 and 4096")
        self.max_completion_tokens = max_completion_tokens
        key = os.environ.get("GROQ_API_KEY", "").strip()
        self.model = os.environ.get("GROQ_MODEL", "").strip()
        if not key or not self.model:
            raise LLMConfigurationError("Set GROQ_API_KEY and GROQ_MODEL")
        self._client = Groq(api_key=key, timeout=DEFAULT_TIMEOUT_SECONDS, max_retries=0)

    def complete(self, system: str, user: str, *, json_mode: bool = True) -> str:
        """Retry transport errors; optionally return text for local JSON validation.

        JSON-object mode remains the default for all existing callers.
        """
        for attempt in range(TRANSPORT_ATTEMPTS):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object" if json_mode else "text"},
                    temperature=0,
                    max_completion_tokens=self.max_completion_tokens,
                )
                break
            except Exception as exc:
                category = error_category(exc)
                details = safe_provider_details(exc)
                delay = retry_delay(exc, attempt)
                retry = (
                    category in {"rate_limit", "timeout", "connection"}
                    and attempt + 1 < TRANSPORT_ATTEMPTS
                    and delay <= MAX_RETRY_WAIT_SECONDS
                )
                logger.warning(
                    "LLM category=%s status=%s code=%s attempt=%d/%d retry=%s",
                    category,
                    details["status_code"],
                    details["error_code"],
                    attempt + 1,
                    TRANSPORT_ATTEMPTS,
                    retry,
                )
                if not retry:
                    raise LLMTransportError(category, **details) from None
                time.sleep(delay)
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise ValueError("Provider returned no text")
        return content


def parse_output(text: str, schema: type[Output]) -> Output:
    """Accept plain JSON or a whole JSON fence; do not repair malformed data."""
    stripped = text.strip()
    fence = re.fullmatch(
        r"```(?:json)?\s*\n?(.*?)\n?```", stripped, flags=re.DOTALL | re.IGNORECASE
    )
    if fence:
        stripped = fence.group(1).strip()

    # Reject nonstandard JSON NaN/Infinity before schema validation.
    def reject_constant(value: str) -> None:
        raise ValueError("Nonstandard JSON numeric constant")

    decoded = json.loads(stripped, parse_constant=reject_constant)
    return schema.model_validate(decoded)


def validated_completion(
    client: CompletionClient,
    system: str,
    user: str,
    schema: type[Output],
    *,
    attempts: int = 1,
    expected_headline: str | None = None,
) -> Output | None:
    """Return only validated results; exhausted attempts return None.

    Broad catches are intentionally isolated at the external API boundary.
    Logs contain only attempt counts, never raw exception/response text.
    """
    if not 1 <= attempts <= MAX_ATTEMPTS:
        raise ValueError(f"Attempts must be between 1 and {MAX_ATTEMPTS}")
    for attempt in range(1, attempts + 1):
        try:
            output = parse_output(client.complete(system, user), schema)
            if (
                expected_headline is not None
                and getattr(output, "headline", None) != expected_headline
            ):
                raise ValueError("Returned headline does not match input")
            return output
        except LLMTransportError as exc:
            logger.warning("LLM category=%s; transport stopped", exc.category)
            return None
        except Exception:
            logger.warning(
                "LLM request/validation failed (attempt %d/%d)", attempt, attempts
            )
    return None
