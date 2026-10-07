"""Task 2-only, mockable Groq transport with bounded retries and quota pacing."""

import logging
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

from groq import APIConnectionError, APITimeoutError, Groq

from task2_genai.schemas import TEACHER_MODEL

logger = logging.getLogger(__name__)


class TeacherError(RuntimeError):
    """Safe provider category; raw exceptions must not be logged."""

    def __init__(self, category: str, *, raw_response: str | None = None) -> None:
        self.category = category
        self.raw_response = raw_response
        super().__init__(f"Teacher request failed: {category}")


class TeacherClient(Protocol):
    """Minimal interface used by generation and offline tests."""

    model: str

    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


def _category(error: Exception) -> str:
    if isinstance(error, APITimeoutError):
        return "timeout"
    if isinstance(error, APIConnectionError):
        return "connection"
    status = getattr(error, "status_code", None)
    if status == 429:
        return "rate_limit"
    if isinstance(status, int) and status >= 500:
        return "server"
    if status in (401, 403):
        return "authentication"
    return "invalid_request"


def _retry_after(error: Exception) -> float | None:
    headers = getattr(getattr(error, "response", None), "headers", {})
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(str(value))
            return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _json_rejected(error: Exception) -> bool:
    """Read only the known error code; never expose the provider response body."""
    code = getattr(error, "code", None)
    body = getattr(error, "body", None)
    if code is None and isinstance(body, dict):
        detail = body.get("error", body)
        if isinstance(detail, dict):
            code = detail.get("code")
    return getattr(error, "status_code", None) == 400 and code == "json_validate_failed"


class GroqTeacherClient:
    """SDK retries disabled; at most three requests, including a JSON-mode fallback.

    Pacing is optional and occurs only at the transport boundary. Retry-After is
    honored unless it exceeds the bounded wait budget, in which case the batch
    fails without retrying early. Local validation never triggers transport retries.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        sdk: Any = None,
        max_attempts: int = 3,
        max_completion_tokens: int = 2500,
        min_interval_seconds: float = 0.0,
        max_retry_wait_seconds: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 1 <= max_attempts <= 3 or max_completion_tokens < 1:
            raise ValueError("Use 1-3 attempts and a positive completion budget")
        if min_interval_seconds < 0 or max_retry_wait_seconds < 0:
            raise ValueError("Wait budgets must be nonnegative")
        self.model = model or os.environ.get("GROQ_MODEL") or TEACHER_MODEL
        if sdk is None:
            key = os.environ.get("GROQ_API_KEY")
            if not key:
                raise TeacherError("missing_configuration")
            sdk = Groq(api_key=key, max_retries=0, timeout=90.0)
        self._sdk = sdk
        self._attempts = max_attempts
        self._tokens = max_completion_tokens
        self._interval = min_interval_seconds
        self._max_wait = max_retry_wait_seconds
        self._sleep = sleep
        self._clock = clock
        self._completed_at: float | None = None
        self._rate_limit_until: float | None = None

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        json_mode = True
        for attempt in range(1, self._attempts + 1):
            now = self._clock()
            quota_wait = max(0.0, (self._rate_limit_until or now) - now)
            if quota_wait > self._max_wait:
                raise TeacherError("rate_limit")
            wait = quota_wait
            if self._completed_at is not None:
                wait = max(wait, self._interval - (now - self._completed_at))
            if wait > 0:
                logger.info("Task 2 teacher quota pacing: waiting %.1fs", wait)
                self._sleep(wait)
            request: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_completion_tokens": self._tokens,
                "temperature": 0.7,
            }
            if json_mode:
                request["response_format"] = {"type": "json_object"}
            if self.model in ("openai/gpt-oss-120b", "openai/gpt-oss-20b"):
                request["reasoning_effort"] = "low"
            try:
                response = self._sdk.chat.completions.create(**request)
                self._completed_at = self._clock()
                content = response.choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    raise TeacherError("empty_response")
                if getattr(response.choices[0], "finish_reason", None) == "length":
                    logger.warning("Task 2 provider category=completion_truncated")
                    raise TeacherError("completion_truncated", raw_response=content)
                return content
            except TeacherError:
                self._completed_at = self._clock()
                raise
            except Exception as error:
                self._completed_at = self._clock()
                category = _category(error)
                status = getattr(error, "status_code", None)
                json_rejection = json_mode and _json_rejected(error)
                retryable = category in (
                    "rate_limit",
                    "timeout",
                    "connection",
                    "server",
                )
                delay = _retry_after(error)
                delay = delay if delay is not None else float(2 ** (attempt - 1))
                retry = attempt < self._attempts and (json_rejection or retryable)
                if delay > self._max_wait:
                    retry = False
                    if category == "rate_limit":
                        self._rate_limit_until = self._completed_at + delay
                logger.warning(
                    "Task 2 provider category=%s status=%s attempt=%s/%s retry=%s",
                    category,
                    status if isinstance(status, int) else None,
                    attempt,
                    self._attempts,
                    retry,
                )
                if not retry:
                    raise TeacherError(category) from None
                if json_rejection:
                    json_mode = False
                self._sleep(delay)
        raise TeacherError("request_exhausted")
