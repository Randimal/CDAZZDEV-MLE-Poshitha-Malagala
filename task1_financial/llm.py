"""Mockable Groq adapter with bounded application-level validation retries."""

import json
import logging
import os
import re
from typing import Protocol, TypeVar

from groq import Groq
from pydantic import BaseModel

logger = logging.getLogger(__name__)
DEFAULT_TIMEOUT_SECONDS = 20.0
MAX_ATTEMPTS = 3
Output = TypeVar("Output", bound=BaseModel)


class LLMConfigurationError(ValueError):
    """Required Groq configuration is absent."""


class CompletionClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return model text, or raise a provider error."""
        ...


class GroqClient:
    """Use GROQ_API_KEY/GROQ_MODEL; SDK retries disabled to bound calls.

    Never log credentials, raw provider exceptions or response bodies.
    TLS verification remains the SDK default (enabled).
    """

    def __init__(self) -> None:
        key = os.environ.get("GROQ_API_KEY", "").strip()
        self.model = os.environ.get("GROQ_MODEL", "").strip()
        if not key or not self.model:
            raise LLMConfigurationError("Set GROQ_API_KEY and GROQ_MODEL")
        self._client = Groq(api_key=key, timeout=DEFAULT_TIMEOUT_SECONDS, max_retries=0)

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=700,
        )
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
        except Exception:
            logger.warning(
                "LLM request/validation failed (attempt %d/%d)", attempt, attempts
            )
    return None
