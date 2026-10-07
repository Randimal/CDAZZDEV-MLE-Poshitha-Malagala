"""Task 3 request budgets and optional notebook pacing; no graph changes."""

import inspect
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from groq.resources.chat.completions import Completions

from task1_financial.llm import (
    MAX_RETRY_WAIT_SECONDS,
    TRANSPORT_ATTEMPTS,
    GroqClient,
    LLMTransportError,
    error_category,
    retry_delay,
    safe_provider_details,
)
from task1_financial.prompts import SENTIMENT_SYSTEM
from task3_agentic.prompts import FOLLOWUP_SYSTEM

logger = logging.getLogger(__name__)
MAX_COMPLETION_TOKENS = 4096
TOKEN_WINDOW_SECONDS = 60.0
CONTROLLED_DEMO_DECISIONS = 3
# Send optional reasoning parameters only to verified supported models/SDKs.
LOW_MEDIUM_REASONING_MODELS = frozenset({"openai/gpt-oss-20b", "openai/gpt-oss-120b"})


@dataclass(frozen=True)
class TokenBudgets:
    """Output limits, not estimates of prompt size or measured token usage."""

    planner: int = 768  # Includes room to pass typical retrieved headline titles.
    critique: int = 512
    brief: int = 1200
    clarification: int = 900
    followup: int = 640
    sentiment: int = 384
    report: int = 1800

    def __post_init__(self) -> None:
        for value in vars(self).values():
            if type(value) is not int or not 1 <= value <= MAX_COMPLETION_TOKENS:
                raise ValueError(
                    "Completion budgets must be integers between 1 and 4096"
                )


def request_profile(system: str, user: str) -> str:
    """Classify existing request contracts without predicting the chosen action."""
    if system == SENTIMENT_SYSTEM:
        return "sentiment"
    if system == FOLLOWUP_SYSTEM:
        return "followup"
    try:
        payload = json.loads(user)
    except (ValueError, TypeError):
        return "report"  # Unrecognized requests retain the larger safe allowance.
    stage = payload.get("stage") if isinstance(payload, dict) else None
    if not isinstance(stage, str):
        return "report"
    return {
        "single_research": "planner",
        "writer_review": "critique",
        "analyst_brief": "brief",
        "analyst_clarify": "clarification",
        "writer_final": "report",
        "final_synthesis": "report",
    }.get(stage, "report")


class Task3GroqClient(GroqClient):
    """Reuse frozen Task 1 SDK setup/backoff helpers with per-request options.

    Inherits GroqClient so the existing text-mode synthesis repair stays usable.
    No model switch, new credentials, hidden retries or TLS override is introduced.
    """

    def __init__(
        self,
        *,
        budgets: TokenBudgets | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.budgets = budgets or TokenBudgets()
        super().__init__(max_completion_tokens=self.budgets.report)
        self.call_count = 0
        self.last_completed_at: float | None = None
        self._clock, self._sleep = clock, sleep
        self._supports_reasoning = (
            self.model in LOW_MEDIUM_REASONING_MODELS
            and "reasoning_effort" in inspect.signature(Completions.create).parameters
        )

    def complete(self, system: str, user: str, *, json_mode: bool = True) -> str:
        """Keep bounded transport retries/Retry-After separate from demo pacing."""
        profile = request_profile(system, user)
        budget = getattr(self.budgets, profile)
        options = {}
        if self._supports_reasoning:
            options["reasoning_effort"] = "medium" if profile == "report" else "low"
        logger.info(
            "Task3 request_profile=%s completion_budget=%d reasoning_effort=%s",
            profile,
            budget,
            options.get("reasoning_effort", "provider_default"),
        )
        self.call_count += 1
        try:
            for attempt in range(TRANSPORT_ATTEMPTS):
                try:
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        response_format={
                            "type": "json_object" if json_mode else "text"
                        },
                        temperature=0,
                        max_completion_tokens=budget,
                        **options,
                    )
                    content = response.choices[0].message.content
                    if not isinstance(content, str):
                        raise ValueError("Provider returned no text")
                    return content
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
                    self._sleep(delay)
            raise AssertionError("Bounded transport must return or raise")
        finally:
            # Failed requests may consume quota too; section pacing follows either.
            self.last_completed_at = self._clock()


class DemoPacer:
    """Free-tier pacing between independent demonstration sections only.

    Wait the remaining part of one minute since the last completion, not a full
    minute unconditionally. This is conservative demo spacing, not a quota grant:
    other clients and requests within a workflow can still hit provider limits.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        window_seconds: float = TOKEN_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        announce: Callable[[str], None] = logger.info,
    ) -> None:
        if not math.isfinite(window_seconds) or not 0 <= window_seconds <= 300:
            raise ValueError("Demo pacing window must be between 0 and 300 seconds")
        self.enabled, self.window_seconds = enabled, window_seconds
        self._clock, self._sleep, self._announce = clock, sleep, announce

    def before_section(self, section: str, last_completed_at: float | None) -> float:
        if not self.enabled or last_completed_at is None:
            return 0.0
        wait = max(0.0, self.window_seconds - (self._clock() - last_completed_at))
        if wait:
            self._announce(
                f"Groq free-tier pacing enabled — waiting {wait:.1f}s for the next "
                f"token window before {section}. Free-tier pacing between independent "
                "demonstration sections; provider Retry-After remains authoritative."
            )
            self._sleep(wait)
        return wait
