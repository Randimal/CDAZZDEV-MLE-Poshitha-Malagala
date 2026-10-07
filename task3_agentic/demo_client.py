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
REQUEST_INTERVAL_SECONDS = 60.0
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
    sentiment_batch: int = 1800  # Up to ten item records, including copied titles.
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
        "sentiment_batch": "sentiment_batch",
    }.get(stage, "report")


class Task3GroqClient(GroqClient):
    """Reuse frozen Task 1 SDK setup/backoff helpers with per-request options.

    Inherits GroqClient so the existing text-mode synthesis repair stays usable.
    No model switch, new credentials, hidden retries or TLS override is introduced.
    """

    handles_provider_json_fallback = True

    def __init__(
        self,
        *,
        budgets: TokenBudgets | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        free_tier_pacing: bool = False,
        request_interval_seconds: float = REQUEST_INTERVAL_SECONDS,
        announce: Callable[[str], None] = logger.info,
    ) -> None:
        self.budgets = budgets or TokenBudgets()
        super().__init__(max_completion_tokens=self.budgets.report)
        self.call_count = 0
        self.last_completed_at: float | None = None
        self._clock, self._sleep = clock, sleep
        self.request_pacer = DemoPacer(
            enabled=free_tier_pacing,
            window_seconds=request_interval_seconds,
            clock=clock,
            sleep=sleep,
            announce=announce,
        )
        self._supports_reasoning = (
            self.model in LOW_MEDIUM_REASONING_MODELS
            and "reasoning_effort" in inspect.signature(Completions.create).parameters
        )

    def complete(self, system: str, user: str, *, json_mode: bool = True) -> str:
        """One provider JSON fallback; local parsing/validation stays with callers.

        A provider's rejection of JSON generation is distinct from malformed
        returned text or local schema failures. Only HTTP 400/json_validate_failed
        changes response mode; transient errors retain existing transport backoff.
        """
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
            return self._request(system, user, json_mode, budget, options)
        except LLMTransportError as exc:
            if not (
                json_mode
                and exc.status_code == 400
                and exc.error_code == "json_validate_failed"
            ):
                raise
            logger.warning(
                "Task3 provider JSON fallback status=400 code=json_validate_failed attempt=1/1"
            )
            repair_system = system + (
                "\nProvider JSON generation was rejected. Return exactly one valid "
                "JSON object matching the supplied schema. No Markdown fences or "
                "surrounding prose; do not invent fields, actions or evidence."
            )
            # This call cannot fall back recursively. Its transport may still
            # honor Retry-After on actual transient errors, bounded as before.
            return self._request(repair_system, user, False, budget, options)

    def _request(
        self,
        system: str,
        user: str,
        json_mode: bool,
        budget: int,
        options: dict[str, str],
    ) -> str:
        """Pace generation requests; provider backoff controls transport retries."""
        self.request_pacer.before_request(self.last_completed_at)
        for attempt in range(TRANSPORT_ATTEMPTS):
            try:
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
                finally:
                    # Failed requests can consume quota. Both pacers share this
                    # timestamp; neither pacing nor transport sleeps update it.
                    self.last_completed_at = self._clock()
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


class DemoPacer:
    """Optional free-tier pacing at notebook sections or request boundaries.

    Wait the remaining part of one minute since the last completion, not a full
    minute unconditionally. This is conservative demo spacing, not a quota grant:
    other clients and unusually large requests can still hit provider limits.
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
        return self._wait(
            last_completed_at,
            "Groq free-tier pacing enabled — waiting {wait:.1f}s for the next "
            f"token window before {section}. Free-tier pacing between independent "
            "demonstration sections; provider Retry-After remains authoritative.",
        )

    def before_request(self, last_completed_at: float | None) -> float:
        return self._wait(
            last_completed_at,
            "Groq free-tier intra-workflow pacing — waiting {wait:.1f}s before next LLM request.",
        )

    def _wait(self, last_completed_at: float | None, message: str) -> float:
        if not self.enabled or last_completed_at is None:
            return 0.0
        wait = max(0.0, self.window_seconds - (self._clock() - last_completed_at))
        if wait:
            self._announce(message.format(wait=wait))
            self._sleep(wait)
        return wait
