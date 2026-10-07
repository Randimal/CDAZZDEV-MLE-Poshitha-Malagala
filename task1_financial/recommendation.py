"""Build indicator evidence and request a validated technical recommendation."""

import logging
from typing import Any

from pydantic import ValidationError

from task1_financial.json_utils import json_safe
from task1_financial.llm import (
    CompletionClient,
    GroqClient,
    LLMTransportError,
    error_category,
    parse_output,
)
from task1_financial.llm_models import SentimentAggregate, TechnicalRecommendation
from task1_financial.models import PipelineResult
from task1_financial.prompts import (
    RECOMMENDATION_REPAIR,
    RECOMMENDATION_SYSTEM,
    recommendation_user,
)
from task1_financial.technical_facts import build_technical_facts

logger = logging.getLogger(__name__)
RECOMMENDATION_ATTEMPTS = 2  # Initial generation plus at most one repair.

INDICATOR_COLUMNS = (
    "SMA_50",
    "SMA_200",
    "RSI",
    "MACD",
    "MACD_signal",
    "MACD_histogram",
    "BB_upper",
    "BB_middle",
    "BB_lower",
)


def recommendation_payload(
    result: PipelineResult,
    sentiment: SentimentAggregate | None = None,
) -> dict[str, Any]:
    latest = result.data.iloc[-1]
    return json_safe(
        {
            "ticker": result.ticker,
            "price_basis": "adjusted daily OHLC",
            "as_of": result.data.index[-1],
            "current_price": result.summary["current_price"],
            "indicators": {column: latest.get(column) for column in INDICATOR_COLUMNS},
            "deterministic_momentum": result.momentum.signal,
            "technical_facts": build_technical_facts(
                latest.to_dict(), momentum_signal=result.momentum.signal
            ),
            "aggregate_news_sentiment": sentiment.model_dump() if sentiment else None,
        }
    )


def _validation_feedback(exc: ValidationError) -> str:
    """Describe schema rules without echoing generated values or exception text."""
    fields = {
        error["loc"][0]
        for error in exc.errors(include_input=False, include_context=False)
        if error["loc"] and error["loc"][0] in {"signal", "reasoning"}
    }
    rules = []
    if "signal" in fields:
        rules.append("signal is required and must be BUY, HOLD or SELL")
    if "reasoning" in fields:
        rules.append(
            "reasoning is required: a string of 3–5 complete punctuated sentences, "
            "at most 900 characters"
        )
    return "; ".join(rules) or "expected a JSON object with only signal and reasoning"


def get_recommendation(
    result: PipelineResult,
    client: CompletionClient,
    sentiment: SentimentAggregate | None = None,
    *,
    attempts: int = RECOMMENDATION_ATTEMPTS,
) -> TechnicalRecommendation | None:
    """Validate locally with one repair; provider backoff stays inside the client.

    Only Groq's JSON-generation rejection permits a text-mode JSON repair.
    Other terminal provider errors stop; no recommendation is substituted.
    """
    if not 1 <= attempts <= RECOMMENDATION_ATTEMPTS:
        raise ValueError(f"Attempts must be between 1 and {RECOMMENDATION_ATTEMPTS}")
    user = recommendation_user(recommendation_payload(result, sentiment))
    system = RECOMMENDATION_SYSTEM
    json_mode = True
    for attempt in range(1, attempts + 1):
        try:
            if not json_mode and isinstance(client, GroqClient):
                text = client.complete(system, user, json_mode=False)
            else:
                text = client.complete(system, user)
        except LLMTransportError as exc:
            if (
                exc.status_code == 400
                and exc.error_code == "json_validate_failed"
                and isinstance(client, GroqClient)
            ):
                json_mode = False
                feedback = "provider could not generate valid JSON syntax"
                category = "provider_json"
            else:
                logger.warning(
                    "Recommendation category=%s; transport stopped", exc.category
                )
                return None
        except Exception as exc:
            logger.warning(
                "Recommendation category=%s; provider stopped", error_category(exc)
            )
            return None
        else:
            try:
                return parse_output(text, TechnicalRecommendation)
            except ValidationError as exc:
                feedback = _validation_feedback(exc)
                category = "schema_validation"
            except (ValueError, TypeError):
                feedback = "invalid JSON syntax; return exactly one valid JSON object"
                category = "json_parse"
        logger.warning(
            "Recommendation category=%s attempt=%d/%d repair=%s",
            category,
            attempt,
            attempts,
            attempt < attempts,
        )
        system = RECOMMENDATION_SYSTEM + RECOMMENDATION_REPAIR.format(feedback=feedback)
    return None
