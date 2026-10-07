"""Build indicator evidence and request a validated technical recommendation."""

from typing import Any

from task1_financial.json_utils import json_safe
from task1_financial.llm import MAX_ATTEMPTS, CompletionClient, validated_completion
from task1_financial.llm_models import SentimentAggregate, TechnicalRecommendation
from task1_financial.models import PipelineResult
from task1_financial.prompts import RECOMMENDATION_SYSTEM, recommendation_user
from task1_financial.technical_facts import build_technical_facts

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


def get_recommendation(
    result: PipelineResult,
    client: CompletionClient,
    sentiment: SentimentAggregate | None = None,
    *,
    attempts: int = MAX_ATTEMPTS,
) -> TechnicalRecommendation | None:
    """Bound response-validation attempts; transport retries stay inside the client."""
    return validated_completion(
        client,
        RECOMMENDATION_SYSTEM,
        recommendation_user(recommendation_payload(result, sentiment)),
        TechnicalRecommendation,
        attempts=attempts,
    )
