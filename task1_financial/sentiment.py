"""Per-headline LLM analysis with deterministic, failure-aware aggregation."""

from collections.abc import Sequence

from task1_financial.llm import CompletionClient, validated_completion
from task1_financial.llm_models import (
    HeadlineSentiment,
    SentimentAggregate,
    SentimentBatch,
)
from task1_financial.models import NewsHeadline
from task1_financial.prompts import SENTIMENT_SYSTEM, sentiment_user

# A net confidence-weighted direction above 0.20 is meaningful by policy.
# Symmetric inclusive boundaries are directional; the interior is neutral.
# These are transparent heuristic thresholds, not empirically calibrated.
POSITIVE_THRESHOLD = 0.20
NEGATIVE_THRESHOLD = -POSITIVE_THRESHOLD


def aggregate_sentiment(
    results: Sequence[HeadlineSentiment],
    failed_count: int = 0,
) -> SentimentAggregate:
    if failed_count < 0:
        raise ValueError("Failed count cannot be negative")
    counts = {
        label: sum(item.sentiment == label for item in results)
        for label in ("positive", "negative", "neutral")
    }
    score = None
    label = "unavailable"
    if results:
        direction = {"positive": 1, "negative": -1, "neutral": 0}
        score = sum(
            direction[item.sentiment] * item.confidence for item in results
        ) / len(results)
        label = (
            "positive"
            if score >= POSITIVE_THRESHOLD
            else "negative"
            if score <= NEGATIVE_THRESHOLD
            else "neutral"
        )
    return SentimentAggregate(
        overall_score=score,
        overall_label=label,
        positive_count=counts["positive"],
        negative_count=counts["negative"],
        neutral_count=counts["neutral"],
        successful_count=len(results),
        failed_count=failed_count,
    )


def analyze_headlines(
    ticker: str,
    headlines: Sequence[NewsHeadline],
    client: CompletionClient,
) -> SentimentBatch:
    """One call per headline; failures are counted, never substituted."""
    results: list[HeadlineSentiment] = []
    failed_count = 0
    for item in headlines:
        result = validated_completion(
            client,
            SENTIMENT_SYSTEM,
            sentiment_user(ticker, item.title),
            HeadlineSentiment,
            expected_headline=item.title,
        )
        if result is None:
            failed_count += 1
        else:
            results.append(result)
    return SentimentBatch(
        results=results, aggregate=aggregate_sentiment(results, failed_count)
    )
