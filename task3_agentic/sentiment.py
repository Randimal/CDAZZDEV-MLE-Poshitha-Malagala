"""One Task 3 sentiment request, with independent validation of each headline."""

import logging
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from task1_financial.json_utils import json_payload
from task1_financial.llm import CompletionClient, error_category, parse_output
from task1_financial.llm_models import HeadlineSentiment, SentimentBatch
from task1_financial.models import NewsHeadline
from task1_financial.sentiment import aggregate_sentiment
from task3_agentic.prompts import BATCH_SENTIMENT_SYSTEM
from task3_agentic.state import compact_schema

logger = logging.getLogger(__name__)
TASK3_HEADLINE_LIMIT = 10


class BatchSentimentResponse(BaseModel):
    """Advertised LLM contract: every supplied headline, exactly once."""

    model_config = ConfigDict(extra="forbid")
    results: list[HeadlineSentiment] = Field(
        min_length=1, max_length=TASK3_HEADLINE_LIMIT
    )


class _ResponseEnvelope(BaseModel):
    """Parse the container before validating items so one bad item is isolated."""

    model_config = ConfigDict(extra="forbid", strict=True)
    results: list[Any]


def analyze_batch(
    ticker: str, headlines: Sequence[NewsHeadline], client: CompletionClient
) -> SentimentBatch:
    """Analyze up to ten trusted, unique headlines in one logical LLM request.

    Valid items retain input order. Missing, invalid, renamed or duplicated items
    fail their corresponding input headline; no positional guessing or neutral
    substitution occurs. Aggregation uses the unchanged Task 1 formula. Provider
    backoff/JSON-mode fallback belongs to the client; no item repair calls occur.
    """
    if not 1 <= len(headlines) <= TASK3_HEADLINE_LIMIT:
        raise ValueError("Task 3 sentiment requires 1–10 unique retrieved headlines")
    titles = [item.title for item in headlines]
    if len(set(titles)) != len(titles):
        raise ValueError("Sentiment batch headlines must be unique")
    payload = json_payload(
        {
            "stage": "sentiment_batch",
            "ticker": ticker,
            "headlines": [{"title": title} for title in titles],
            "response_schema": compact_schema(
                BatchSentimentResponse.model_json_schema()
            ),
        }
    )
    try:
        envelope = parse_output(
            client.complete(BATCH_SENTIMENT_SYSTEM, payload), _ResponseEnvelope
        )
    except Exception as exc:
        # Never log provider bodies, raw model output, headline text or secrets.
        logger.warning(
            "Task3 sentiment batch unavailable category=%s; failed_count=%d",
            error_category(exc),
            len(titles),
        )
        return SentimentBatch(
            results=[], aggregate=aggregate_sentiment([], len(titles))
        )

    candidates: dict[str, list[Any]] = {title: [] for title in titles}
    for item in envelope.results:
        if isinstance(item, dict) and isinstance(item.get("headline"), str):
            title = item["headline"].strip()
            if title in candidates:
                candidates[title].append(item)
    results: list[HeadlineSentiment] = []
    for title in titles:
        records = candidates[title]
        if len(records) != 1:
            continue
        try:
            results.append(HeadlineSentiment.model_validate(records[0]))
        except ValidationError:
            continue
    failed_count = len(titles) - len(results)
    if failed_count:
        logger.warning(
            "Task3 sentiment batch partial validation: successful_count=%d failed_count=%d",
            len(results),
            failed_count,
        )
    return SentimentBatch(
        results=results, aggregate=aggregate_sentiment(results, failed_count)
    )
