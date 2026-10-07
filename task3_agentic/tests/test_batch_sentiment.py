"""Mocked regressions for Task 3-only batching and the ten-headline quota limit."""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from task1_financial.models import NewsHeadline
from task1_financial.sentiment import analyze_headlines as task1_analyze_headlines
from task3_agentic.sentiment import BatchSentimentResponse
from task3_agentic.tools import NewsArguments, SentimentArguments


def ten_headlines():
    headlines = [NewsHeadline(f"Retrieved headline {i}") for i in range(10)]
    results = [
        {
            "headline": item.title,
            "sentiment": "positive" if i < 3 else "negative" if i < 5 else "neutral",
            "confidence": 0.8 if i < 3 else 0.6 if i < 5 else 0.5,
            "brief_reason": "Synthetic fixture evidence for an offline test.",
        }
        for i, item in enumerate(headlines)
    ]
    return headlines, results


def seed_response(executor, headlines, results):
    executor.tools.known_headlines.update({item.title: item for item in headlines})
    executor.tools.client.complete.return_value = json.dumps({"results": results})
    return executor.invoke(
        "analyst",
        "llm_sentiment",
        {"headlines": [{"title": item.title} for item in headlines]},
    )


def test_ten_headlines_one_request_preserves_results_and_aggregate(executor):
    headlines, results = ten_headlines()
    observation = seed_response(executor, headlines, list(reversed(results)))
    assert observation.success
    executor.tools.client.complete.assert_called_once()
    output = observation.output
    assert output["results"] == results  # Exact fields and original input order.
    BatchSentimentResponse.model_validate({"results": output["results"]})
    aggregate = output["aggregate"]
    assert {
        key: aggregate[key]
        for key in (
            "positive_count",
            "negative_count",
            "neutral_count",
            "successful_count",
            "failed_count",
        )
    } == {
        "positive_count": 3,
        "negative_count": 2,
        "neutral_count": 5,
        "successful_count": 10,
        "failed_count": 0,
    }
    assert aggregate["overall_score"] == pytest.approx((3 * 0.8 - 2 * 0.6) / 10)
    assert aggregate["overall_label"] == "neutral"
    payload = json.loads(executor.tools.client.complete.call_args.args[1])
    assert payload["stage"] == "sentiment_batch" and len(payload["headlines"]) == 10
    assert payload["response_schema"]["properties"]["results"]["maxItems"] == 10


@pytest.mark.parametrize("failure", ["confidence", "missing", "duplicate", "renamed"])
def test_one_bad_item_is_failed_without_discarding_other_results(executor, failure):
    headlines, results = ten_headlines()
    bad = deepcopy(results)
    if failure == "confidence":
        bad[0]["confidence"] = 1.1
    elif failure == "missing":
        bad.pop(0)
    elif failure == "duplicate":
        bad.append(deepcopy(bad[0]))
    else:
        bad[0]["headline"] = "Not a supplied headline"
    observation = seed_response(executor, headlines, bad)
    assert observation.success
    executor.tools.client.complete.assert_called_once()
    assert observation.output["results"] == results[1:]
    aggregate = observation.output["aggregate"]
    assert aggregate["successful_count"] == 9 and aggregate["failed_count"] == 1
    assert (
        aggregate["positive_count"],
        aggregate["negative_count"],
        aggregate["neutral_count"],
    ) == (2, 2, 5)
    assert aggregate["overall_score"] == pytest.approx((2 * 0.8 - 2 * 0.6) / 9)


@pytest.mark.parametrize("failure", ["json", "provider"])
def test_batch_failure_counts_every_headline_without_item_retries(executor, failure):
    headlines, _ = ten_headlines()
    executor.tools.known_headlines.update({item.title: item for item in headlines})
    if failure == "json":
        executor.tools.client.complete.return_value = "{broken"
    else:
        executor.tools.client.complete.side_effect = RuntimeError(
            "sensitive provider body"
        )
    output = executor.tools.llm_sentiment([{"title": item.title} for item in headlines])
    executor.tools.client.complete.assert_called_once()
    assert output["results"] == []
    assert output["aggregate"]["successful_count"] == 0
    assert output["aggregate"]["failed_count"] == 10
    assert output["aggregate"]["overall_score"] is None
    assert output["aggregate"]["overall_label"] == "unavailable"


def test_news_default_limit_is_ten_and_oversize_requests_never_fetch(executor):
    headlines, _ = ten_headlines()
    news_client = executor.tools.news_client_factory.return_value
    news_client.get_news.return_value = [
        {"title": item.title} for item in headlines
    ] + [{"title": "Extra headline"}]
    assert NewsArguments(ticker="NVDA").n == 10
    assert len(executor.tools.get_news("NVDA")["headlines"]) == 10
    assert news_client.get_news.call_args.kwargs["count"] == 10
    with pytest.raises(ValidationError):
        NewsArguments(ticker="NVDA", n=11)
    with pytest.raises(ValueError):
        executor.tools.get_news("NVDA", n=20)
    assert not executor.invoke(
        "writer", "get_news", {"ticker": "NVDA", "n": 20}
    ).success
    news_client.get_news.assert_called_once()
    with pytest.raises(ValidationError):
        SentimentArguments(headlines=[{"title": str(i)} for i in range(11)])
    executor.tools.client.complete.assert_not_called()


def test_task1_remains_per_headline_with_unchanged_aggregate():
    headlines, results = ten_headlines()
    client = Mock()
    client.complete.side_effect = [json.dumps(item) for item in results]
    batch = task1_analyze_headlines("NVDA", headlines, client)
    assert client.complete.call_count == 10
    assert [item.model_dump() for item in batch.results] == results
    assert batch.aggregate.successful_count == 10 and batch.aggregate.failed_count == 0
    assert batch.aggregate.overall_score == pytest.approx((3 * 0.8 - 2 * 0.6) / 10)
