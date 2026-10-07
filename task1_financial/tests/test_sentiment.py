import json
from unittest.mock import Mock

import pytest

from task1_financial.llm_models import HeadlineSentiment
from task1_financial.models import NewsHeadline
from task1_financial.sentiment import aggregate_sentiment, analyze_headlines


@pytest.mark.parametrize(
    "items,label,score",
    [
        ([("positive", 0.8)], "positive", 0.8),
        ([("negative", 0.8)], "negative", -0.8),
        ([("neutral", 0.9)], "neutral", 0),
        ([("positive", 0.8), ("negative", 0.8)], "neutral", 0),
        ([("positive", 0.2)], "positive", 0.2),
        ([("negative", 0.2)], "negative", -0.2),
        ([("positive", 0.19)], "neutral", 0.19),
        ([("positive", 0.9), ("negative", 0.3), ("neutral", 1)], "positive", 0.2),
    ],
)
def test_aggregate(items: list[tuple[str, float]], label: str, score: float) -> None:
    values = [
        HeadlineSentiment(
            headline=str(i),
            sentiment=kind,
            confidence=confidence,
            brief_reason="Evidence",
        )
        for i, (kind, confidence) in enumerate(items)
    ]
    result = aggregate_sentiment(values, failed_count=2)
    assert result.overall_score == pytest.approx(score)
    assert result.overall_label == label
    assert result.successful_count == len(items) and result.failed_count == 2
    assert result.positive_count + result.negative_count + result.neutral_count == len(
        items
    )


def test_failed_headline_does_not_abort_batch() -> None:
    client = Mock()
    client.complete.side_effect = [
        "bad JSON",
        json.dumps(
            {
                "headline": "Second",
                "sentiment": "positive",
                "confidence": 0.8,
                "brief_reason": "Evidence",
            }
        ),
        RuntimeError("offline"),
    ]
    batch = analyze_headlines(
        "NVDA", [NewsHeadline(title) for title in ("First", "Second", "Third")], client
    )
    assert len(batch.results) == 1 and batch.results[0].headline == "Second"
    assert batch.aggregate.failed_count == 2
    assert batch.aggregate.overall_score == 0.8
    assert client.complete.call_count == 3


def test_all_failed_and_no_headlines() -> None:
    client = Mock()
    client.complete.side_effect = RuntimeError("offline")
    result = analyze_headlines(
        "NVDA", [NewsHeadline("First"), NewsHeadline("Second")], client
    )
    assert result.results == []
    assert result.aggregate.overall_score is None
    assert result.aggregate.overall_label == "unavailable"
    assert result.aggregate.successful_count == 0 and result.aggregate.failed_count == 2
    empty = analyze_headlines("NVDA", [], client)
    assert (
        empty.aggregate.failed_count == 0
        and empty.aggregate.overall_label == "unavailable"
    )


def test_changed_headline_rejected() -> None:
    client = Mock()
    client.complete.return_value = json.dumps(
        {
            "headline": "Invented",
            "sentiment": "neutral",
            "confidence": 0.5,
            "brief_reason": "Unknown",
        }
    )
    result = analyze_headlines("NVDA", [NewsHeadline("Actual")], client)
    assert result.aggregate.failed_count == 1 and not result.results
