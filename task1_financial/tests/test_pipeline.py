from unittest.mock import Mock, PropertyMock

import numpy as np
import pandas as pd
import pytest

from shared.config import PipelineConfig
from task1_financial.data_pipeline import build_summary, run_pipeline, validate_history
from task1_financial.models import DataFetchError, InvalidHistoryError, MomentumResult
from task1_financial.news import fetch_news, parse_news


def test_pipeline_mocked(prices: pd.DataFrame) -> None:
    client = Mock()
    client.history.return_value = prices
    client.info = {}
    client.get_news.return_value = [{"title": f"Headline {i}"} for i in range(10)]
    result = run_pipeline(client=client)
    assert result.summary["pe_ratio"] is None
    assert result.summary["ticker"] == "NVDA"
    assert len(result.news) == 10
    assert result.data.SMA_200.notna().iloc[-1]
    assert client.history.call_args.kwargs["period"] == "2y"
    assert client.history.call_args.kwargs["interval"] == "1d"


def test_summary_ytd_holiday_and_missing_pe() -> None:
    data = pd.DataFrame(
        {"Close": [100, 110, 120], "High": [102, 112, 122], "Low": [98, 108, 118]},
        index=pd.to_datetime(["2025-12-31", "2026-01-02", "2026-01-05"]),
    )
    result = build_summary(
        "TEST",
        data,
        {},
        MomentumResult("NEUTRAL", 0, []),
        as_of=pd.Timestamp("2026-01-05"),
    )
    assert result["ytd_return_pct"] == pytest.approx(20)
    assert result["pe_ratio"] is None
    assert result["52_week_high"] == 122
    assert result["52_week_low"] == 98
    missing = build_summary(
        "TEST",
        data.iloc[1:],
        None,
        MomentumResult("NEUTRAL", 0, []),
        as_of=pd.Timestamp("2026-01-05"),
    )
    assert missing["ytd_return_pct"] is None


@pytest.mark.parametrize(
    "kind", ["empty", "short", "missing_column", "null", "negative", "bad_index"]
)
def test_invalid_history(prices: pd.DataFrame, kind: str) -> None:
    if kind == "empty":
        prices = prices.iloc[:0]
    elif kind == "short":
        prices = prices.iloc[:199]
    elif kind == "missing_column":
        prices = prices.drop(columns="Volume")
    elif kind == "null":
        prices.iloc[-1, prices.columns.get_loc("Close")] = np.nan
    elif kind == "negative":
        prices.iloc[-1, prices.columns.get_loc("Volume")] = -1
    else:
        prices = prices.reset_index(drop=True)
    with pytest.raises((DataFetchError, InvalidHistoryError)):
        validate_history(prices)


def test_history_gap_and_duplicate(prices: pd.DataFrame) -> None:
    prices.iloc[10, 0] = np.nan
    result = validate_history(pd.concat([prices, prices.iloc[-1:]]))
    assert len(result) == len(prices)
    assert result.iloc[10].isna().all()


def test_network_error_is_domain_error() -> None:
    client = Mock()
    client.history.side_effect = ConnectionError("offline")
    with pytest.raises(DataFetchError, match="offline"):
        run_pipeline(client=client)


def test_optional_failures(
    prices: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    client = Mock()
    client.history.return_value = prices
    type(client).info = PropertyMock(side_effect=ConnectionError("metadata offline"))
    client.get_news.side_effect = ConnectionError("news offline")
    result = run_pipeline(client=client)
    assert result.news == []
    assert result.summary["pe_ratio"] is None
    assert "Only 0 of 10" in caplog.text


def test_malformed_news() -> None:
    raw = [
        None,
        7,
        {},
        {"content": None},
        {"title": " "},
        {
            "title": "Old",
            "publisher": "Desk",
            "providerPublishTime": 0,
            "link": "https://example.com",
        },
        {
            "content": {
                "title": "New",
                "provider": {"displayName": "Wire"},
                "pubDate": "2026-10-06T12:00:00Z",
                "canonicalUrl": {"url": "https://example.com/new"},
            }
        },
        {"title": "Old"},
        {
            "title": "Bad fields",
            "publisher": {},
            "providerPublishTime": float("nan"),
            "canonicalUrl": 3,
        },
    ]
    parsed = parse_news(raw)
    assert [entry.title for entry in parsed] == ["Old", "New", "Bad fields"]
    assert parsed[0].published_at == "1970-01-01T00:00:00+00:00"
    assert parsed[1].publisher == "Wire"
    assert parsed[2].publisher is None and parsed[2].published_at is None
    assert parse_news({"invalid": []}) == []


def test_legacy_news_fallback(caplog: pytest.LogCaptureFixture) -> None:
    client = Mock()
    client.get_news.side_effect = TypeError("old API")
    client.news = [{"title": "Real headline"}]
    assert len(fetch_news(client)) == 1
    assert "Only 1 of 10" in caplog.text


def test_short_period_rejected() -> None:
    with pytest.raises(ValueError):
        PipelineConfig(period="1y")
