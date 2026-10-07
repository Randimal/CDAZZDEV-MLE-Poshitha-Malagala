import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from task1_financial.indicators import add_indicators
from task1_financial.json_utils import json_payload, json_safe
from task1_financial.llm_models import TechnicalRecommendation
from task1_financial.models import MomentumResult, NewsHeadline, PipelineResult
from task1_financial.prompts import RECOMMENDATION_SYSTEM
from task1_financial.recommendation import get_recommendation, recommendation_payload
from task1_financial.report import generate_report
from task1_financial.sentiment import aggregate_sentiment
from task1_financial.tests.test_llm import REASONING


def test_recommendation_prompt_precise_threshold_language() -> None:
    assert "overbought only when RSI >= 70" in RECOMMENDATION_SYSTEM
    assert "oversold only when RSI <= 30" in RECOMMENDATION_SYSTEM
    assert "60 up to but excluding 70" in RECOMMENDATION_SYSTEM
    assert "approaching overbought, not overbought" in RECOMMENDATION_SYSTEM
    assert "price is strictly above BB_upper" in RECOMMENDATION_SYSTEM
    assert "touching or approaching the upper band is not a breakout" in (
        RECOMMENDATION_SYSTEM
    )
    assert "short-term overextension risk" in RECOMMENDATION_SYSTEM


@pytest.fixture
def pipeline_result(prices: pd.DataFrame) -> PipelineResult:
    return PipelineResult(
        "TEST",
        add_indicators(prices),
        [
            NewsHeadline(title)
            for title in ("First <script>bad</script>", "Second", "Third", "Fourth")
        ],
        {
            "current_price": 250.0,
            "52_week_high": 252.0,
            "52_week_low": 100.0,
            "pe_ratio": None,
            "ytd_return_pct": 10.0,
            "momentum_signal": "BULLISH",
        },
        MomentumResult("BULLISH", 5, ["Price above average"]),
    )


def test_numpy_pandas_conversion() -> None:
    values = {
        "integer": np.int64(3),
        "float": np.float64(1.2),
        "bool": np.bool_(True),
        "array": np.array([1, np.nan, np.inf]),
        "missing": pd.NA,
        "time": pd.Timestamp("2026-10-06"),
        "nat": pd.NaT,
        "series": pd.Series([1, pd.NA]),
        "frame": pd.DataFrame({"x": [np.nan]}),
    }
    result = json.loads(json_payload(values))
    assert result["integer"] == 3 and result["bool"] is True
    assert result["array"] == [1.0, None, None]
    assert result["missing"] is None and result["nat"] is None
    assert result["time"] == "2026-10-06T00:00:00"
    assert result["frame"] == [{"x": None}]
    assert result["series"] == {"0": 1, "1": None}
    with pytest.raises(TypeError):
        json_safe(object())


def test_recommendation_payload_and_retry(pipeline_result: PipelineResult) -> None:
    pipeline_result.data.iloc[-1, pipeline_result.data.columns.get_loc("RSI")] = np.nan
    aggregate = aggregate_sentiment([], failed_count=4)
    payload = recommendation_payload(pipeline_result, aggregate)
    assert payload["indicators"]["RSI"] is None
    assert payload["deterministic_momentum"] == "BULLISH"
    assert payload["aggregate_news_sentiment"]["failed_count"] == 4
    assert len(payload["indicators"]) == 9
    assert payload["technical_facts"]["rsi_regime"] is None
    assert payload["technical_facts"]["trend_structure"] == "bullish"
    assert "rows" not in payload and "data" not in payload
    client = Mock()
    client.complete.side_effect = [
        '{"signal":"BUY"}',
        json.dumps({"signal": "HOLD", "reasoning": REASONING}),
    ]
    result = get_recommendation(pipeline_result, client, aggregate, attempts=2)
    assert result.signal == "HOLD" and client.complete.call_count == 2
    sent = client.complete.call_args.args[1].split("\n", 1)[1]
    assert json.loads(sent)["indicators"]["RSI"] is None
    assert json.loads(sent)["technical_facts"] == payload["technical_facts"]
    assert "NaN" not in sent


@pytest.mark.parametrize("pe", [None, 35.2])
def test_report_generation(
    pipeline_result: PipelineResult, tmp_path: Path, pe: float | None
) -> None:
    pipeline_result.summary["pe_ratio"] = pe
    recommendation = TechnicalRecommendation(signal="BUY", reasoning=REASONING)
    artifacts = generate_report(
        pipeline_result, aggregate_sentiment([], 4), recommendation, tmp_path
    )
    markdown = artifacts.markdown.read_text(encoding="utf-8")
    html = artifacts.html.read_text(encoding="utf-8")
    for section in (
        "Company Snapshot",
        "Technical Outlook",
        "News Sentiment",
        "Recommendation",
        "Risk Disclaimer",
    ):
        assert section in markdown and section in html
    assert artifacts.chart.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifacts.chart.name in html and artifacts.chart.name in markdown
    assert "<script>bad</script>" not in html
    assert "&lt;script&gt;bad&lt;/script&gt;" in html
    assert "Fourth" not in html and "Third" in html
    assert "BUY:" in html and "failed: 4" in html
    assert f"Trailing PE: {'Unavailable' if pe is None else '35.20'}" in html
    assert "@page" in html and "size: A4" in html


def test_report_without_llm_analysis(
    pipeline_result: PipelineResult, tmp_path: Path
) -> None:
    pipeline_result.news = []
    artifacts = generate_report(pipeline_result, None, None, tmp_path)
    html = artifacts.html.read_text(encoding="utf-8")
    assert "LLM recommendation unavailable" in html
    assert "Aggregate sentiment unavailable" in html
    assert "No usable headlines available" in html
