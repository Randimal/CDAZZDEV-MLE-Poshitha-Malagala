"""Relationships must preserve confirmations, contradictions and missingness."""

import json

import numpy as np
import pandas as pd
import pytest

from task1_financial.technical_facts import build_technical_facts


def test_bullish_relationships() -> None:
    facts = build_technical_facts(
        {
            "Close": 110,
            "SMA_50": 100,
            "SMA_200": 90,
            "RSI": 60,
            "MACD": 2,
            "MACD_signal": 1,
            "MACD_histogram": 1,
            "BB_lower": 90,
            "BB_middle": 100,
            "BB_upper": 120,
        },
        momentum_signal="BULLISH",
    )
    assert facts["price_vs_sma50_pct"] == pytest.approx(10)
    assert facts["price_vs_sma200_pct"] == pytest.approx(100 * (110 / 90 - 1))
    assert facts["sma50_vs_sma200"] == "above"
    assert facts["trend_structure"] == "bullish"
    assert facts["rsi_regime"] == "strong"
    assert facts["macd_vs_signal"] == "above"
    assert facts["macd_histogram_sign"] == "positive"
    assert facts["bollinger_location"] == "upper_half"
    assert facts["deterministic_momentum"] == "BULLISH"


def test_contradictory_relationships_preserved() -> None:
    latest = {
        "Close": 220,
        "SMA_50": 200,
        "SMA_200": 180,
        "RSI": 78,
        "MACD": 1,
        "MACD_signal": 2,
        "MACD_histogram": -1,
        "BB_lower": 190,
        "BB_middle": 200,
        "BB_upper": 210,
    }
    facts = build_technical_facts(latest, momentum_signal="BULLISH")
    assert facts["trend_structure"] == "bullish"
    assert facts["rsi_regime"] == "overbought"
    assert facts["macd_vs_signal"] == "below"
    assert facts["macd_histogram_sign"] == "negative"
    assert facts["bollinger_location"] == "above_upper"
    # Price above both averages alone does not establish the full ordering.
    latest.update(SMA_50=180, SMA_200=200, RSI=50, MACD=2, MACD_histogram=0)
    mixed = build_technical_facts(latest)
    assert mixed["trend_structure"] == "neutral" and mixed["rsi_regime"] == "neutral"
    assert mixed["macd_vs_signal"] == "equal" and mixed["macd_histogram_sign"] == "zero"


def test_missing_or_invalid_relationship_inputs() -> None:
    facts = build_technical_facts(
        {
            "Close": np.nan,
            "SMA_50": 0,
            "SMA_200": pd.NA,
            "RSI": float("inf"),
            "MACD": None,
            "MACD_histogram": "invalid",
            "BB_lower": 1,
            "BB_middle": 2,
            "BB_upper": 3,
        }
    )
    assert all(value is None for value in facts.values())
    assert "NaN" not in json.dumps(facts, allow_nan=False)
