"""Deterministic service fixtures; no Yahoo, Groq or search requests."""

import json
from datetime import date
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from task1_financial.indicators import add_indicators
from task1_financial.models import MomentumResult, NewsHeadline, PipelineResult
from task3_agentic.tools import FinancialTools, ToolExecutor
from task3_agentic.tracing import ToolTracer

AS_OF = date(2026, 10, 7)


@pytest.fixture(autouse=True)
def offline_news_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a mocked short Yahoo response trigger live RSS access."""
    monkeypatch.setattr("task1_financial.news.fetch_rss_news", lambda *args: [])


class ScriptedClient:
    def __init__(self, responses: list[dict | Exception]) -> None:
        self.responses = iter(responses)
        self.requests: list[dict] = []

    def complete(self, system: str, user: str) -> str:
        self.requests.append(json.loads(user))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return json.dumps(response)


def tool(name: str, **arguments: object) -> dict:
    return {
        "kind": "tool",
        "reason": "Select evidence based on current observations",
        "tool_name": name,
        "arguments": arguments,
    }


def finish(output: dict) -> dict:
    return {
        "kind": "finish",
        "reason": "Available evidence supports a qualified output",
        "output": output,
    }


def report(
    *,
    price_id: str = "obs-1",
    research_id: str = "obs-3",
    clarification: str | None = None,
) -> dict:
    return {
        "ticker": "NVDA",
        "as_of": AS_OF.isoformat(),
        "horizon_days": 90,
        "financial_health_summary": "Price history supports market-condition analysis, not a solvency assessment.",
        "market_sentiment_summary": "News and search coverage is limited.",
        "top_three_risks": [
            {
                "title": title,
                "supporting_evidence": evidence,
                "evidence_ids": [research_id],
            }
            for title, evidence in (
                (
                    "Demand uncertainty",
                    "Retrieved news flags a demand concern, not a forecast.",
                ),
                (
                    "Policy uncertainty",
                    "Retrieved search highlights a policy issue requiring source verification.",
                ),
                (
                    "Market repricing",
                    "Retrieved evidence suggests sensitivity to changing expectations.",
                ),
            )
        ],
        "hedge_strategy_recommendation": {
            "strategy": "Consider a limited-horizon protective put on an existing position.",
            "data_driven_rationale": "Use the retrieved price or historical volatility to assess the amount of downside exposure to protect.",
            "evidence_ids": [price_id],
            "limitations": "No option quotes were retrieved; strike, premium and liquidity need verification. This is a hedge concept, not an executable trade.",
        },
        "limitations": ["Historical observations do not forecast 90-day outcomes."],
        "clarification_used": clarification,
    }


@pytest.fixture
def prices_result() -> PipelineResult:
    close = 100 * np.cumprod(1 + 0.002 + 0.01 * np.sin(np.arange(300)))
    data = pd.DataFrame(
        {
            "Open": close,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": np.full(300, 1000),
        },
        index=pd.bdate_range(end="2026-10-07", periods=300),
    )
    summary = {
        "ticker": "NVDA",
        "current_price": float(close[-1]),
        "52_week_high": float(close.max() + 2),
        "52_week_low": float(close.min() - 2),
        "pe_ratio": None,
        "ytd_return_pct": 10.0,
        "momentum_signal": "BULLISH",
    }
    return PipelineResult(
        "NVDA", add_indicators(data), [], summary, MomentumResult("BULLISH", 3, [])
    )


@pytest.fixture
def services(prices_result: PipelineResult) -> FinancialTools:
    client = Mock()
    client.complete.return_value = json.dumps(
        {
            "headline": "Demand concern",
            "sentiment": "negative",
            "confidence": 0.8,
            "brief_reason": "Demand uncertainty",
        }
    )
    news_client = Mock()
    news_client.get_news.return_value = [
        {"title": "Demand concern", "publisher": "Desk"}
    ]
    return FinancialTools(
        "NVDA",
        client,
        pipeline=Mock(return_value=prices_result),
        news_client_factory=Mock(return_value=news_client),
        search=Mock(
            return_value=[
                {
                    "title": "Policy concern",
                    "href": "https://example.com/policy",
                    "body": "Policy uncertainty",
                }
            ]
        ),
        seed_headlines=[NewsHeadline("Demand concern")],
    )


@pytest.fixture
def executor(services: FinancialTools, tmp_path: Path) -> ToolExecutor:
    return ToolExecutor(services, ToolTracer(tmp_path / "agent_trace.jsonl"))
