import json
from unittest.mock import Mock

import numpy as np
import pytest

from task1_financial.data_pipeline import run_pipeline
from task1_financial.llm import GroqClient
from task1_financial.models import PipelineResult
from task3_agentic.tools import FinancialTools, ToolExecutor
from task3_agentic.tracing import OUTPUT_LIMIT, ToolTracer


@pytest.mark.parametrize(
    "name,args,expected_key",
    [
        ("get_price_data", {"ticker": "NVDA", "period": "2y"}, "rows"),
        ("get_news", {"ticker": "NVDA", "n": 10}, "headlines"),
        (
            "calculate_volatility",
            {"ticker": "NVDA", "window": 60},
            "annualized_volatility",
        ),
        ("llm_sentiment", {"headlines": [{"title": "Demand concern"}]}, "aggregate"),
        ("web_search", {"query": "NVDA policy"}, "results"),
    ],
)
def test_tool_shapes(
    executor: ToolExecutor, name: str, args: dict, expected_key: str
) -> None:
    observation = executor.invoke("researcher", name, args)
    assert observation.success
    assert isinstance(observation.output, dict) and expected_key in observation.output
    if name == "get_price_data":
        row = observation.output["rows"][-1]
        assert {
            "date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "SMA_50",
            "SMA_200",
            "RSI",
            "MACD",
        } <= row.keys()
    elif name == "get_news":
        assert {"title", "publisher", "published_at", "url"} == observation.output[
            "headlines"
        ][0].keys()
    elif name == "web_search":
        assert {"title", "url", "snippet"} == observation.output["results"][0].keys()
    elif name == "llm_sentiment":
        assert observation.output["aggregate"]["negative_count"] == 1


def test_volatility_formula_and_price_reuse(
    services: FinancialTools, prices_result: PipelineResult
) -> None:
    services.get_price_data("NVDA", "2y")
    result = services.calculate_volatility("NVDA", 30)
    closes = prices_result.data.Close.to_numpy()
    returns = closes[1:] / closes[:-1] - 1
    assert result["annualized_volatility"] == pytest.approx(
        np.std(returns[-30:], ddof=1) * np.sqrt(252)
    )
    assert result["unit"] == "fraction" and result["observations"] == 30
    services.pipeline.assert_called_once()


@pytest.mark.parametrize(
    "role,tool_name,args",
    [
        ("analyst", "get_news", {"ticker": "NVDA"}),
        ("analyst", "web_search", {"query": "NVDA"}),
        ("writer", "get_price_data", {"ticker": "NVDA"}),
        ("writer", "calculate_volatility", {"ticker": "NVDA"}),
        ("writer", "llm_sentiment", {"headlines": [{"title": "Demand concern"}]}),
    ],
)
def test_enforced_role_permissions(
    executor: ToolExecutor, role: str, tool_name: str, args: dict
) -> None:
    result = executor.invoke(role, tool_name, args)
    assert not result.success and "PermissionError" in result.error
    executor.tools.pipeline.assert_not_called()
    executor.tools.news_client_factory.assert_not_called()
    executor.tools.search.assert_not_called()


def test_cache_cannot_bypass_role_permissions(executor: ToolExecutor) -> None:
    assert executor.invoke("analyst", "get_price_data", {"ticker": "NVDA"}).success
    assert not executor.invoke("writer", "get_price_data", {"ticker": "NVDA"}).success


def test_failures_empty_results_and_invalid_arguments(executor: ToolExecutor) -> None:
    executor.tools.search.side_effect = RuntimeError("sensitive provider error")
    failed = executor.invoke("writer", "web_search", {"query": "NVDA"})
    assert not failed.success and "sensitive" not in failed.error
    executor.tools.search.side_effect = None
    executor.tools.search.return_value = []
    assert not executor.invoke("writer", "web_search", {"query": "NVDA"}).success
    assert not executor.invoke(
        "analyst", "calculate_volatility", {"ticker": "NVDA", "window": 1}
    ).success
    assert not executor.invoke("analyst", "get_price_data", {"ticker": "OTHER"}).success
    assert not executor.invoke(
        "analyst", "llm_sentiment", {"headlines": [{"title": "Invented news"}]}
    ).success


def test_tool_cache_and_every_invocation_logged(executor: ToolExecutor) -> None:
    first = executor.invoke("analyst", "get_price_data", {"ticker": "NVDA"})
    second = executor.invoke("analyst", "get_price_data", {"ticker": "NVDA"})
    assert first.success and second.cache_hit
    executor.tools.pipeline.assert_called_once()
    entries = [
        json.loads(line)
        for line in executor.tracer.path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(entries) == 2
    assert entries[1]["cache_hit"] is True
    for entry in entries:
        assert {
            "timestamp",
            "tool_name",
            "input_arguments",
            "output",
            "duration_ms",
            "success",
        } <= entry.keys()
        assert entry["duration_ms"] >= 0
        assert len(entry["output"]) <= OUTPUT_LIMIT


def test_truncation_and_no_secret_logging(
    executor: ToolExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "gsk_TestOnlySecretCredential123456789"
    monkeypatch.setenv("GROQ_API_KEY", secret)
    tracer = ToolTracer(executor.tracer.path)
    tracer.record(
        role="writer",
        tool_name="web_search",
        arguments={"query": f"NVDA {secret}", "api_key": secret},
        output={"message": secret + "x" * 400, "Authorization": f"Bearer {secret}"},
        duration_ms=2,
        success=True,
    )
    text = tracer.path.read_text(encoding="utf-8")
    entry = json.loads(text)
    assert secret not in text and "[REDACTED]" in text
    assert len(entry["output"]) == 200
    assert entry["input_arguments"]["api_key"] == "[REDACTED]"


def test_price_pipeline_opt_out_of_news(prices_result: PipelineResult) -> None:
    client = Mock()
    client.history.return_value = prices_result.data
    client.info = {}
    assert run_pipeline(client=client, include_news=False).news == []
    client.get_news.assert_not_called()


def test_groq_token_configuration_is_bounded() -> None:
    with pytest.raises(ValueError):
        GroqClient(max_completion_tokens=10000)
