"""Live-failure regressions using the real graph with deterministic decisions."""

from unittest.mock import patch

from task1_financial.llm import LLMTransportError
from task3_agentic.memory import PersistentMemory
from task3_agentic.runtime import AgentRuntime
from task3_agentic.schemas import ResearchReport
from task3_agentic.single_agent import SingleResearchAgent
from task3_agentic.tests.conftest import AS_OF, ScriptedClient, ready, report, tool
from task3_agentic.tools import tool_call_key


def test_failed_call_normalization_includes_defaults() -> None:
    assert tool_call_key("get_news", {"ticker": " nvda "}) == tool_call_key(
        "get_news", {"n": 10, "ticker": "NVDA"}
    )


def test_duplicate_failed_call_blocked_then_other_source_succeeds(executor, tmp_path):
    executor.tools.news_client_factory.return_value.get_news.return_value = []
    client = ScriptedClient(
        [
            tool("get_news", ticker="NVDA"),
            tool("get_news", n=10, ticker=" nvda "),
            tool("web_search", query="NVDA risks"),
            tool("get_price_data", ticker="NVDA"),
            ready(),
            report(price_id="obs-3", research_id="obs-2"),
        ]
    )
    run = SingleResearchAgent(
        AgentRuntime(client, executor), PersistentMemory(tmp_path)
    ).run(as_of=AS_OF)
    assert run.report is not None
    assert [item.tool_name for item in run.observations].count("get_news") == 1
    assert any(item.event == "duplicate_failed_call_blocked" for item in run.trace)
    assert "blocked" in client.requests[2]["feedback"]
    assert len(client.requests[1]["failed_tool_calls"]) == 1


def test_changed_search_arguments_allowed_after_failure(executor, tmp_path):
    executor.tools.search.side_effect = [
        [],
        [{"title": "Policy concern", "href": "https://example.com/policy"}],
    ]
    client = ScriptedClient(
        [
            tool("web_search", query="NVDA too narrow"),
            tool("web_search", query="NVDA broader risks"),
            tool("get_price_data", ticker="NVDA"),
            ready(),
            report(price_id="obs-3", research_id="obs-2"),
        ]
    )
    run = SingleResearchAgent(
        AgentRuntime(client, executor), PersistentMemory(tmp_path)
    ).run(as_of=AS_OF)
    assert run.report and executor.tools.search.call_count == 2


def test_transport_exhaustion_does_not_spend_planning_budget(executor):
    client = ScriptedClient([LLMTransportError("rate_limit")])
    result = AgentRuntime(client, executor).run(
        "Research NVDA",
        role="researcher",
        stage="single_research",
        output_schema=ResearchReport,
        as_of=AS_OF,
    )
    assert result.output is None and "rate_limit" in result.error
    assert len(client.requests) == 1 and executor.counter == 0
    assert not any(item.event == "decision" for item in result.trace)
    assert "budget" not in result.error


def test_transient_legacy_client_failure_does_not_consume_decision(executor, tmp_path):
    client = ScriptedClient(
        [
            ConnectionError("private"),
            tool("get_price_data", ticker="NVDA"),
            tool("web_search", query="NVDA risk"),
            ready(),
            report(price_id="obs-1", research_id="obs-2"),
        ]
    )
    with patch("task3_agentic.runtime.time.sleep") as sleep:
        run = SingleResearchAgent(
            AgentRuntime(client, executor, max_decisions=3), PersistentMemory(tmp_path)
        ).run(as_of=AS_OF)
    assert run.report and len(client.requests) == 5
    assert client.requests[1]["remaining_decisions"] == 3
    sleep.assert_called_once_with(1)


def test_qualitative_only_final_rejected_then_quantitative_added(executor, tmp_path):
    client = ScriptedClient(
        [
            tool("web_search", query="NVDA risks"),
            ready(),
            tool("calculate_volatility", ticker="NVDA", window=60),
            ready(),
            report(price_id="obs-2", research_id="obs-1"),
        ]
    )
    run = SingleResearchAgent(
        AgentRuntime(client, executor), PersistentMemory(tmp_path)
    ).run(as_of=AS_OF)
    assert run.report
    assert client.requests[1]["report_evidence_coverage"] == {
        "quantitative": False,
        "qualitative": True,
    }
    assert any(item.event == "output_rejected" for item in run.trace)
