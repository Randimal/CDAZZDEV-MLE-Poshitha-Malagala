from copy import deepcopy
from pathlib import Path

from task3_agentic.memory import PersistentMemory, answer_followup
from task3_agentic.multi_agent import TwoAgentResearch
from task3_agentic.runtime import AgentRuntime
from task3_agentic.schemas import QuantitativeBrief
from task3_agentic.single_agent import SingleResearchAgent
from task3_agentic.tests.conftest import (
    AS_OF,
    ScriptedClient,
    finish,
    ready,
    report,
    tool,
)
from task3_agentic.tools import ToolExecutor


def single_responses() -> list[dict]:
    return [
        tool("get_price_data", ticker="NVDA", period="2y"),
        tool("get_news", ticker="NVDA", n=10),
        tool("web_search", query="NVDA risk evidence"),
        ready(),
        report(),
    ]


def multi_responses(executor: ToolExecutor) -> tuple[list[dict], str]:
    price = executor.tools.pipeline.return_value.summary["current_price"]
    daily_returns = executor.tools.pipeline.return_value.data.Close.pct_change(
        fill_method=None
    ).iloc[-60:]
    volatility = float(daily_returns.std(ddof=1) * 252**0.5)
    brief = {
        "ticker": "NVDA",
        "summary": "Retrieved adjusted price with fundamental limitations.",
        "metrics": [
            {
                "name": "current_price",
                "value": price,
                "evidence_id": "obs-1",
                "path": "summary.current_price",
            }
        ],
        "limitations": ["No balance sheet retrieved."],
    }
    question = "What is the 60-day annualized historical volatility as a fraction?"
    answer = f"The retrieved annualized historical volatility is {volatility:.6f} as a fraction."
    request = {
        "question": question,
        "requested_metric": "annualized_volatility",
        "reason": "Quantify the historical risk scale for the hedge.",
    }
    clarification = {
        "question": question,
        "answer": answer,
        "metric": {
            "name": "annualized_volatility",
            "value": volatility,
            "evidence_id": "obs-3",
            "path": "annualized_volatility",
        },
        "limitations": ["Historical volatility is not a 90-day forecast."],
    }
    return [
        tool("get_price_data", ticker="NVDA"),
        finish(brief),
        tool("get_news", ticker="NVDA", n=10),
        finish(request),
        tool("calculate_volatility", ticker="NVDA", window=60),
        finish(clarification),
        tool("web_search", query="NVDA policy risk"),
        finish(
            report(
                price_id="analyst_clarification",
                research_id="obs-4",
                clarification=answer,
            )
        ),
    ], answer


def test_single_observe_replan_and_failure_recovery(
    executor: ToolExecutor, tmp_path: Path
) -> None:
    executor.tools.news_client_factory.return_value.get_news.return_value = []
    client = ScriptedClient(single_responses())
    run = SingleResearchAgent(
        AgentRuntime(client, executor), PersistentMemory(tmp_path / "memory")
    ).run(as_of=AS_OF)
    assert run.report is not None and run.error is None
    assert [item.tool_name for item in run.observations] == [
        "get_price_data",
        "get_news",
        "web_search",
    ]
    assert not run.observations[1].success
    assert client.requests[2]["observations"][-1]["success"] is False
    events = [item.event for item in run.trace]
    assert (
        events.index("decision")
        < events.index("tool_call")
        < events.index("observation")
        < events.index("replan")
    )
    assert events.count("decision") == 4 and events.count("replan") == 3


def test_another_llm_can_choose_different_tool_order(
    executor: ToolExecutor, tmp_path: Path
) -> None:
    client = ScriptedClient(
        [
            tool("web_search", query="NVDA risks"),
            tool("get_price_data", ticker="NVDA"),
            ready(),
            report(price_id="obs-2", research_id="obs-1"),
        ]
    )
    run = SingleResearchAgent(
        AgentRuntime(client, executor), PersistentMemory(tmp_path)
    ).run(as_of=AS_OF)
    assert run.report is not None
    assert [item.tool_name for item in run.observations] == [
        "web_search",
        "get_price_data",
    ]


def test_complete_multi_agent_critique_and_structured_handoff(
    executor: ToolExecutor, tmp_path: Path
) -> None:
    responses, answer = multi_responses(executor)
    client = ScriptedClient(responses)
    run = TwoAgentResearch(
        AgentRuntime(client, executor), PersistentMemory(tmp_path / "memory")
    ).run(as_of=AS_OF)
    assert run.report and isinstance(run.brief, QuantitativeBrief)
    assert run.critique_request.requested_metric == "annualized_volatility"
    assert run.clarification.question == run.critique_request.question
    assert run.report.clarification_used == answer
    assert (
        "analyst_clarification" in run.report.hedge_strategy_recommendation.evidence_ids
    )
    stages = [request["stage"] for request in client.requests]
    assert (
        stages
        == ["analyst_brief"] * 2
        + ["writer_review"] * 2
        + ["analyst_clarify"] * 2
        + ["writer_final"] * 2
    )
    handoff = client.requests[2]["handoff_context"]["brief"]
    assert isinstance(handoff, dict) and isinstance(
        handoff["metrics"][0]["value"], float
    )
    assert client.requests[6]["handoff_context"]["clarification"]["answer"] == answer
    for request in client.requests:
        if request["role"] == "analyst":
            assert set(request["allowed_tools"]) == {
                "get_price_data",
                "calculate_volatility",
                "llm_sentiment",
            }
        if request["role"] == "writer":
            assert set(request["allowed_tools"]) == {"get_news", "web_search"}
    events = [item.event for item in run.trace]
    assert (
        events.count("critique_request") == events.count("clarification_response") == 1
    )
    assert (
        events.index("structured_handoff")
        < events.index("critique_request")
        < events.index("clarification_response")
    )
    executor.tools.pipeline.assert_called_once()


def test_short_term_followup_does_not_fetch_again(
    executor: ToolExecutor, tmp_path: Path
) -> None:
    responses, answer = multi_responses(executor)
    client = ScriptedClient(
        responses + [{"answer": answer, "evidence_ids": ["analyst_clarification"]}]
    )
    run = TwoAgentResearch(
        AgentRuntime(client, executor), PersistentMemory(tmp_path)
    ).run(as_of=AS_OF)
    tool_calls = executor.counter
    lines = executor.tracer.path.read_text(encoding="utf-8").splitlines()
    followup = answer_followup(run, "What volatility did Agent A calculate?", client)
    assert followup.answer == answer
    assert executor.counter == tool_calls
    assert executor.tracer.path.read_text(encoding="utf-8").splitlines() == lines
    executor.tools.pipeline.assert_called_once()


def test_invalid_handoff_metric_replans(executor: ToolExecutor, tmp_path: Path) -> None:
    responses, _ = multi_responses(executor)
    bad = {
        "ticker": "NVDA",
        "summary": "Wrong metric",
        "metrics": [
            {
                "name": "current_price",
                "value": 9999.0,
                "evidence_id": "obs-1",
                "path": "summary.current_price",
            }
        ],
        "limitations": ["Synthetic test"],
    }
    client = ScriptedClient([responses[0], finish(bad)] + responses[1:])
    run = TwoAgentResearch(
        AgentRuntime(client, executor), PersistentMemory(tmp_path)
    ).run(as_of=AS_OF)
    assert run.report
    assert any(item.event == "output_rejected" for item in run.trace)
    assert client.requests[2]["feedback"] is not None


def test_invalid_final_evidence_is_not_accepted(
    executor: ToolExecutor, tmp_path: Path
) -> None:
    client = ScriptedClient([ready()] * 2)
    run = SingleResearchAgent(
        AgentRuntime(client, executor, max_decisions=2), PersistentMemory(tmp_path)
    ).run(as_of=AS_OF)
    assert run.report is None and "budget" in run.error
    assert not list(tmp_path.glob("NVDA_*.json"))


def test_api_failure_is_bounded(executor: ToolExecutor, tmp_path: Path) -> None:
    client = ScriptedClient([RuntimeError("private API failure")] * 2)
    run = SingleResearchAgent(
        AgentRuntime(client, executor, max_decisions=2), PersistentMemory(tmp_path)
    ).run(as_of=AS_OF)
    assert run.report is None and run.error
    assert len(client.requests) == 2
    assert "private API failure" not in str(run.trace)


def test_price_only_report_is_rejected_and_replanned(
    executor: ToolExecutor, tmp_path: Path
) -> None:
    client = ScriptedClient(
        [
            tool("get_price_data", ticker="NVDA"),
            ready(),
            tool("web_search", query="NVDA market evidence"),
            ready(),
            report(price_id="obs-1", research_id="obs-2"),
        ]
    )
    run = SingleResearchAgent(
        AgentRuntime(client, executor), PersistentMemory(tmp_path)
    ).run(as_of=AS_OF)
    assert run.report and any(item.event == "output_rejected" for item in run.trace)


def test_writer_cannot_omit_clarification(
    executor: ToolExecutor, tmp_path: Path
) -> None:
    responses, answer = multi_responses(executor)
    invalid = deepcopy(responses[-1])
    invalid["output"]["clarification_used"] = None
    client = ScriptedClient(responses[:-1] + [invalid, responses[-1]])
    run = TwoAgentResearch(
        AgentRuntime(client, executor), PersistentMemory(tmp_path)
    ).run(as_of=AS_OF)
    assert run.report.clarification_used == answer
    assert any(
        item.event == "output_rejected" and item.role == "writer" for item in run.trace
    )
