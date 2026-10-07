"""Focused graph/synthesis/coordination regressions with no live dependencies."""

from copy import deepcopy

import pytest

from task3_agentic.demonstrations import ControlledFailureExecutor
from task3_agentic.memory import PersistentMemory
from task3_agentic.multi_agent import TwoAgentResearch
from task3_agentic.runtime import AgentRuntime
from task3_agentic.schemas import ResearchReport
from task3_agentic.single_agent import SingleResearchAgent
from task3_agentic.tests.conftest import (
    AS_OF,
    ScriptedClient,
    finish,
    ready,
    report,
    tool,
)
from task3_agentic.tests.test_workflows import multi_responses
from task3_agentic.validation import GroundingError, validate_report_language


def research_responses() -> list[dict]:
    return [
        tool("web_search", query="NVDA market risks"),
        tool("get_price_data", ticker="NVDA"),
        ready(),
    ]


def test_final_synthesis_is_separate_compact_and_successful(executor, tmp_path):
    client = ScriptedClient(
        research_responses() + [report(price_id="obs-2", research_id="obs-1")]
    )
    run = SingleResearchAgent(
        AgentRuntime(client, executor), PersistentMemory(tmp_path)
    ).run(as_of=AS_OF)
    assert run.report and executor.counter == 2
    planners, synthesis = client.requests[:-1], client.requests[-1]
    assert all("finish_output_schema" not in request for request in planners)
    assert all(
        "output" not in request["action_schema"]["properties"] for request in planners
    )
    assert synthesis["stage"] == "final_synthesis" and "report_schema" in synthesis
    assert "allowed_tools" not in synthesis  # Synthesis cannot execute tools.
    assert set(synthesis["allowed_evidence_ids"]) == {"obs-1", "obs-2"}
    assert "rows" not in synthesis["observations"][1]["output"]
    assert len(run.observations[1].output["rows"]) == 300
    assert (
        synthesis["observations"][0]["output"]["results"][0]["source"] == "example.com"
    )
    assert [event.event for event in run.trace].count("synthesis_started") == 1
    assert [event.event for event in run.trace].count("decision") == 3


@pytest.mark.parametrize("failure", ["json_parse", "schema_validation", "grounding"])
def test_final_synthesis_failure_gets_only_one_targeted_repair(
    executor, tmp_path, failure
):
    bad = report(price_id="obs-2", research_id="obs-1")
    if failure == "json_parse":
        pass  # The injected client below returns malformed JSON at synthesis.
    elif failure == "schema_validation":
        bad["top_three_risks"] = bad["top_three_risks"][:2]
    else:
        bad["top_three_risks"][0]["evidence_ids"] = ["missing-evidence"]
    client = ScriptedClient(research_responses() + [bad, bad])
    # Override only malformed JSON generation; other cases use normal scripted JSON.
    if failure == "json_parse":
        original = client.complete

        def malformed(system, user):
            text = original(system, user)
            return (
                "{broken" if client.requests[-1]["stage"] == "final_synthesis" else text
            )

        client.complete = malformed
    run = SingleResearchAgent(
        AgentRuntime(client, executor), PersistentMemory(tmp_path)
    ).run(as_of=AS_OF)
    assert run.report is None and "validation exhausted" in run.error
    assert executor.counter == 2 and len(client.requests) == 5
    rejected = [event for event in run.trace if event.event == "output_rejected"]
    assert len(rejected) == 2 and all(
        event.content["category"] == failure for event in rejected
    )
    feedback = client.requests[-1]["repair_feedback"]
    assert {
        "json_parse": "JSON syntax",
        "schema_validation": "top_three_risks",
        "grounding": "evidence_ids",
    }[failure] in feedback
    assert client.requests[-1]["observations"] == client.requests[-2]["observations"]
    assert not list(tmp_path.glob("NVDA_*.json"))


def test_final_synthesis_repair_accepts_corrected_output_without_replanning(
    executor, tmp_path
):
    valid = report(price_id="obs-2", research_id="obs-1")
    bad = deepcopy(valid)
    bad["top_three_risks"] = []
    client = ScriptedClient(research_responses() + [bad, valid])
    run = SingleResearchAgent(
        AgentRuntime(client, executor), PersistentMemory(tmp_path)
    ).run(as_of=AS_OF)
    assert run.report and executor.counter == 2
    assert (
        sum(request["stage"] == "final_synthesis" for request in client.requests) == 2
    )
    assert sum(event.event == "decision" for event in run.trace) == 3


def test_critique_rejects_metric_already_supplied_in_brief(executor, tmp_path):
    responses, _ = multi_responses(executor)
    bad_request = deepcopy(responses[3])
    bad_request["output"]["requested_metric"] = "current_price"
    client = ScriptedClient(responses[:3] + [bad_request] + responses[3:])
    run = TwoAgentResearch(
        AgentRuntime(client, executor), PersistentMemory(tmp_path)
    ).run(as_of=AS_OF)
    assert run.report and "already supplied" in client.requests[4]["feedback"]
    assert sum(event.event == "critique_request" for event in run.trace) == 1
    assert executor.counter == 4


def test_writer_headlines_enable_analyst_sentiment_clarification(executor, tmp_path):
    original, _ = multi_responses(executor)
    question = "What is the sentiment score of the retrieved demand headline?"
    answer = "The retrieved headline has an aggregate sentiment score of -0.8."
    request = {
        "question": question,
        "requested_metric": "sentiment_score",
        "reason": "The brief has price data but lacks the news direction needed for risks.",
    }
    clarification = {
        "question": question,
        "answer": answer,
        "metric": {
            "name": "sentiment_score",
            "value": -0.8,
            "evidence_id": "obs-3",
            "path": "aggregate.overall_score",
        },
        "limitations": ["One headline, not complete market coverage."],
    }
    final = report(price_id="analyst_brief", research_id="obs-4", clarification=answer)
    final["top_three_risks"][0]["evidence_ids"].append("analyst_clarification")
    client = ScriptedClient(
        original[:3]
        + [
            finish(request),
            tool("llm_sentiment", headlines=[{"title": "Demand concern"}]),
            finish(clarification),
            original[6],
            finish(final),
        ]
    )
    run = TwoAgentResearch(
        AgentRuntime(client, executor), PersistentMemory(tmp_path)
    ).run(as_of=AS_OF)
    assert run.report and run.report.clarification_used == answer
    assert run.critique_request.requested_metric == "sentiment_score"
    sentiment = next(
        item for item in run.observations if item.tool_name == "llm_sentiment"
    )
    assert sentiment.role == "analyst" and sentiment.success
    assert all(
        item.tool_name in {"get_news", "web_search"}
        for item in run.observations
        if item.role == "writer"
    )


def test_controlled_failure_uses_real_graph_then_autonomous_changed_query(
    executor, tmp_path
):
    injected = ControlledFailureExecutor(executor.tools, executor.tracer)
    client = ScriptedClient(
        [
            tool("web_search", query="NVDA initial risks"),
            tool("web_search", query="NVDA revised research"),
            tool("get_price_data", ticker="NVDA"),
            ready(),
            report(price_id="obs-3", research_id="obs-2"),
        ]
    )
    run = SingleResearchAgent(
        AgentRuntime(client, injected), PersistentMemory(tmp_path)
    ).run(as_of=AS_OF)
    assert run.report and injected.injected_tool == "web_search"
    assert [item.success for item in run.observations] == [False, True, True]
    assert client.requests[1]["observations"][0]["success"] is False
    assert injected.tools.search.call_count == 1  # The first call was injected.
    assert len(injected.tracer.path.read_text().splitlines()) == 3


@pytest.mark.parametrize(
    "claim", ["Implied volatility is cheap.", "The option premium is attractive."]
)
def test_option_valuation_claims_require_unavailable_options_data(claim):
    value = report()
    value["hedge_strategy_recommendation"]["data_driven_rationale"] = claim
    with pytest.raises(GroundingError, match="option-chain"):
        validate_report_language(ResearchReport.model_validate(value), [])


def test_good_solvency_claim_requires_unavailable_fundamentals():
    value = report()
    value["financial_health_summary"] = "Good solvency."
    with pytest.raises(GroundingError, match="unsupported fundamental"):
        validate_report_language(ResearchReport.model_validate(value), [])
