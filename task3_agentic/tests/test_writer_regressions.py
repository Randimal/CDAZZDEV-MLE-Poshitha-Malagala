"""Focused regressions for the observed writer/token/hedge failures; no live APIs."""

import json
from copy import deepcopy
from unittest.mock import patch

import pytest

from task1_financial.llm import GroqClient, LLMTransportError
from task1_financial.tests.test_transport import status_error
from task3_agentic.memory import PersistentMemory
from task3_agentic.multi_agent import TwoAgentResearch
from task3_agentic.runtime import AgentRuntime
from task3_agentic.schemas import ResearchReport, WriterResearchReport
from task3_agentic.state import compact_schema, observation_view
from task3_agentic.tests.conftest import AS_OF, ScriptedClient, ready, report
from task3_agentic.tests.test_workflows import multi_responses
from task3_agentic.tools import horizon_volatility
from task3_agentic.validation import GroundingError, validate_report_language


@pytest.mark.parametrize("invalid", [None, "", "Used the clarification."])
def test_writer_clarification_feedback_then_valid_report(executor, tmp_path, invalid):
    responses, answer = multi_responses(executor)
    bad = deepcopy(responses[-1])
    bad["output"]["clarification_used"] = invalid
    client = ScriptedClient(responses[:-1] + [bad, responses[-1]])
    run = TwoAgentResearch(
        AgentRuntime(client, executor), PersistentMemory(tmp_path)
    ).run(as_of=AS_OF)
    assert run.report and run.report.clarification_used == answer
    feedback = client.requests[-1]["feedback"]
    assert "clarification_used" in feedback and "verbatim" in feedback
    schema = client.requests[-1]["finish_output_schema"]
    assert "clarification_used" in schema["required"]
    assert schema["properties"]["clarification_used"]["type"] == "string"
    assert schema["properties"]["clarification_used"]["minLength"] == 1
    assert any(
        "clarification_used" in item.content.get("reason", "")
        for item in run.trace
        if item.event == "output_rejected"
    )
    # Repair uses retained evidence; it adds one synthesis call and no tools.
    assert executor.counter == 4
    assert sum(item.event == "decision" for item in run.trace) == 8


def test_writer_validation_retries_are_bounded(executor, tmp_path):
    responses, _ = multi_responses(executor)
    bad = deepcopy(responses[-1])
    bad["output"]["clarification_used"] = None
    client = ScriptedClient(responses[:-1] + [bad] * 2)
    run = TwoAgentResearch(
        AgentRuntime(client, executor), PersistentMemory(tmp_path)
    ).run(as_of=AS_OF)
    assert run.report is None and "Writer output validation exhausted" in run.error
    assert "clarification_used" in run.error
    assert sum(request["stage"] == "writer_final" for request in client.requests) == 3


@pytest.mark.parametrize(
    "claim",
    [
        "Overall the balance sheet is solid.",
        "The company is solvent.",
        "Cash-flow health is strong.",
        "Earnings quality is high.",
    ],
)
def test_unsupported_fundamental_claims_rejected(claim):
    value = report()
    value["financial_health_summary"] = claim
    with pytest.raises(GroundingError, match="financial_health_summary: unsupported"):
        validate_report_language(ResearchReport.model_validate(value), [])


def test_explicit_market_condition_and_unassessed_fundamentals_accepted():
    value = report()
    value["financial_health_summary"] = (
        "Market/technical condition is mixed. OHLCV does not establish balance-sheet strength. "
        "Cash-flow health is not assessed. Earnings quality is unavailable. "
        "Headlines raise cash-flow risk."
    )
    validate_report_language(ResearchReport.model_validate(value), [])


def test_compact_planner_payload_preserves_full_state_and_ids(executor):
    price = executor.invoke("researcher", "get_price_data", {"ticker": "NVDA"})
    news = executor.invoke("researcher", "get_news", {"ticker": "NVDA"})
    news.output["headlines"][0]["url"] = "https://example.com/" + "x" * 2000
    original = deepcopy(price.output)
    view = observation_view(price)
    assert "rows" not in view["output"]
    assert view["output"]["row_count"] == 300
    assert view["output"]["summary"] == price.output["summary"]
    assert view["output"]["latest_indicators"] == price.output["latest_indicators"]
    assert view["evidence_id"] == price.evidence_id and price.output == original
    assert "url" not in observation_view(news)["output"]["headlines"][0]
    client = ScriptedClient(
        [ready(), report(price_id=price.evidence_id, research_id=news.evidence_id)]
    )
    result = AgentRuntime(client, executor).run(
        "Research NVDA",
        role="researcher",
        stage="single_research",
        output_schema=ResearchReport,
        as_of=AS_OF,
        observations=[price, news],
    )
    assert result.output
    payload = client.requests[0]
    assert set(payload["allowed_evidence_ids"]) == {price.evidence_id, news.evidence_id}
    assert '"url":' not in json.dumps(payload)
    assert news.output["headlines"][0]["url"] not in json.dumps(payload)
    assert all(set(item) == {"title"} for item in payload["available_headlines"])
    assert len(json.dumps(view)) < len(json.dumps(price.model_dump())) / 10


def test_schema_compaction_preserves_constraints_and_title_field():
    schema = WriterResearchReport.model_json_schema()
    compact = compact_schema(schema)
    assert len(json.dumps(compact)) < len(json.dumps(schema))
    assert compact["required"] == schema["required"]
    assert compact["properties"]["clarification_used"]["minLength"] == 1
    assert "title" in compact["$defs"]["SharePriceRisk"]["properties"]


def test_historical_horizon_scaling_and_annualized_wording():
    annualized = 0.3734
    scaled = horizon_volatility(annualized)
    assert scaled == pytest.approx(annualized * (90 / 252) ** 0.5)
    value = report()
    value["hedge_strategy_recommendation"]["data_driven_rationale"] = (
        "The 90-day expected move is 37.34%."
    )
    with pytest.raises(GroundingError, match=r"sqrt\(90 / 252\)"):
        validate_report_language(ResearchReport.model_validate(value), [annualized])
    value["hedge_strategy_recommendation"]["data_driven_rationale"] = (
        f"Annualized historical volatility is 37.34%. The 90-day one-standard-deviation move is {100 * scaled:.2f}%. "
        "This assumes 90 trading days and constant/independent return variance, not a forecast."
    )
    validate_report_language(ResearchReport.model_validate(value), [annualized])


def test_optimal_put_claim_requires_missing_option_data():
    value = report()
    value["hedge_strategy_recommendation"]["strategy"] = (
        "The optimal put strike is $150."
    )
    with pytest.raises(GroundingError, match="option-chain"):
        validate_report_language(ResearchReport.model_validate(value), [])
    value["hedge_strategy_recommendation"]["strategy"] = (
        "Cannot determine an optimal put strike without option-chain data; consider a protective put concept."
    )
    validate_report_language(ResearchReport.model_validate(value), [])


def test_provider_status_and_safe_code_logged_without_raw_body(monkeypatch, caplog):
    monkeypatch.setenv("GROQ_API_KEY", "test-placeholder")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    exc = status_error(400)
    exc.body = {
        "error": {"code": "context_length_exceeded", "message": "private raw response"}
    }
    with patch("task1_financial.llm.Groq") as sdk:
        sdk.return_value.chat.completions.create.side_effect = exc
        with pytest.raises(LLMTransportError) as caught:
            GroqClient().complete("system", "user")
    assert caught.value.status_code == 400
    assert caught.value.error_code == "context_length_exceeded"
    assert "status=400" in caplog.text and "context_length_exceeded" in caplog.text
    assert "private raw response" not in caplog.text
    assert sdk.return_value.chat.completions.create.call_count == 1
