"""Regressions for live clarification JSON rejection and ungrounded report claims."""

import json
from copy import deepcopy

import pytest

from task1_financial.llm import GroqClient, LLMTransportError
from task3_agentic.memory import PersistentMemory, answer_followup
from task3_agentic.multi_agent import TwoAgentResearch
from task3_agentic.runtime import AgentRuntime
from task3_agentic.schemas import ClarificationResponse, ResearchReport, ToolObservation
from task3_agentic.tests.conftest import AS_OF, finish, report, tool
from task3_agentic.tests.test_workflows import multi_responses
from task3_agentic.tools import horizon_volatility
from task3_agentic.validation import (
    GroundingError,
    grounding_facts,
    validate_report_language,
)


class RepairClient(GroqClient):
    """Mock the provider boundary, retaining Groq's optional text-mode interface."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.modes = []

    def complete(self, system, user, *, json_mode=True):
        self.requests.append(json.loads(user))
        self.modes.append(json_mode)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response)


def json_rejection():
    return LLMTransportError(
        "invalid_request", status_code=400, error_code="json_validate_failed"
    )


def sentiment_flow(executor):
    original, _ = multi_responses(executor)
    question = "What sentiment score do the retrieved headlines support?"
    answer = "1 of 1 headlines successfully analyzed; 0 failed, 0 positive, 1 negative, 0 neutral. Aggregate score is -0.8."
    request = {
        "question": question,
        "requested_metric": "sentiment_score",
        "reason": "News direction is missing from the price brief.",
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
        "limitations": ["Only one retrieved headline; confidence is not calibrated."],
    }
    final = report(price_id="analyst_brief", research_id="obs-4", clarification=answer)
    final["top_three_risks"][0]["evidence_ids"].append("analyst_clarification")
    final["market_sentiment_summary"] = answer
    return original[:3] + [
        finish(request),
        tool("llm_sentiment", headlines=[{"title": "Demand concern"}]),
        finish(clarification),
        original[6],
        finish(final),
    ]


@pytest.mark.parametrize("before_tool", [True, False])
def test_analyst_json_repair_preserves_real_sentiment_and_completion(
    executor, tmp_path, before_tool
):
    responses = sentiment_flow(executor)
    index = 4 if before_tool else 5
    responses.insert(index, json_rejection())
    client = RepairClient(
        responses
        + [
            {
                "answer": "The aggregate score is -0.8.",
                "evidence_ids": ["analyst_clarification"],
            }
        ]
    )
    memory = PersistentMemory(tmp_path)
    workflow = TwoAgentResearch(AgentRuntime(client, executor), memory)
    run = workflow.run(as_of=AS_OF, use_cache=False)
    assert run.report and isinstance(run.clarification, ClarificationResponse)
    assert run.clarification.metric.value == -0.8
    assert client.modes[index : index + 2] == [True, False]
    sentiment = next(
        item for item in run.observations if item.tool_name == "llm_sentiment"
    )
    assert sentiment.role == "analyst" and sentiment.success
    executor.tools.client.complete.assert_called_once()
    final_request = next(
        item for item in client.requests if item["stage"] == "writer_final"
    )
    coverage = final_request["grounding_facts"]["sentiment"][0]
    assert coverage["successful_count"] == coverage["requested_headline_count"] == 1
    assert coverage["failed_count"] == 0 and coverage["negative_count"] == 1
    assert run.report.clarification_used == run.clarification.answer
    before = executor.counter
    assert answer_followup(run, "What was the sentiment?", client).answer
    assert executor.counter == before
    calls = len(client.requests)
    cached = workflow.run(as_of=AS_OF)
    assert cached.cached and cached.report and executor.counter == before
    assert len(client.requests) == calls  # Neither tools nor LLM used on cache hit.


@pytest.mark.parametrize(
    "repair",
    [
        "{broken",
        finish({"question": "q", "answer": "a", "metric": None}),
        json_rejection(),
    ],
)
def test_analyst_repair_exhaustion_never_fabricates_response(executor, repair):
    observed = executor.invoke(
        "analyst", "llm_sentiment", {"headlines": [{"title": "Demand concern"}]}
    )
    client = RepairClient([json_rejection(), repair])
    result = AgentRuntime(client, executor).run(
        "Research NVDA",
        role="analyst",
        stage="analyst_clarify",
        output_schema=ClarificationResponse,
        as_of=AS_OF,
        context={
            "brief": {"metrics": []},
            "request": {"question": "q", "requested_metric": "sentiment_score"},
        },
        observations=[observed],
    )
    assert result.output is None and result.error
    assert "no fabricated output" in result.error
    assert client.modes == [True, False]
    assert executor.counter == 1


def coverage_facts():
    aggregate = {
        "overall_score": 0.5,
        "successful_count": 19,
        "failed_count": 1,
        "positive_count": 15,
        "negative_count": 2,
        "neutral_count": 2,
    }
    observation = ToolObservation(
        evidence_id="obs-sentiment",
        role="analyst",
        tool_name="llm_sentiment",
        arguments={"headlines": [{"title": str(i)} for i in range(20)]},
        success=True,
        output={"aggregate": aggregate},
    )
    return grounding_facts([observation], {})


def test_historical_sigma_is_already_the_90_trading_day_return_scale():
    annualized = 0.3796
    observation = ToolObservation(
        evidence_id="obs-vol",
        role="analyst",
        tool_name="calculate_volatility",
        arguments={},
        success=True,
        output={"annualized_volatility": annualized},
    )
    facts = grounding_facts([observation], {})
    sigma = facts["volatility"][0]["historical_90_trading_day_sigma_pct"]
    assert sigma == pytest.approx(100 * annualized * (90 / 252) ** 0.5)
    assert horizon_volatility(annualized) == pytest.approx(0.22685439)
    value = report()
    value["hedge_strategy_recommendation"]["data_driven_rationale"] = (
        "The historical 90-day one-standard-deviation return scale is 22.7%, assuming 90 trading days; not a forecast."
    )
    validate_report_language(ResearchReport.model_validate(value), [annualized], facts)
    value["top_three_risks"][0]["supporting_evidence"] = (
        "The 90-day volatility is 22.7%, implying a ~5% one-sigma move."
    )
    with pytest.raises(GroundingError, match="without scaling again"):
        validate_report_language(
            ResearchReport.model_validate(value), [annualized], facts
        )


@pytest.mark.parametrize(
    "claim",
    [
        "Offset ~20% of the equity exposure using a sector ETF.",
        "Use a 5% stop-loss.",
        "Buy NVDA futures to hedge.",
        "Use a hedge ratio of 20%.",
        "Buy a put with a strike at 150.",
    ],
)
def test_unsupported_hedge_sizing_or_instrument_rejected(claim):
    value = report()
    value["hedge_strategy_recommendation"]["strategy"] = claim
    with pytest.raises(GroundingError, match="unsupported numerical sizing"):
        validate_report_language(ResearchReport.model_validate(value), [])
    value["hedge_strategy_recommendation"]["strategy"] = (
        "Consider a correlated sector ETF hedge; exact hedge ratio requires unavailable beta/correlation/exposure data."
    )
    validate_report_language(ResearchReport.model_validate(value), [])


def test_sentiment_coverage_counts_do_not_turn_failures_into_success_or_neutral():
    facts = coverage_facts()
    assert (
        facts["sentiment"][0]["total_headline_count"]
        == facts["sentiment"][0]["requested_headline_count"]
        == 20
    )
    value = report()
    value["market_sentiment_summary"] = (
        "19 of 20 headlines were successfully analyzed: 15 positive, 2 negative, 2 neutral; 1 failed."
    )
    validate_report_language(ResearchReport.model_validate(value), [], facts)
    for bad in (
        "Sentiment analysis of 20 recent headlines yields positive overall coverage.",
        "19 headlines were analyzed: 15 positive, 2 negative, 3 neutral.",
    ):
        invalid = deepcopy(value)
        invalid["market_sentiment_summary"] = bad
        with pytest.raises(GroundingError, match="sentiment coverage"):
            validate_report_language(ResearchReport.model_validate(invalid), [], facts)


@pytest.mark.parametrize(
    "claim",
    [
        "PE 30.2 is higher than historical averages.",
        "Valuation is expensive relative to history.",
        "The PE ratio trades at a sector premium.",
    ],
)
def test_pe_comparison_requires_missing_benchmark_evidence(claim):
    value = report()
    value["financial_health_summary"] = claim
    with pytest.raises(GroundingError, match="benchmark evidence"):
        validate_report_language(ResearchReport.model_validate(value), [])
    value["financial_health_summary"] = (
        "PE ratio is 30.2; no historical PE benchmark is available."
    )
    validate_report_language(ResearchReport.model_validate(value), [])
