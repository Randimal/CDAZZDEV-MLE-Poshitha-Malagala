"""Writer action contracts depend on evidence, never a fixed tool sequence."""

import pytest

from task3_agentic.runtime import AgentRuntime
from task3_agentic.schemas import (
    ClarificationRequest,
    QuantitativeBrief,
    ToolObservation,
)
from task3_agentic.tests.conftest import AS_OF, ScriptedClient, finish, tool
from task3_agentic.tests.test_workflows import multi_responses


def review_inputs(executor):
    responses, _ = multi_responses(executor)
    context = {
        "brief": QuantitativeBrief.model_validate(responses[1]["output"]).model_dump()
    }
    return context, responses[3]["output"]


def run_review(client, executor, context, *, observations=None, max_decisions=3):
    return AgentRuntime(client, executor, max_decisions=max_decisions).run(
        "Identify one missing analyst metric after qualitative research.",
        role="writer",
        stage="writer_review",
        output_schema=ClarificationRequest,
        as_of=AS_OF,
        context=context,
        observations=observations,
    )


@pytest.mark.parametrize("failed_tool", [None, "get_news", "web_search"])
def test_no_successful_qualitative_evidence_disallows_finish(executor, failed_tool):
    context, request = review_inputs(executor)
    observations = []
    if failed_tool:
        observations.append(
            ToolObservation(
                evidence_id="failed-qualitative",
                role="writer",
                tool_name=failed_tool,
                arguments={},
                success=False,
                error="Controlled failed observation",
            )
        )
    client = ScriptedClient([finish(request)])
    result = run_review(
        client, executor, context, observations=observations, max_decisions=1
    )
    payload = client.requests[0]
    assert payload["action_schema"]["properties"]["kind"]["const"] == "tool"
    assert "finish_output_schema" not in payload
    assert payload["stage_readiness"]["finish_allowed"] is False
    assert set(payload["allowed_tools"]) == {"get_news", "web_search"}
    assert result.output is None and executor.counter == 0
    # The same restricted model is used for local validation, not just prompting.
    assert any(
        event.event == "llm_failure" and event.content["category"] == "validation"
        for event in result.trace
    )


def assert_finish_allowed(executor, name, arguments):
    context, request = review_inputs(executor)
    observation = executor.invoke("writer", name, arguments)
    assert observation.success
    count = executor.counter
    client = ScriptedClient([finish(request)])
    result = run_review(client, executor, context, observations=[observation])
    payload = client.requests[0]
    assert set(payload["action_schema"]["properties"]["kind"]["enum"]) == {
        "tool",
        "finish",
    }
    assert "finish_output_schema" in payload
    assert payload["stage_readiness"]["finish_allowed"] is True
    assert isinstance(result.output, ClarificationRequest)
    assert result.output.model_dump() == request
    assert executor.counter == count and len(client.requests) == 1


def test_successful_get_news_allows_finish(executor):
    assert_finish_allowed(executor, "get_news", {"ticker": "NVDA", "n": 10})


def test_successful_web_search_allows_finish(executor):
    assert_finish_allowed(executor, "web_search", {"query": "NVDA qualitative risks"})


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("get_news", {"ticker": "NVDA", "n": 10}),
        ("web_search", {"query": "NVDA qualitative risks"}),
    ],
)
def test_llm_autonomously_selects_either_qualitative_tool(executor, name, arguments):
    context, request = review_inputs(executor)
    client = ScriptedClient([tool(name, **arguments), finish(request)])
    original_complete = client.complete

    def complete(system, user):
        if not client.requests:
            assert executor.counter == 0  # Nothing is dispatched before LLM choice.
        return original_complete(system, user)

    client.complete = complete
    result = run_review(client, executor, context)
    assert result.output and result.error is None
    assert [observation.tool_name for observation in result.observations] == [name]
    assert executor.counter == 1 and len(client.requests) == 2
    assert client.requests[0]["action_schema"]["properties"]["kind"]["const"] == "tool"
    assert "finish" in client.requests[1]["action_schema"]["properties"]["kind"]["enum"]
    assert client.requests[1]["observations"][0]["success"] is True
