"""Task 3 budgets/pacing only: mocked SDK and injectable clocks/sleep."""

import json
from unittest.mock import Mock, patch

import pytest

from task1_financial.llm import GroqClient, LLMTransportError
from task1_financial.prompts import SENTIMENT_SYSTEM
from task1_financial.tests.test_transport import status_error
from task3_agentic.demo_client import DemoPacer, Task3GroqClient, TokenBudgets
from task3_agentic.prompts import FOLLOWUP_SYSTEM
from task3_agentic.runtime import AgentRuntime
from task3_agentic.schemas import (
    ClarificationResponse,
    QuantitativeBrief,
    ResearchReport,
)
from task3_agentic.tests.conftest import AS_OF, ready, report, tool
from task3_agentic.tests.test_workflows import multi_responses


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Task3GroqClient:
    monkeypatch.setenv("GROQ_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("GROQ_MODEL", "test-model-without-reasoning")
    with patch("task1_financial.llm.Groq"):
        value = Task3GroqClient(clock=lambda: 100.0, sleep=Mock())
    value._client.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content='{"valid":true}'))
    ]
    return value


@pytest.mark.parametrize(
    ("stage", "system", "expected"),
    [
        ("single_research", "planner", 768),
        ("writer_review", "critique", 512),
        ("analyst_brief", "brief", 1200),
        ("analyst_clarify", "clarification", 900),
        ("writer_final", "writer", 1800),
        ("final_synthesis", "synthesis", 1800),
        (None, FOLLOWUP_SYSTEM, 640),
        (None, SENTIMENT_SYSTEM, 384),
    ],
)
def test_request_budgets_preserve_larger_handoffs_and_reports(
    client: Task3GroqClient, stage: str | None, system: str, expected: int
) -> None:
    assert client.complete(system, json.dumps({"stage": stage})) == '{"valid":true}'
    args = client._client.chat.completions.create.call_args.kwargs
    assert args["max_completion_tokens"] == expected
    assert "reasoning_effort" not in args
    assert client.call_count == 1 and client.last_completed_at == 100
    assert isinstance(
        client, GroqClient
    )  # Existing synthesis text repair remains usable.


def test_supported_reasoning_and_text_repair_configuration(
    client: Task3GroqClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate a supported SDK/model pair without calling the provider.
    monkeypatch.setattr(client, "_supports_reasoning", True)
    client.complete("planner", '{"stage":"single_research"}')
    create = client._client.chat.completions.create
    assert create.call_args.kwargs["reasoning_effort"] == "low"
    client.complete("synthesis", '{"stage":"final_synthesis"}', json_mode=False)
    assert create.call_args.kwargs["reasoning_effort"] == "medium"
    assert create.call_args.kwargs["response_format"] == {"type": "text"}
    assert create.call_args.kwargs["max_completion_tokens"] == 1800


def test_unknown_request_uses_report_budget_and_budget_validation(client) -> None:
    client.complete("unknown", '{"stage":{}}')
    assert (
        client._client.chat.completions.create.call_args.kwargs["max_completion_tokens"]
        == 1800
    )
    with pytest.raises(ValueError):
        TokenBudgets(planner=0)


def test_rate_limit_retry_after_stays_in_transport_and_logs_safely(
    client, caplog
) -> None:
    create = client._client.chat.completions.create
    success = create.return_value
    error = status_error(429, "7")
    error.body = {
        "error": {"code": "rate_limit_exceeded", "message": "private provider body"}
    }
    create.side_effect = [error, success]
    client.complete("planner", '{"stage":"single_research"}')
    client._sleep.assert_called_once_with(7)
    assert create.call_count == 2 and client.call_count == 1
    assert "rate_limit" in caplog.text and "private provider body" not in caplog.text
    create.side_effect = status_error(400)
    with pytest.raises(LLMTransportError):
        client.complete("planner", '{"stage":"single_research"}')
    assert create.call_count == 3  # No retry of the deterministic request error.


def test_pacing_waits_only_remaining_window_and_announces() -> None:
    sleep, announce = Mock(), Mock()
    pacer = DemoPacer(clock=lambda: 125.0, sleep=sleep, announce=announce)
    assert pacer.before_section("the autonomous workflow", 100.0) == 35.0
    sleep.assert_called_once_with(35.0)
    assert "Groq free-tier pacing enabled" in announce.call_args.args[0]
    assert "the autonomous workflow" in announce.call_args.args[0]


def test_pacing_disabled_fresh_or_elapsed_sections_never_sleep() -> None:
    sleep = Mock()
    pacer = DemoPacer(clock=lambda: 200.0, sleep=sleep)
    assert pacer.before_section("first", None) == 0
    assert pacer.before_section("elapsed", 100.0) == 0
    pacer.enabled = False
    assert pacer.before_section("higher quota", 199.0) == 0
    sleep.assert_not_called()
    with pytest.raises(ValueError):
        DemoPacer(window_seconds=float("nan"))


def json_generation_error():
    error = status_error(400)
    error.body = {"error": {"code": "json_validate_failed", "message": "private body"}}
    return error


def sdk_response(value):
    text = value if isinstance(value, str) else json.dumps(value)
    return Mock(choices=[Mock(message=Mock(content=text))])


@pytest.mark.parametrize("role", ["researcher", "analyst"])
def test_planner_provider_json_fallback_keeps_local_action_validation(
    client, executor, role
):
    if role == "researcher":
        stage, schema = "single_research", ResearchReport
        responses = [
            tool("get_price_data", ticker="NVDA"),
            tool("get_news", ticker="NVDA"),
            ready(),
            report(price_id="obs-1", research_id="obs-2"),
        ]
    else:
        stage, schema = "analyst_brief", QuantitativeBrief
        responses = multi_responses(executor)[0][:2]
    create = client._client.chat.completions.create
    create.side_effect = [json_generation_error()] + [
        sdk_response(value) for value in responses
    ]
    result = AgentRuntime(client, executor).run(
        "Research NVDA", role=role, stage=stage, output_schema=schema, as_of=AS_OF
    )
    assert result.output and not result.error
    requests = create.call_args_list
    assert [call.kwargs["response_format"] for call in requests[:2]] == [
        {"type": "json_object"},
        {"type": "text"},
    ]
    assert (
        "Return exactly one valid JSON object"
        in requests[1].kwargs["messages"][0]["content"]
    )
    assert requests[0].kwargs["messages"][1] == requests[1].kwargs["messages"][1]
    assert sum(item.event == "decision" for item in result.trace) == (
        3 if role == "researcher" else 2
    )
    assert client.call_count == len(responses)  # Fallback is not another planning step.


def test_invalid_text_repaired_action_still_rejected_by_existing_schema(
    client, executor
):
    create = client._client.chat.completions.create
    create.side_effect = [
        json_generation_error(),
        sdk_response({"kind": "INVENT", "reason": "Invalid action"}),
    ]
    result = AgentRuntime(client, executor, max_decisions=1).run(
        "Research NVDA",
        role="analyst",
        stage="analyst_brief",
        output_schema=QuantitativeBrief,
        as_of=AS_OF,
    )
    assert result.output is None and result.error
    assert executor.counter == 0 and create.call_count == 2
    assert not any(item.event == "decision" for item in result.trace)
    assert any(
        item.event == "llm_failure" and item.content["category"] == "validation"
        for item in result.trace
    )


def test_provider_fallback_exhaustion_does_not_trigger_legacy_stage_repair(
    client, executor
):
    create = client._client.chat.completions.create
    create.side_effect = json_generation_error()
    result = AgentRuntime(client, executor).run(
        "Research NVDA",
        role="analyst",
        stage="analyst_clarify",
        output_schema=ClarificationResponse,
        as_of=AS_OF,
    )
    assert result.output is None and result.error
    assert create.call_count == 2 and client.call_count == 1
    assert executor.counter == 0


def test_explicit_text_mode_never_gets_another_provider_json_fallback(client):
    create = client._client.chat.completions.create
    create.side_effect = json_generation_error()
    with pytest.raises(LLMTransportError):
        client.complete("planner", '{"stage":"single_research"}', json_mode=False)
    assert create.call_count == 1


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.waits = []

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.waits.append(duration)
        self.now += duration


def inject_pacing(client, *, enabled):
    clock, announce = FakeClock(), Mock()
    client._clock, client._sleep = clock, clock.sleep
    client.request_pacer = DemoPacer(
        enabled=enabled,
        window_seconds=35,
        clock=clock,
        sleep=clock.sleep,
        announce=announce,
    )
    return clock, announce


def test_intra_workflow_pacing_waits_remaining_interval(client):
    clock, announce = inject_pacing(client, enabled=True)
    client.complete("planner", '{"stage":"single_research"}')
    assert clock.waits == []  # No initial delay.
    clock.now += 3.6
    client.complete("critique", '{"stage":"writer_review"}')
    assert clock.waits == [pytest.approx(31.4)]
    assert client.last_completed_at == 135.0 and client.call_count == 2
    announce.assert_called_once()
    assert "intra-workflow pacing" in announce.call_args.args[0]
    assert "31.4s" in announce.call_args.args[0]


def test_client_pacing_defaults_disabled_and_can_be_disabled_for_higher_quotas(client):
    assert not client.request_pacer.enabled
    clock, announce = inject_pacing(client, enabled=False)
    client.complete("planner", '{"stage":"single_research"}')
    client.complete("planner", '{"stage":"single_research"}')
    assert clock.waits == []
    announce.assert_not_called()


def test_section_pacing_satisfies_request_interval_without_double_wait(client):
    clock, announce = inject_pacing(client, enabled=True)
    client.complete("planner", '{"stage":"single_research"}')
    clock.now += 10
    section = DemoPacer(enabled=True, clock=clock, sleep=clock.sleep)
    assert (
        section.before_section("multi-agent workflow", client.last_completed_at) == 50
    )
    client.complete("planner", '{"stage":"analyst_brief"}')
    assert clock.waits == [50]  # The shared completion timestamp avoids another wait.
    announce.assert_not_called()


def test_retry_after_remains_authoritative_with_request_pacing_enabled(client):
    clock, _ = inject_pacing(client, enabled=True)
    create = client._client.chat.completions.create
    success = create.return_value
    create.side_effect = [status_error(429, "7"), success, success]
    client.complete("planner", '{"stage":"single_research"}')
    assert clock.waits == [7]
    assert client.last_completed_at == 107
    client.complete("planner", '{"stage":"single_research"}')
    assert clock.waits == [7, 35]


def test_provider_json_fallback_also_observes_request_pacing(client):
    clock, _ = inject_pacing(client, enabled=True)
    create = client._client.chat.completions.create
    create.side_effect = [json_generation_error(), create.return_value]
    client.complete("planner", '{"stage":"single_research"}')
    assert clock.waits == [35]
    assert client.call_count == 1 and create.call_count == 2
