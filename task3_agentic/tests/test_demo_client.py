"""Task 3 budgets/pacing only: mocked SDK and injectable clocks/sleep."""

import json
from unittest.mock import Mock, patch

import pytest

from task1_financial.llm import GroqClient, LLMTransportError
from task1_financial.prompts import SENTIMENT_SYSTEM
from task1_financial.tests.test_transport import status_error
from task3_agentic.demo_client import DemoPacer, Task3GroqClient, TokenBudgets
from task3_agentic.prompts import FOLLOWUP_SYSTEM


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
