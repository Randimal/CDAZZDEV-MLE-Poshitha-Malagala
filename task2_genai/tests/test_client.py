from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from task2_genai.client import GroqTeacherClient, TeacherError


def sdk_with(responses):
    create = Mock(side_effect=responses)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def response(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


class ProviderError(Exception):
    def __init__(self, status, code=None, headers=None):
        super().__init__("raw-sensitive-provider-body-api-key")
        self.status_code = status
        self.code = code
        self.response = SimpleNamespace(headers=headers or {})


def test_rate_limit_backoff_and_safe_logging(caplog):
    sdk = sdk_with(
        [
            ProviderError(429, headers={"retry-after": "2.5"}),
            response('{"examples": []}'),
        ]
    )
    waits = []
    client = GroqTeacherClient(sdk=sdk, sleep=waits.append)
    assert client.complete("JSON only", "Generate examples") == '{"examples": []}'
    assert waits == [2.5]
    assert "rate_limit" in caplog.text
    assert "raw-sensitive" not in caplog.text


def test_one_provider_json_fallback_then_failure():
    sdk = sdk_with(
        [
            ProviderError(400, "json_validate_failed"),
            ProviderError(400, "json_validate_failed"),
        ]
    )
    client = GroqTeacherClient(sdk=sdk, sleep=lambda _: None)
    with pytest.raises(TeacherError, match="invalid_request"):
        client.complete("JSON only", "Generate examples")
    assert sdk.chat.completions.create.call_count == 2
    calls = sdk.chat.completions.create.call_args_list
    assert calls[0].kwargs["response_format"] == {"type": "json_object"}
    assert "response_format" not in calls[1].kwargs


def test_nonretryable_provider_error():
    sdk = sdk_with([ProviderError(401)])
    with pytest.raises(TeacherError, match="authentication"):
        GroqTeacherClient(sdk=sdk, sleep=lambda _: None).complete("JSON", "Generate")
    assert sdk.chat.completions.create.call_count == 1


def test_nested_json_rejection_code_recovers():
    error = ProviderError(400)
    error.body = {"error": {"code": "json_validate_failed", "message": "sensitive"}}
    sdk = sdk_with([error, response('{"examples": []}')])
    client = GroqTeacherClient(sdk=sdk, sleep=lambda _: None)
    assert client.complete("JSON only", "Generate") == '{"examples": []}'
    assert sdk.chat.completions.create.call_count == 2


def test_rate_limit_retries_are_bounded():
    sdk = sdk_with([ProviderError(429)] * 3)
    waits = []
    with pytest.raises(TeacherError):
        GroqTeacherClient(sdk=sdk, sleep=waits.append).complete("JSON", "Generate")
    assert sdk.chat.completions.create.call_count == 3
    assert waits == [1.0, 2.0]


def test_injected_pacing_does_not_delay_tests():
    sdk = sdk_with([response("{}"), response("{}")])
    now = [0.0]
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds

    client = GroqTeacherClient(
        sdk=sdk, min_interval_seconds=60, clock=lambda: now[0], sleep=sleep
    )
    client.complete("JSON", "Generate")
    now[0] += 10
    client.complete("JSON", "Generate")
    assert waits == [50]


def test_five_example_token_budget_and_65_second_retry_after():
    sdk = sdk_with(
        [
            ProviderError(429, headers={"retry-after": "70"}),
            response('{"examples": []}'),
            response('{"examples": []}'),
        ]
    )
    now = [0.0]
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds

    client = GroqTeacherClient(
        sdk=sdk, min_interval_seconds=65, clock=lambda: now[0], sleep=sleep
    )
    client.complete("JSON", "Generate five")
    client.complete("JSON", "Generate five")
    assert waits == [70, 65]
    assert all(
        call.kwargs["max_completion_tokens"] == 2500
        for call in sdk.chat.completions.create.call_args_list
    )


def test_long_retry_after_prevents_early_calls_across_batches():
    sdk = sdk_with([ProviderError(429, headers={"retry-after": "300"})])
    client = GroqTeacherClient(sdk=sdk, sleep=lambda _: None, clock=lambda: 0.0)
    for _ in range(2):
        with pytest.raises(TeacherError, match="rate_limit"):
            client.complete("JSON", "Generate five")
    assert sdk.chat.completions.create.call_count == 1


def test_token_truncation_is_explicit_and_raw_response_preserved(caplog):
    value = response('{"examples": [')
    value.choices[0].finish_reason = "length"
    sdk = sdk_with([value])
    with pytest.raises(TeacherError, match="completion_truncated") as caught:
        GroqTeacherClient(sdk=sdk).complete("JSON", "Generate five")
    assert caught.value.raw_response == '{"examples": ['
    assert "completion_truncated" in caplog.text
    assert sdk.chat.completions.create.call_count == 1
