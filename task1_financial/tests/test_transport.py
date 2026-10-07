"""SDK transport backoff is bounded, safe and independent of model validation."""

from unittest.mock import Mock, patch

import httpx
import pytest
from groq import APIConnectionError, APIStatusError, APITimeoutError

from task1_financial.llm import GroqClient, LLMTransportError, validated_completion
from task1_financial.llm_models import TechnicalRecommendation


def status_error(status: int, retry_after: str | None = None) -> APIStatusError:
    response = httpx.Response(
        status,
        headers={"retry-after": retry_after} if retry_after else {},
        request=httpx.Request("POST", "https://api.groq.com"),
    )
    return APIStatusError(
        "private response must not be logged", response=response, body=None
    )


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    with patch("task1_financial.llm.Groq"):
        client = GroqClient()
    return client


@pytest.mark.parametrize("kind", ["rate_limit", "timeout", "connection"])
def test_retryable_errors_backoff_inside_transport(adapter, kind, caplog) -> None:
    request = httpx.Request("POST", "https://api.groq.com")
    error = {
        "rate_limit": status_error(429),
        "timeout": APITimeoutError(request=request),
        "connection": APIConnectionError(request=request),
    }[kind]
    success = Mock()
    success.choices = [Mock(message=Mock(content="{}"))]
    adapter._client.chat.completions.create.side_effect = [error, error, error, success]
    with patch("task1_financial.llm.time.sleep") as sleep:
        assert adapter.complete("system", "user") == "{}"
    assert [call.args[0] for call in sleep.call_args_list] == [1, 2, 4]
    assert f"category={kind}" in caplog.text
    assert "private response" not in caplog.text


def test_retry_after_respected(adapter) -> None:
    success = Mock()
    success.choices = [Mock(message=Mock(content="{}"))]
    adapter._client.chat.completions.create.side_effect = [
        status_error(429, "7"),
        success,
    ]
    with patch("task1_financial.llm.time.sleep") as sleep:
        adapter.complete("system", "user")
    sleep.assert_called_once_with(7)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_deterministic_error_does_not_retry(adapter, status) -> None:
    adapter._client.chat.completions.create.side_effect = status_error(status)
    with patch("task1_financial.llm.time.sleep") as sleep:
        with pytest.raises(LLMTransportError, match="invalid_request"):
            adapter.complete("system", "user")
    assert adapter._client.chat.completions.create.call_count == 1
    sleep.assert_not_called()


def test_exhaustion_not_retried_by_validation_layer(adapter) -> None:
    adapter._client.chat.completions.create.side_effect = status_error(429)
    with patch("task1_financial.llm.time.sleep"):
        assert (
            validated_completion(
                adapter, "system", "user", TechnicalRecommendation, attempts=3
            )
            is None
        )
    assert adapter._client.chat.completions.create.call_count == 4


def test_long_retry_after_stops_instead_of_retrying_early(adapter) -> None:
    adapter._client.chat.completions.create.side_effect = status_error(429, "3600")
    with patch("task1_financial.llm.time.sleep") as sleep:
        with pytest.raises(LLMTransportError, match="rate_limit"):
            adapter.complete("system", "user")
    sleep.assert_not_called()
    assert adapter._client.chat.completions.create.call_count == 1
