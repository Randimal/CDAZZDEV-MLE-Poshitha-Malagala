"""All provider calls are mocked; credential values never enter logs."""

import json
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from task1_financial.llm import (
    GroqClient,
    LLMConfigurationError,
    parse_output,
    validated_completion,
)
from task1_financial.llm_models import HeadlineSentiment, TechnicalRecommendation

REASONING = (
    "Price above both averages supports the trend. "
    "MACD agreement reinforces momentum but RSI suggests caution. "
    "Limited news coverage leaves uncertainty."
)


@pytest.mark.parametrize("sentiment", ["positive", "negative", "neutral"])
def test_valid_sentiment(sentiment: str) -> None:
    value = parse_output(
        json.dumps(
            {
                "headline": "News",
                "sentiment": sentiment,
                "confidence": 0.8,
                "brief_reason": "Evidence.",
            }
        ),
        HeadlineSentiment,
    )
    assert value.sentiment == sentiment and value.confidence == 0.8


@pytest.mark.parametrize(
    "confidence", [-0.1, 1.1, float("nan"), float("inf"), "0.5", None]
)
def test_confidence_validation(confidence: object) -> None:
    with pytest.raises(ValidationError):
        HeadlineSentiment(
            headline="News",
            sentiment="positive",
            confidence=confidence,
            brief_reason="Evidence",
        )


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_confidence_boundaries(confidence: float) -> None:
    assert (
        HeadlineSentiment(
            headline="News",
            sentiment="neutral",
            confidence=confidence,
            brief_reason="Evidence",
        ).confidence
        == confidence
    )


@pytest.mark.parametrize(
    "text",
    [
        "not JSON",
        '{"signal":',
        "[1,2]",
        '{"headline":"News","sentiment":"mixed","confidence":0.5,"brief_reason":"Why"}',
        '{"headline":"News","sentiment":"neutral","confidence":0.5}',
        '{"headline":"News","sentiment":"neutral","confidence":NaN,"brief_reason":"Why"}',
    ],
)
def test_malformed_missing_or_invalid_sentiment(text: str) -> None:
    with pytest.raises((ValueError, ValidationError)):
        parse_output(text, HeadlineSentiment)


def test_fenced_json() -> None:
    value = parse_output(
        '```json\n{"headline":"News","sentiment":"neutral",'
        '"confidence":0.5,"brief_reason":"No direction"}\n```',
        HeadlineSentiment,
    )
    assert value.sentiment == "neutral"


@pytest.mark.parametrize("signal", ["BUY", "HOLD", "SELL"])
def test_valid_recommendation(signal: str) -> None:
    value = parse_output(
        json.dumps({"signal": signal, "reasoning": REASONING}), TechnicalRecommendation
    )
    assert value.signal == signal


@pytest.mark.parametrize(
    "payload",
    [
        {"signal": "BULLISH", "reasoning": REASONING},
        {"signal": "BUY"},
        {"signal": "BUY", "reasoning": "One sentence."},
        {"signal": "SELL", "reasoning": "First. Second. Third. Fourth. Fifth. Sixth."},
        {"signal": "HOLD", "reasoning": REASONING, "invented_price": 1},
    ],
)
def test_invalid_recommendation(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TechnicalRecommendation.model_validate(payload)


def test_bounded_retry_and_safe_logging(caplog: pytest.LogCaptureFixture) -> None:
    client = Mock()
    client.complete.side_effect = [
        RuntimeError("secret must not be logged"),
        "bad JSON",
        json.dumps({"signal": "HOLD", "reasoning": REASONING}),
    ]
    result = validated_completion(
        client, "system", "user", TechnicalRecommendation, attempts=3
    )
    assert result.signal == "HOLD"
    assert client.complete.call_count == 3
    assert "secret must not be logged" not in caplog.text


def test_exhausted_retry_returns_none() -> None:
    client = Mock()
    client.complete.side_effect = RuntimeError("offline")
    assert (
        validated_completion(
            client, "system", "user", TechnicalRecommendation, attempts=2
        )
        is None
    )
    assert client.complete.call_count == 2
    with pytest.raises(ValueError):
        validated_completion(
            client, "system", "user", TechnicalRecommendation, attempts=4
        )


def test_groq_config_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    with pytest.raises(LLMConfigurationError, match="Set GROQ_API_KEY and GROQ_MODEL"):
        GroqClient()


def test_adapter_uses_environment_and_json_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    with patch("task1_financial.llm.Groq") as sdk:
        sdk.return_value.chat.completions.create.return_value.choices[
            0
        ].message.content = "{}"
        client = GroqClient()
        assert client.complete("system", "user") == "{}"
        assert sdk.call_args.kwargs["max_retries"] == 0
        args = sdk.return_value.chat.completions.create.call_args.kwargs
        assert args["model"] == "test-model"
        assert args["response_format"] == {"type": "json_object"}
        assert args["messages"][0]["content"] == "system"
