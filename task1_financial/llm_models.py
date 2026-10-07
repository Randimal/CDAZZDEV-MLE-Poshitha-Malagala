"""Validated Task 1B outputs; unavailable analysis is never a fake signal."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HeadlineSentiment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    headline: str = Field(min_length=1)
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    brief_reason: str = Field(min_length=1, max_length=500)


class TechnicalRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    signal: Literal["BUY", "HOLD", "SELL"]
    reasoning: str = Field(min_length=1, max_length=900)

    @field_validator("reasoning")
    @classmethod
    def validate_sentences(cls, value: str) -> str:
        """Require 3–5 sentences using punctuation/whitespace boundaries.

        This is a structural heuristic, not a semantic reasoning validator.
        Prompts request simple sentences without abbreviations for clarity.
        """
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Za-z])", value)
        if not 3 <= len(sentences) <= 5 or any(
            not sentence.endswith((".", "!", "?")) for sentence in sentences
        ):
            raise ValueError("Reasoning must contain 3–5 complete sentences")
        return value


class SentimentAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    overall_score: float | None
    positive_count: int
    negative_count: int
    neutral_count: int
    successful_count: int
    failed_count: int
    overall_label: Literal["positive", "negative", "neutral", "unavailable"]


class SentimentBatch(BaseModel):
    results: list[HeadlineSentiment]
    aggregate: SentimentAggregate
