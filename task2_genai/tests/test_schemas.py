import pytest
from pydantic import ValidationError

from task2_genai.schemas import GenerationConfig, PolicyExample


def test_valid_example(valid_example):
    example = PolicyExample.model_validate(valid_example)
    assert example.risk_level in ("low", "medium", "high")
    assert example.policy_excerpt == valid_example["policy_excerpt"]


def test_invalid_risk_rejected(valid_example):
    with pytest.raises(ValidationError):
        PolicyExample.model_validate(valid_example | {"risk_level": "critical"})


@pytest.mark.parametrize(
    "field",
    [
        "id",
        "topic",
        "category",
        "policy_excerpt",
        "scenario",
        "recommended_action",
        "rationale",
    ],
)
def test_empty_field_rejected(valid_example, field):
    with pytest.raises(ValidationError):
        PolicyExample.model_validate(valid_example | {field: "   "})


def test_category_topic_mismatch_rejected(valid_example):
    with pytest.raises(ValidationError, match="topic/category"):
        PolicyExample.model_validate(valid_example | {"category": "Other"})


def test_length_and_extra_field_rejected(valid_example):
    with pytest.raises(ValidationError):
        PolicyExample.model_validate(valid_example | {"scenario": "x" * 2001})
    with pytest.raises(ValidationError):
        PolicyExample.model_validate(valid_example | {"external_law": "unrequested"})


def test_configuration_bounds():
    with pytest.raises(ValidationError):
        GenerationConfig(batch_size=1)
    with pytest.raises(ValidationError):
        GenerationConfig(minimum_clean=200)
