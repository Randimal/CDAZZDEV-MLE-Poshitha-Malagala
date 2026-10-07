"""Deterministic teacher fixtures, never submission dataset artifacts."""

import pytest

from task2_genai.generation import build_assignments
from task2_genai.policies import POLICY_BY_TOPIC
from task2_genai.schemas import GenerationConfig, PolicyExample


def example_from_assignment(assignment: dict[str, str]) -> dict[str, str]:
    risk = assignment["target_risk"]
    condition, action = POLICY_BY_TOPIC[assignment["topic"]].rules[
        ("low", "medium", "high").index(risk)
    ]
    return {
        key: assignment[key] for key in ("id", "topic", "category", "policy_excerpt")
    } | {
        "scenario": f"For review reference {assignment['id']}, staff establish that {condition}.",
        "risk_level": risk,
        "recommended_action": action,
        "rationale": f"Under the supplied internal policy, {condition} triggers {risk} risk. Staff must {action}.",
    }


@pytest.fixture
def valid_example() -> dict[str, str]:
    return example_from_assignment(build_assignments(GenerationConfig())[0])


@pytest.fixture
def many_examples() -> list[PolicyExample]:
    return [
        PolicyExample.model_validate(example_from_assignment(item))
        for item in build_assignments(GenerationConfig())
    ]
