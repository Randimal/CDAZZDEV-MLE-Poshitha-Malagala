"""Auditable prompts, separated from generation and dataset logic."""

import json
from pathlib import Path
from typing import Any

TEACHER_SYSTEM_PROMPT = (
    Path(__file__).with_name("teacher_system_prompt.txt").read_text(encoding="utf-8")
)
TRIAGE_SYSTEM_PROMPT = (
    "You are a policy-grounded financial compliance triage assistant. "
    "Use only the supplied internal policy and scenario, not external law. "
    "Return JSON with category, risk_level (low, medium or high), "
    "recommended_action and rationale. Ground the rationale in that policy. "
    "This is internal policy triage, not legal advice."
)


def batch_prompt(assignments: list[dict[str, Any]], avoidance: list[str]) -> str:
    """Give the teacher controlled coverage hints without supplying target answers."""
    return json.dumps(
        {"assignments": assignments, "avoid_similar_scenarios": avoidance[-5:]},
        ensure_ascii=False,
        allow_nan=False,
    )
