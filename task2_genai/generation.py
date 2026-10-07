"""Balanced assignments, safe batch parsing, and per-record rejection auditing."""

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from task2_genai.client import TeacherClient, TeacherError
from task2_genai.dataset import write_jsonl
from task2_genai.policies import POLICIES
from task2_genai.prompts import TEACHER_SYSTEM_PROMPT, batch_prompt
from task2_genai.schemas import GenerationConfig, PolicyExample

logger = logging.getLogger(__name__)
CUSTOMER_TYPES = (
    "retail customer",
    "small business",
    "corporate treasury",
    "nonprofit",
    "sole trader",
)
AMOUNT_BANDS = (
    "small routine payment",
    "moderate transfer",
    "large transfer",
    "no transaction yet",
)
WORD_RANGES = ("25-40", "45-65", "70-95")
AMBIGUITY = ("clear evidence", "borderline evidence", "incomplete evidence")


@dataclass
class GenerationResult:
    """Raw batch envelopes preserve provenance, including unsuccessful batches."""

    raw_records: list[dict[str, Any]] = field(default_factory=list)
    examples: list[PolicyExample] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)
    failed_batches: int = 0
    missing_count: int = 0
    returned_count: int = 0


def build_assignments(config: GenerationConfig) -> list[dict[str, str]]:
    """Rotate topic/risk, then shuffle: topic counts differ by at most one."""
    rng = random.Random(config.seed)
    assignments = []
    for index in range(config.candidate_count):
        policy = POLICIES[index % len(POLICIES)]
        cycle = index // len(POLICIES)
        risk = ("low", "medium", "high")[(cycle + index % len(POLICIES)) % 3]
        assignments.append(
            {
                "id": f"triage_{index + 1:04d}",
                "topic": policy.topic,
                "category": policy.category,
                "policy_excerpt": policy.excerpt(cycle % 2),
                "target_risk": risk,
                "customer_type": rng.choice(CUSTOMER_TYPES),
                "transaction_size": rng.choice(AMOUNT_BANDS),
                "scenario_word_range": WORD_RANGES[cycle % 3],
                "ambiguity": "clear evidence"
                if risk == "low"
                else rng.choice(AMBIGUITY),
            }
        )
    rng.shuffle(assignments)
    return assignments


def parse_teacher_batch(response: str) -> list[Any]:
    """Parse JSON or one complete JSON fence; no eval or substring recovery."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if (
            len(lines) < 3
            or lines[0].lower() not in ("```", "```json")
            or lines[-1] != "```"
        ):
            raise ValueError("Invalid JSON fence")
        text = "\n".join(lines[1:-1])

    def reject_constant(value: str) -> None:
        raise ValueError("Non-finite JSON number")

    payload = json.loads(text, parse_constant=reject_constant)
    if not isinstance(payload, dict) or set(payload) != {"examples"}:
        raise ValueError("Expected an object containing only examples")
    if not isinstance(payload["examples"], list):
        raise ValueError("examples must be an array")
    return payload["examples"]


def generate_examples(
    client: TeacherClient,
    config: GenerationConfig,
    *,
    raw_path: Path | None = None,
) -> GenerationResult:
    """Keep valid records from partial batches; never invent missing candidates.

    Parse/schema errors are rejected rather than regenerated or silently repaired.
    Retryable provider failures are handled only by the transport.
    """
    if raw_path is not None and raw_path.exists():
        raise FileExistsError("Raw generation already exists; use a new run directory")
    result = GenerationResult()
    plan = build_assignments(config)
    seen_ids: set[str] = set()
    prompt_hash = hashlib.sha256(TEACHER_SYSTEM_PROMPT.encode()).hexdigest()
    for offset in range(0, len(plan), config.batch_size):
        assignments = plan[offset : offset + config.batch_size]
        expected = {item["id"]: item for item in assignments}
        record: dict[str, Any] = {
            "batch_id": offset // config.batch_size + 1,
            "teacher_model": client.model,
            "seed": config.seed,
            "system_prompt_sha256": prompt_hash,
            "assignments": assignments,
            "response": None,
            "error_category": None,
        }
        result.raw_records.append(record)
        try:
            record["response"] = client.complete(
                TEACHER_SYSTEM_PROMPT,
                batch_prompt(
                    assignments, [example.scenario for example in result.examples]
                ),
            )
            candidates = parse_teacher_batch(record["response"])
        except (TeacherError, ValueError, TypeError) as error:
            category = (
                error.category if isinstance(error, TeacherError) else "json_parse"
            )
            record["error_category"] = category
            result.failed_batches += 1
            result.missing_count += len(expected)
            logger.warning(
                "Task 2 batch=%s failed category=%s", record["batch_id"], category
            )
            continue
        finally:
            if raw_path is not None:
                write_jsonl(raw_path, result.raw_records)
        returned_ids: set[str] = set()
        result.returned_count += len(candidates)
        for position, candidate in enumerate(candidates):
            reference = f"batch_{record['batch_id']}_item_{position + 1}"
            if isinstance(candidate, dict) and isinstance(candidate.get("id"), str):
                reference = candidate["id"]
                returned_ids.add(reference)
            try:
                example = PolicyExample.model_validate(candidate)
                assignment = expected.get(example.id)
                if assignment is None:
                    raise ValueError("unrequested_id")
                if example.id in seen_ids:
                    raise ValueError("duplicate_id")
                if any(
                    getattr(example, key) != assignment[key]
                    for key in ("topic", "category", "policy_excerpt")
                ):
                    raise ValueError("assignment_mismatch")
                if example.risk_level != assignment["target_risk"]:
                    raise ValueError("assigned_risk_mismatch")
            except ValidationError:
                result.rejected.append({"id": reference, "reason": "schema_validation"})
                continue
            except ValueError as error:
                result.rejected.append({"id": reference, "reason": str(error)})
                continue
            seen_ids.add(example.id)
            result.examples.append(example)
        result.missing_count += len(set(expected) - returned_ids)
        logger.info(
            "Task 2 batch=%s processed; total accepted=%s",
            record["batch_id"],
            len(result.examples),
        )
    return result
