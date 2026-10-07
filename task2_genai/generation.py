"""Balanced assignments, safe batch parsing, and per-record rejection auditing."""

import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from task2_genai.client import TeacherClient, TeacherError
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
    existing_valid: int = 0
    missing_before_resume: int = 0
    newly_generated: int = 0

    @property
    def resume_summary(self) -> dict[str, int]:
        return {
            "existing_valid": self.existing_valid,
            "missing_before_resume": self.missing_before_resume,
            "newly_generated": self.newly_generated,
            "final_valid": len(self.examples),
        }


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


def _accept_record(
    record: dict[str, Any], result: GenerationResult, seen_ids: set[str]
) -> str | None:
    """One validation path for both checkpoint replay and newly returned batches."""
    if record.get("error_category") or record.get("response") is None:
        result.failed_batches += 1
        return record.get("error_category") or "missing_response"
    try:
        candidates = parse_teacher_batch(record["response"])
    except (ValueError, TypeError):
        result.failed_batches += 1
        return "json_parse"
    expected = {item["id"]: item for item in record["assignments"]}
    result.returned_count += len(candidates)
    for position, candidate in enumerate(candidates):
        reference = f"batch_{record['batch_id']}_item_{position + 1}"
        if isinstance(candidate, dict) and isinstance(candidate.get("id"), str):
            reference = candidate["id"]
        try:
            example = PolicyExample.model_validate(candidate)
            if example.id in seen_ids:
                raise ValueError("duplicate_id")
            assignment = expected.get(example.id)
            if assignment is None:
                raise ValueError("unrequested_id")
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
    return None


def load_checkpoint(
    raw_path: Path,
    config: GenerationConfig,
    *,
    teacher_model: str | None = None,
) -> GenerationResult:
    """Revalidate journal responses without calls, mutation or trusting clean files.

    Batch size can change. Seed, prompt, model (when supplied) and every recorded
    assignment must still match. Corrupt/incompatible journals stop before any API
    call or write; failed batches remain in the audit but contribute no examples.
    """
    result = GenerationResult()
    planned = {item["id"]: item for item in build_assignments(config)}
    prompt_hash = hashlib.sha256(TEACHER_SYSTEM_PROMPT.encode()).hexdigest()
    seen_ids: set[str] = set()
    batch_ids: set[int] = set()
    expected_model = teacher_model
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError
                    if (
                        record["seed"] != config.seed
                        or record["system_prompt_sha256"] != prompt_hash
                    ):
                        raise ValueError
                    model = record["teacher_model"]
                    if not isinstance(model, str) or not model.strip():
                        raise ValueError
                    if expected_model is None:
                        expected_model = model
                    if model != expected_model:
                        raise ValueError
                    batch_id = record["batch_id"]
                    if (
                        type(batch_id) is not int
                        or batch_id < 1
                        or batch_id in batch_ids
                    ):
                        raise ValueError
                    assignments = record["assignments"]
                    if not isinstance(assignments, list) or not assignments:
                        raise ValueError
                    assignment_ids = [item["id"] for item in assignments]
                    if len(set(assignment_ids)) != len(assignment_ids):
                        raise ValueError
                    if any(item != planned.get(item["id"]) for item in assignments):
                        raise ValueError
                    if record.get("response") is not None and not isinstance(
                        record["response"], str
                    ):
                        raise ValueError
                    if record.get("error_category") is not None and not isinstance(
                        record["error_category"], str
                    ):
                        raise ValueError
                except (ValueError, KeyError, TypeError):
                    raise ValueError(
                        f"Checkpoint line {line_number} is corrupt or incompatible; audit unchanged"
                    ) from None
                batch_ids.add(batch_id)
                result.raw_records.append(record)
                _accept_record(record, result, seen_ids)
    result.examples.sort(key=lambda example: example.id)
    result.existing_valid = len(result.examples)
    result.missing_count = result.missing_before_resume = len(planned) - len(seen_ids)
    return result


def _append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    """Append and flush one batch; preserve all earlier audit bytes verbatim."""
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if path.exists() and path.stat().st_size:
        with path.open("rb") as stream:
            stream.seek(-1, os.SEEK_END)
            needs_newline = stream.read(1) != b"\n"
    encoded = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8")
    with path.open("ab") as stream:
        if needs_newline:
            stream.write(b"\n")
        stream.write(encoded + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def generate_examples(
    client: TeacherClient,
    config: GenerationConfig,
    *,
    raw_path: Path | None = None,
    resume: bool = False,
) -> GenerationResult:
    """Request only missing IDs; merge first-valid records in canonical ID order.

    Resume is explicit. Existing successes are revalidated, never regenerated or
    replaced. Parse/schema failures remain rejected and auditable; provider retries
    stay exclusively inside the transport. A future resume can retry missing IDs.
    """
    if raw_path is not None and raw_path.exists() and not resume:
        raise FileExistsError("Raw generation already exists; enable resume=True")
    result = (
        load_checkpoint(raw_path, config, teacher_model=client.model)
        if resume and raw_path is not None
        else GenerationResult()
    )
    plan = build_assignments(config)
    seen_ids = {example.id for example in result.examples}
    pending = [item for item in plan if item["id"] not in seen_ids]
    result.existing_valid = len(seen_ids)
    result.missing_before_resume = len(pending)
    prompt_hash = hashlib.sha256(TEACHER_SYSTEM_PROMPT.encode()).hexdigest()
    next_batch_id = (
        max((record["batch_id"] for record in result.raw_records), default=0) + 1
    )
    for offset in range(0, len(pending), config.batch_size):
        assignments = pending[offset : offset + config.batch_size]
        record: dict[str, Any] = {
            "batch_id": next_batch_id,
            "teacher_model": client.model,
            "seed": config.seed,
            "system_prompt_sha256": prompt_hash,
            "assignments": assignments,
            "response": None,
            "error_category": None,
        }
        next_batch_id += 1
        try:
            record["response"] = client.complete(
                TEACHER_SYSTEM_PROMPT,
                batch_prompt(
                    assignments, [example.scenario for example in result.examples]
                ),
            )
        except TeacherError as error:
            record["error_category"] = error.category
            record["response"] = error.raw_response
        finally:
            category = _accept_record(record, result, seen_ids)
            if category:
                record["error_category"] = category
                logger.warning(
                    "Task 2 batch=%s failed category=%s", record["batch_id"], category
                )
            result.raw_records.append(record)
            if raw_path is not None:
                _append_checkpoint(raw_path, record)
        logger.info(
            "Task 2 batch=%s processed; total accepted=%s",
            record["batch_id"],
            len(result.examples),
        )
    result.examples.sort(key=lambda example: example.id)
    result.missing_count = len(plan) - len(seen_ids)
    result.newly_generated = len(result.examples) - result.existing_valid
    return result
