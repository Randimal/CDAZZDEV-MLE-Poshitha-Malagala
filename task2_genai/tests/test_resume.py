"""Checkpoint recovery and threshold regressions; all teacher requests mocked."""

import json
from dataclasses import dataclass
from pathlib import Path

import nbformat
import pytest

from task2_genai.client import TeacherError
from task2_genai.dataset import split_examples
from task2_genai.generation import generate_examples, load_checkpoint
from task2_genai.schemas import GenerationConfig
from task2_genai.tests.conftest import example_from_assignment


class RecordingTeacher:
    model = "mock-teacher-not-a-live-model"

    def __init__(self, *, successful_batches=None, reverse=False, duplicate=None):
        self.requests = []
        self.successful_batches = successful_batches
        self.reverse = reverse
        self.duplicate = duplicate

    def complete(self, system_prompt, user_prompt):
        assignments = json.loads(user_prompt)["assignments"]
        self.requests.append(assignments)
        if (
            self.successful_batches is not None
            and len(self.requests) > self.successful_batches
        ):
            raise TeacherError("rate_limit")
        examples = [example_from_assignment(item) for item in assignments]
        if self.reverse:
            examples.reverse()
        if self.duplicate is not None:
            examples += [self.duplicate.model_dump(), examples[0]]
        return json.dumps({"examples": examples})


@dataclass
class LegacyRun:
    path: Path
    valid: list
    original_bytes: bytes


@pytest.fixture
def legacy_run(tmp_path):
    path = tmp_path / "raw_teacher_examples.jsonl"
    result = generate_examples(
        RecordingTeacher(successful_batches=2),
        GenerationConfig(batch_size=10),
        raw_path=path,
    )
    assert len(result.examples) == 20 and result.failed_batches == 14
    return LegacyRun(path, result.examples, path.read_bytes())


def test_resume_only_missing_ids_preserves_valid_examples_and_audit(legacy_run):
    teacher = RecordingTeacher()
    result = generate_examples(
        teacher, GenerationConfig(), raw_path=legacy_run.path, resume=True
    )
    old = {example.id: example for example in legacy_run.valid}
    requested_ids = [item["id"] for batch in teacher.requests for item in batch]
    assert len(teacher.requests) == 28
    assert all(len(batch) == 5 for batch in teacher.requests)
    assert len(requested_ids) == len(set(requested_ids)) == 140
    assert old.keys().isdisjoint(requested_ids)
    assert (
        set(requested_ids) == {example.id for example in result.examples} - old.keys()
    )
    assert result.resume_summary == {
        "existing_valid": 20,
        "missing_before_resume": 140,
        "newly_generated": 140,
        "final_valid": 160,
    }
    assert all(
        example == old[example.id] for example in result.examples if example.id in old
    )
    assert legacy_run.path.read_bytes().startswith(legacy_run.original_bytes)
    assert len(result.raw_records) == 44
    assert result.failed_batches == 14 and result.missing_count == 0
    assert [record["batch_id"] for record in result.raw_records] == list(range(1, 45))
    assert all(record["response"] is None for record in result.raw_records[2:16])


def test_resume_rejects_duplicate_ids_without_replacing_old_examples(legacy_run):
    first = legacy_run.valid[0]
    result = generate_examples(
        RecordingTeacher(duplicate=first),
        GenerationConfig(),
        raw_path=legacy_run.path,
        resume=True,
    )
    assert (
        len(result.examples) == len({example.id for example in result.examples}) == 160
    )
    assert (
        next(example for example in result.examples if example.id == first.id) == first
    )
    assert len(result.rejected) == 56
    assert {item["reason"] for item in result.rejected} == {"duplicate_id"}
    replay = load_checkpoint(legacy_run.path, GenerationConfig())
    assert replay.examples == result.examples and replay.rejected == result.rejected


def test_checkpoint_duplicate_results_are_rejected_on_replay(legacy_run):
    records = [
        json.loads(line) for line in legacy_run.original_bytes.decode().splitlines()
    ]
    payload = json.loads(records[0]["response"])
    payload["examples"].append(payload["examples"][0])
    records[0]["response"] = json.dumps(payload)
    legacy_run.path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    before = legacy_run.path.read_bytes()
    result = load_checkpoint(legacy_run.path, GenerationConfig())
    assert len(result.examples) == 20
    assert len(result.rejected) == 1 and result.rejected[0]["reason"] == "duplicate_id"
    assert legacy_run.path.read_bytes() == before


def test_resume_merge_is_deterministic_despite_response_order(legacy_run, tmp_path):
    other_path = tmp_path / "other_raw.jsonl"
    other_path.write_bytes(legacy_run.original_bytes)
    first = generate_examples(
        RecordingTeacher(), GenerationConfig(), raw_path=legacy_run.path, resume=True
    )
    second = generate_examples(
        RecordingTeacher(reverse=True),
        GenerationConfig(),
        raw_path=other_path,
        resume=True,
    )
    assert first.examples == second.examples
    assert [example.id for example in first.examples] == sorted(
        example.id for example in first.examples
    )


def test_completed_checkpoint_makes_zero_teacher_calls(legacy_run):
    generate_examples(
        RecordingTeacher(), GenerationConfig(), raw_path=legacy_run.path, resume=True
    )
    before = legacy_run.path.read_bytes()
    teacher = RecordingTeacher()
    result = generate_examples(
        teacher, GenerationConfig(), raw_path=legacy_run.path, resume=True
    )
    assert teacher.requests == []
    assert result.resume_summary == {
        "existing_valid": 160,
        "missing_before_resume": 0,
        "newly_generated": 0,
        "final_valid": 160,
    }
    assert legacy_run.path.read_bytes() == before


def test_incompatible_checkpoint_stops_before_calls_or_writes(legacy_run):
    teacher = RecordingTeacher()
    with pytest.raises(ValueError, match="incompatible"):
        generate_examples(
            teacher, GenerationConfig(seed=43), raw_path=legacy_run.path, resume=True
        )
    assert teacher.requests == []
    assert legacy_run.path.read_bytes() == legacy_run.original_bytes


@pytest.mark.parametrize("count", [20, 119])
def test_no_split_before_hard_minimum(many_examples, count):
    with pytest.raises(ValueError, match="120 clean"):
        split_examples(many_examples[:count])
    with pytest.raises(ValueError, match="120 clean"):
        split_examples(many_examples[:count], minimum_clean=20)


def test_notebook_blocks_split_for_partial_clean_dataset(
    tmp_path, many_examples, capsys
):
    notebook_path = Path(__file__).resolve().parents[1] / "task2_genai.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    source = next(cell.source for cell in notebook.cells if cell.id == "task2-split")
    exec(
        source,
        {
            "DATA_DIR": tmp_path,
            "dataset_ready": False,
            "config": GenerationConfig(),
            "clean_examples": many_examples[:20],
            "json": json,
        },
    )
    assert "minimum clean size not reached" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())
