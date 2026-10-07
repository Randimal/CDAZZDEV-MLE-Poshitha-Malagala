import json
from collections import Counter

import pytest

from task2_genai.client import TeacherError
from task2_genai.generation import (
    build_assignments,
    generate_examples,
    parse_teacher_batch,
)
from task2_genai.policies import POLICIES
from task2_genai.schemas import GenerationConfig
from task2_genai.tests.conftest import example_from_assignment


class MockTeacher:
    model = "mock-teacher-not-a-live-model"

    def __init__(self, fail_first=False, invalid_item=False):
        self.calls = 0
        self.fail_first = fail_first
        self.invalid_item = invalid_item

    def complete(self, system_prompt, user_prompt):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise TeacherError("rate_limit")
        examples = [
            example_from_assignment(item)
            for item in json.loads(user_prompt)["assignments"]
        ]
        if self.invalid_item and self.calls == 1:
            examples[0]["risk_level"] = "invalid"
        return json.dumps({"examples": examples})


def test_balanced_assignment_plan():
    config = GenerationConfig()
    first = build_assignments(config)
    assert first == build_assignments(config)
    assert len({item["id"] for item in first}) == 160
    counts = Counter(item["topic"] for item in first)
    assert len(counts) == len(POLICIES) == 14
    assert max(counts.values()) - min(counts.values()) == 1
    risks = Counter(item["target_risk"] for item in first)
    assert max(risks.values()) - min(risks.values()) <= 1
    assert len({item["scenario_word_range"] for item in first}) == 3


@pytest.mark.parametrize("fenced", [False, True])
def test_teacher_batch_parsing(valid_example, fenced):
    response = json.dumps({"examples": [valid_example]})
    if fenced:
        response = "```json\n" + response + "\n```"
    assert parse_teacher_batch(response) == [valid_example]


@pytest.mark.parametrize(
    "response",
    ["not json", '{"examples": NaN}', '{"examples": {}}', "[{}]", "```json\n{}"],
)
def test_malformed_teacher_batch_rejected(response):
    with pytest.raises(ValueError):
        parse_teacher_batch(response)


def test_batched_generation_not_per_example():
    teacher = MockTeacher()
    result = generate_examples(teacher, GenerationConfig())
    assert teacher.calls == 16
    assert len(result.examples) == 160
    assert not result.rejected
    assert result.failed_batches == 0
    assert len(result.raw_records) == 16
    assert all(
        record["response"] and record["system_prompt_sha256"]
        for record in result.raw_records
    )


def test_failed_batch_does_not_fabricate_examples():
    teacher = MockTeacher(fail_first=True)
    result = generate_examples(teacher, GenerationConfig())
    assert len(result.examples) == 150
    assert result.failed_batches == 1
    assert result.missing_count == 10
    assert result.raw_records[0]["response"] is None
    missing_ids = {item["id"] for item in result.raw_records[0]["assignments"]}
    assert missing_ids.isdisjoint(example.id for example in result.examples)


def test_raw_checkpoint_and_overwrite_protection(tmp_path):
    raw_path = tmp_path / "raw_teacher_examples.jsonl"
    result = generate_examples(
        MockTeacher(fail_first=True), GenerationConfig(), raw_path=raw_path
    )
    records = [json.loads(line) for line in raw_path.read_text().splitlines()]
    assert records == result.raw_records
    assert records[0]["response"] is None
    assert records[0]["error_category"] == "rate_limit"
    assert result.returned_count == 150
    with pytest.raises(FileExistsError):
        generate_examples(MockTeacher(), GenerationConfig(), raw_path=raw_path)


def test_all_failed_batches_produce_no_examples():
    class FailedTeacher(MockTeacher):
        def complete(self, system_prompt, user_prompt):
            raise TeacherError("connection")

    result = generate_examples(FailedTeacher(), GenerationConfig())
    assert result.examples == []
    assert result.failed_batches == 16
    assert result.missing_count == 160


def test_individual_invalid_record_rejected_not_repaired():
    result = generate_examples(MockTeacher(invalid_item=True), GenerationConfig())
    assert len(result.examples) == 159
    assert len(result.rejected) == 1
    assert result.rejected[0]["reason"] == "schema_validation"


def test_mutated_policy_and_duplicate_id_rejected():
    class MutatingTeacher(MockTeacher):
        def complete(self, system_prompt, user_prompt):
            payload = json.loads(super().complete(system_prompt, user_prompt))
            payload["examples"][0]["policy_excerpt"] += " Invented requirement."
            payload["examples"].append(payload["examples"][1])
            return json.dumps(payload)

    result = generate_examples(MutatingTeacher(), GenerationConfig())
    reasons = Counter(item["reason"] for item in result.rejected)
    assert reasons == {"assignment_mismatch": 16, "duplicate_id": 16}
