import json

import pytest

from task2_genai.dataset import (
    chat_record,
    deduplicate_examples,
    examples_frame,
    normalize_scenario,
    save_splits,
    scenario_similarity,
    split_examples,
)
from task2_genai.schemas import PolicyExample


def test_exact_duplicates(valid_example):
    first = PolicyExample.model_validate(valid_example)
    second = PolicyExample.model_validate(
        valid_example | {"id": "duplicate", "scenario": first.scenario.upper() + "!"}
    )
    result = deduplicate_examples([first, second])
    assert result.examples == [first]
    assert result.statistics["exact_duplicates"] == 1
    assert result.removed[0]["duplicate_of"] == first.id


def test_near_duplicates(valid_example):
    scenario = "A small business submitted consistent source documents for a routine transfer and the reviewer verified the funding origin."
    other = "A small business submitted consistent source documents for a routine transfer and the reviewer confirmed the funding origin."
    first = PolicyExample.model_validate(valid_example | {"scenario": scenario})
    second = PolicyExample.model_validate(
        valid_example | {"id": "near_duplicate", "scenario": other}
    )
    result = deduplicate_examples([first, second])
    assert result.statistics["near_duplicates"] == 1
    assert scenario_similarity(scenario, other) > 0.88


def test_duplicate_ids_rejected_by_dedup(valid_example):
    first = PolicyExample.model_validate(valid_example)
    second = PolicyExample.model_validate(
        valid_example
        | {
            "scenario": "An entirely different anonymous customer requests review of a new account."
        }
    )
    assert deduplicate_examples([first, second]).statistics["duplicate_ids"] == 1


def test_deterministic_stratified_split_and_no_leakage(many_examples):
    first = split_examples(many_examples)
    second = split_examples(list(reversed(many_examples)))
    assert first == second
    assert first.counts == {"train": 128, "validation": 16, "test": 16}
    assert first.stratification == "topic"
    sets = [
        set(normalize_scenario(item.scenario) for item in getattr(first, name))
        for name in ("train", "validation", "test")
    ]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(set.union(*sets)) == 160
    assert split_examples(many_examples, seed=7) != first


def test_small_strata_fall_back_to_risk(many_examples):
    split = split_examples(many_examples[:120])
    assert split.counts == {"train": 96, "validation": 12, "test": 12}
    assert split.stratification == "risk"
    for name in ("train", "validation", "test"):
        assert {example.risk_level for example in getattr(split, name)} == {
            "low",
            "medium",
            "high",
        }


def test_split_rejects_duplicate_scenarios_and_ids(many_examples):
    with pytest.raises(ValueError, match="Deduplicate"):
        split_examples(many_examples + [many_examples[0]])
    duplicate = many_examples[0].model_copy(update={"id": "changed_id"})
    with pytest.raises(ValueError, match="Deduplicate"):
        split_examples(many_examples + [duplicate])


def test_chat_format_and_length_metrics(valid_example):
    example = PolicyExample.model_validate(valid_example)
    record = chat_record(example)
    assert [message["role"] for message in record["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert example.policy_excerpt in record["messages"][1]["content"]
    assert example.scenario in record["messages"][1]["content"]
    target = json.loads(record["messages"][2]["content"])
    assert set(target) == {"category", "risk_level", "recommended_action", "rationale"}
    assert target["risk_level"] == example.risk_level
    frame = examples_frame([example])
    assert frame.iloc[0]["scenario_words"] == len(example.scenario.split())
    assert frame.iloc[0]["input_words"] == len(example.policy_excerpt.split()) + len(
        example.scenario.split()
    )


def test_export_and_frozen_test_set(tmp_path, many_examples):
    split = split_examples(many_examples)
    paths = save_splits(tmp_path, split)
    assert len((tmp_path / "test.jsonl").read_text().splitlines()) == 16
    original = (tmp_path / "test.jsonl").read_bytes()
    manifest = json.loads((tmp_path / "split_manifest.json").read_text())
    assert manifest["seed"] == 42
    assert manifest["counts"] == split.counts
    assert set(paths) == {"train", "validation", "test"}
    with pytest.raises(FileExistsError, match="frozen"):
        save_splits(tmp_path, split)
    assert (tmp_path / "test.jsonl").read_bytes() == original
