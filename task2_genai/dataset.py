"""Lightweight deduplication, diversity summaries, splits and chat JSONL export."""

import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from task2_genai.prompts import TRIAGE_SYSTEM_PROMPT
from task2_genai.schemas import PolicyExample

NEAR_DUPLICATE_THRESHOLD = 0.88
SPLIT_RATIOS = (0.8, 0.1, 0.1)
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_SEED = 42


def normalize_scenario(text: str) -> str:
    """Case/Unicode/punctuation-insensitive normalization for exact comparisons."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text))


def scenario_similarity(left: str, right: str) -> float:
    """Token sequence overlap (0-1); lexical heuristic, not semantic similarity."""
    return SequenceMatcher(
        None,
        normalize_scenario(left).split(),
        normalize_scenario(right).split(),
        autojunk=False,
    ).ratio()


@dataclass
class DeduplicationResult:
    examples: list[PolicyExample]
    removed: list[dict[str, Any]]
    comparisons: int
    maximum_similarity: float

    @property
    def statistics(self) -> dict[str, Any]:
        counts = Counter(record["reason"] for record in self.removed)
        return {
            "kept": len(self.examples),
            "removed": len(self.removed),
            "exact_duplicates": counts["exact_duplicate"],
            "near_duplicates": counts["near_duplicate"],
            "duplicate_ids": counts["duplicate_id"],
            "comparisons": self.comparisons,
            "maximum_similarity": self.maximum_similarity,
            "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        }


def deduplicate_examples(examples: Sequence[PolicyExample]) -> DeduplicationResult:
    """Stable first-record retention, globally before splitting; no target repair."""
    kept: list[PolicyExample] = []
    removed: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    normalized: dict[str, str] = {}
    comparisons = 0
    maximum = 0.0
    for example in examples:
        text = normalize_scenario(example.scenario)
        reason = None
        duplicate_of = None
        similarity = None
        if example.id in seen_ids:
            reason = "duplicate_id"
        elif text in normalized:
            reason, duplicate_of, similarity = "exact_duplicate", normalized[text], 1.0
        else:
            for prior in kept:
                comparisons += 1
                score = scenario_similarity(example.scenario, prior.scenario)
                maximum = max(maximum, score)
                if score >= NEAR_DUPLICATE_THRESHOLD:
                    reason, duplicate_of, similarity = "near_duplicate", prior.id, score
                    break
        seen_ids.add(example.id)
        if reason:
            removed.append(
                {
                    "id": example.id,
                    "reason": reason,
                    "duplicate_of": duplicate_of,
                    "similarity": similarity,
                }
            )
        else:
            normalized[text] = example.id
            kept.append(example)
    return DeduplicationResult(kept, removed, comparisons, maximum)


def examples_frame(examples: Sequence[PolicyExample]) -> pd.DataFrame:
    """Word lengths are transparent proxies; student token lengths come in 2B."""
    columns = list(PolicyExample.model_fields)
    frame = pd.DataFrame(
        [example.model_dump() for example in examples], columns=columns
    )
    frame["scenario_words"] = frame["scenario"].str.split().str.len()
    frame["input_words"] = (
        frame["policy_excerpt"].str.split().str.len() + frame["scenario_words"]
    )
    return frame


@dataclass
class DatasetSplit:
    train: list[PolicyExample]
    validation: list[PolicyExample]
    test: list[PolicyExample]
    stratification: str
    seed: int

    @property
    def counts(self) -> dict[str, int]:
        return {name: len(getattr(self, name)) for name in SPLIT_NAMES}


def _sizes(total: int) -> list[int]:
    targets = [total * ratio for ratio in SPLIT_RATIOS]
    sizes = [math.floor(target) for target in targets]
    order = sorted(
        range(3), key=lambda index: targets[index] - sizes[index], reverse=True
    )
    for index in order[: total - sum(sizes)]:
        sizes[index] += 1
    return sizes


def split_examples(
    examples: Sequence[PolicyExample], seed: int = SPLIT_SEED
) -> DatasetSplit:
    """Exact rounded 80/10/10 sizes, with feasible topic/risk stratification.

    Use topic+risk if each holdout can represent every stratum, otherwise topic,
    otherwise risk, otherwise seeded shuffling. Holdout quotas use proportional
    largest remainders; small datasets cannot represent all 42 topic/risk cells.
    Reject duplicate IDs/scenarios instead of allowing cross-split leakage.
    """
    if len(examples) < 10:
        raise ValueError(
            "At least ten examples are needed for nonempty 80/10/10 splits"
        )
    ids = [example.id for example in examples]
    scenarios = [normalize_scenario(example.scenario) for example in examples]
    if len(set(ids)) != len(ids) or len(set(scenarios)) != len(scenarios):
        raise ValueError("Deduplicate IDs and scenarios before splitting")
    sizes = _sizes(len(examples))
    rng = random.Random(seed)
    stratification = "none"
    keys = ["all"] * len(examples)
    for name, proposed in (
        (
            "topic+risk",
            [f"{example.topic}|{example.risk_level}" for example in examples],
        ),
        ("topic", [example.topic for example in examples]),
        ("risk", [example.risk_level for example in examples]),
    ):
        counts = Counter(proposed)
        if len(counts) <= min(sizes[1:]) and min(counts.values()) >= 3:
            stratification, keys = name, proposed
            break
    groups: dict[str, list[PolicyExample]] = defaultdict(list)
    for key, example in zip(keys, examples):
        groups[key].append(example)
    group_names = sorted(groups)
    for key in group_names:
        groups[key].sort(key=lambda example: example.id)
        rng.shuffle(groups[key])
    quotas: dict[str, list[int]] = {key: [0, 0] for key in group_names}
    for holdout, target in enumerate(sizes[1:]):
        ideals = {key: len(groups[key]) * target / len(examples) for key in group_names}
        for key in group_names:
            quotas[key][holdout] = math.floor(ideals[key])
        remainder = target - sum(value[holdout] for value in quotas.values())
        tie_order = group_names.copy()
        rng.shuffle(tie_order)
        order = sorted(
            tie_order, key=lambda key: ideals[key] - quotas[key][holdout], reverse=True
        )
        for key in order[:remainder]:
            quotas[key][holdout] += 1
    parts: dict[str, list[PolicyExample]] = {name: [] for name in SPLIT_NAMES}
    for key in group_names:
        validation_count, test_count = quotas[key]
        group = groups[key]
        parts["validation"].extend(group[:validation_count])
        parts["test"].extend(group[validation_count : validation_count + test_count])
        parts["train"].extend(group[validation_count + test_count :])
    for part in parts.values():
        rng.shuffle(part)
    split = DatasetSplit(**parts, stratification=stratification, seed=seed)
    assert list(split.counts.values()) == sizes
    return split


def chat_record(example: PolicyExample) -> dict[str, Any]:
    """Messages consumed by Qwen's tokenizer.apply_chat_template in Task 2B.

    Do not bake ChatML tokens into text: the model tokenizer owns its template.
    ID is provenance metadata, not part of the model's user input.
    """
    target = {
        key: getattr(example, key)
        for key in ("category", "risk_level", "recommended_action", "rationale")
    }
    return {
        "id": example.id,
        "messages": [
            {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Internal policy:\n{example.policy_excerpt}\n\nScenario:\n{example.scenario}",
            },
            {
                "role": "assistant",
                "content": json.dumps(target, ensure_ascii=False, allow_nan=False),
            },
        ],
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """UTF-8 JSONL; never emit NaN or use environment-specific paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def save_splits(directory: Path, split: DatasetSplit) -> dict[str, str]:
    """Protect every existing split, particularly the future evaluation test set."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / f"{name}.jsonl" for name in SPLIT_NAMES}
    if (
        any(path.exists() for path in paths.values())
        or (directory / "split_manifest.json").exists()
    ):
        raise FileExistsError(
            "Split artifacts already exist; preserve the frozen test set"
        )
    for name, path in paths.items():
        write_jsonl(path, (chat_record(example) for example in getattr(split, name)))
    manifest = {
        "seed": split.seed,
        "ratios": dict(zip(SPLIT_NAMES, SPLIT_RATIOS)),
        "counts": split.counts,
        "stratification": split.stratification,
        "ids": {
            name: [example.id for example in getattr(split, name)]
            for name in SPLIT_NAMES
        },
        "sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()
        },
    }
    (directory / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {name: str(path) for name, path in paths.items()}
