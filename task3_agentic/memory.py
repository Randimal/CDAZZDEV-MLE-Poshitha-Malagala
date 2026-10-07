"""Session follow-up answers and versioned atomic ticker/date JSON caching."""

import uuid
from datetime import date
from pathlib import Path

from pydantic import Field

from task1_financial.json_utils import json_payload
from task1_financial.llm import CompletionClient, validated_completion
from task3_agentic.prompts import FOLLOWUP_SYSTEM
from task3_agentic.schemas import FollowupAnswer, Model, ResearchRun
from task3_agentic.state import observation_view
from task3_agentic.tools import normalize_ticker
from task3_agentic.tracing import event, redact

DEFAULT_MEMORY_DIR = Path(__file__).resolve().parent / "memory"
CACHE_VERSION = 1


class CacheFile(Model):
    version: int = CACHE_VERSION
    ticker: str
    as_of: date
    workflows: dict[str, ResearchRun] = Field(default_factory=dict)


class PersistentMemory:
    def __init__(self, directory: Path = DEFAULT_MEMORY_DIR) -> None:
        self.directory = Path(directory)

    def path(self, ticker: str, as_of: date) -> Path:
        return self.directory / f"{normalize_ticker(ticker)}_{as_of.isoformat()}.json"

    def _read(self, ticker: str, as_of: date) -> CacheFile | None:
        try:
            entry = CacheFile.model_validate_json(
                self.path(ticker, as_of).read_text(encoding="utf-8")
            )
            if (
                entry.version != CACHE_VERSION
                or entry.ticker != ticker
                or entry.as_of != as_of
            ):
                return None
            return entry
        except (OSError, ValueError):
            return None

    def load(self, ticker: str, as_of: date, workflow: str) -> ResearchRun | None:
        ticker = normalize_ticker(ticker)
        entry = self._read(ticker, as_of)
        run = entry.workflows.get(workflow) if entry else None
        if (
            run is None
            or run.report is None
            or run.error
            or run.ticker != ticker
            or run.as_of != as_of
            or run.workflow != workflow
        ):
            return None
        run.cached = True
        run.trace.append(
            event(
                "memory",
                "persistent_cache_hit",
                {"ticker": ticker, "as_of": as_of, "workflow": workflow},
            )
        )
        return run

    def save(self, run: ResearchRun) -> None:
        if run.report is None or run.error:
            raise ValueError("Only completed research may be cached")
        self.directory.mkdir(parents=True, exist_ok=True)
        entry = self._read(run.ticker, run.as_of) or CacheFile(
            ticker=run.ticker, as_of=run.as_of
        )
        value = run.model_dump(mode="json")
        value["observations"] = [observation_view(item) for item in run.observations]
        entry.workflows[run.workflow] = ResearchRun.model_validate(redact(value))
        target = self.path(run.ticker, run.as_of)
        temporary = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(entry.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def answer_followup(
    run: ResearchRun, question: str, client: CompletionClient
) -> FollowupAnswer | None:
    """No tool executor is reachable here: follow-ups use retrieved state only."""
    payload = redact(
        {
            "question": question,
            "final_report": run.report.model_dump(mode="json") if run.report else None,
            "analyst_brief": run.brief.model_dump() if run.brief else None,
            "analyst_clarification": run.clarification.model_dump()
            if run.clarification
            else None,
            "observations": [observation_view(item) for item in run.observations],
        }
    )
    run.trace.append(event("memory", "followup_question", payload))
    answer = validated_completion(
        client, FOLLOWUP_SYSTEM, json_payload(payload), FollowupAnswer, attempts=2
    )
    allowed = {item.evidence_id for item in run.observations if item.success}
    allowed.update(
        name
        for name, value in (
            ("final_report", run.report),
            ("analyst_brief", run.brief),
            ("analyst_clarification", run.clarification),
        )
        if value is not None
    )
    if answer and not set(answer.evidence_ids).issubset(allowed):
        answer = None
    run.trace.append(
        event(
            "memory",
            "followup_answer",
            answer.model_dump() if answer else {"unavailable": True},
        )
    )
    return answer
