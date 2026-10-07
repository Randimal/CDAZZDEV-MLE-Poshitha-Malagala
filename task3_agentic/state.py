"""LangGraph state and bounded model context views."""

from typing import Any, TypedDict
from urllib.parse import urlsplit

from task3_agentic.schemas import (
    AgentAction,
    ResearchDecision,
    ToolObservation,
    TraceEvent,
)

MAX_SEARCH_CONTEXT_CHARS = 400


def _source_host(url: str) -> str | None:
    """Keep source attribution compact; tolerate malformed provider URLs."""
    try:
        return urlsplit(url).hostname
    except ValueError:
        return None


class AgentState(TypedDict):
    query: str
    ticker: str
    role: str
    stage: str
    steps: int
    transport_failures: int
    failed_calls: list[str]
    output_failures: int
    decision: AgentAction | ResearchDecision | None
    observations: list[ToolObservation]
    pending: ToolObservation | None
    trace: list[TraceEvent]
    output: Any
    feedback: str | None
    error: str | None
    context: dict[str, Any]


def observation_view(observation: ToolObservation) -> dict[str, Any]:
    """Keep evidence IDs/numeric paths; omit historical rows and bulky source URLs.

    Full provider output stays in the observation/session. This view is also used
    for persisted/follow-up context; it never mutates retrieved observations.
    """
    value = observation.model_dump()
    if observation.tool_name == "get_price_data" and observation.success:
        output = {
            key: value["output"].get(key)
            for key in (
                "ticker",
                "period",
                "price_basis",
                "summary",
                "latest_indicators",
            )
        }
        rows = value["output"].get("rows", [])
        output["row_count"] = value["output"].get("row_count", len(rows))
        output["as_of"] = value["output"].get("as_of") or (
            rows[-1].get("date") if rows else None
        )
        value["output"] = output
    elif observation.tool_name == "get_news" and observation.success:
        value["output"] = {
            "ticker": value["output"].get("ticker"),
            "headlines": [
                {key: item.get(key) for key in ("title", "publisher", "published_at")}
                for item in value["output"].get("headlines", [])
            ],
        }
    elif observation.tool_name == "web_search" and observation.success:
        value["output"] = {
            "query": value["output"].get("query"),
            "results": [
                {
                    "title": item.get("title"),
                    "snippet": item.get("snippet", "")[:MAX_SEARCH_CONTEXT_CHARS],
                    "source": _source_host(item.get("url", "")),
                }
                for item in value["output"].get("results", [])
            ],
        }
    elif observation.tool_name == "llm_sentiment" and observation.success:
        value["output"] = {"aggregate": value["output"]["aggregate"]}
        value["arguments"] = {
            "headline_count": observation.arguments.get(
                "headline_count", len(observation.arguments.get("headlines", []))
            )
        }
    return value


def compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove presentation metadata, preserving fields, refs and validation rules."""
    result = {}
    for key, value in schema.items():
        if key in {"title", "description", "default"}:
            continue
        if key in {"properties", "$defs"}:
            result[key] = {name: compact_schema(item) for name, item in value.items()}
        elif isinstance(value, dict):
            result[key] = compact_schema(value)
        elif isinstance(value, list):
            result[key] = [
                compact_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


class MultiState(TypedDict, total=False):
    brief: Any
    critique_request: Any
    clarification: Any
    report: Any
    analyst_observations: list[ToolObservation]
    writer_observations: list[ToolObservation]
    trace: list[TraceEvent]
    error: str | None
