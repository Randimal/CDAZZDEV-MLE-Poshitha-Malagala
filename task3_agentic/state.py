"""LangGraph state and bounded model context views."""

from typing import Any, TypedDict

from task3_agentic.schemas import AgentAction, ToolObservation, TraceEvent


class AgentState(TypedDict):
    query: str
    ticker: str
    role: str
    stage: str
    steps: int
    decision: AgentAction | None
    observations: list[ToolObservation]
    pending: ToolObservation | None
    trace: list[TraceEvent]
    output: Any
    feedback: str | None
    error: str | None
    context: dict[str, Any]


def observation_view(observation: ToolObservation) -> dict[str, Any]:
    """Retain full price history in session; send only recent rows to the LLM."""
    value = observation.model_dump()
    if observation.tool_name == "get_price_data" and observation.success:
        output = dict(value["output"])
        output["row_count"] = output.get("row_count", len(output.get("rows", [])))
        output["rows"] = output.get("rows", [])[-5:]
        value["output"] = output
    return value


class MultiState(TypedDict, total=False):
    brief: Any
    critique_request: Any
    clarification: Any
    report: Any
    analyst_observations: list[ToolObservation]
    writer_observations: list[ToolObservation]
    trace: list[TraceEvent]
    error: str | None
