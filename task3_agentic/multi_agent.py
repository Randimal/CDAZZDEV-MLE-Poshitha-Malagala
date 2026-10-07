"""Mandatory typed analyst -> review -> clarification -> writer LangGraph."""

from datetime import date, datetime, timezone
from typing import Any

from langgraph.graph import END, START, StateGraph

from task3_agentic.memory import PersistentMemory
from task3_agentic.prompts import QUESTION
from task3_agentic.runtime import AgentRuntime
from task3_agentic.schemas import (
    ClarificationRequest,
    ClarificationResponse,
    QuantitativeBrief,
    ResearchReport,
    ResearchRun,
)
from task3_agentic.state import MultiState
from task3_agentic.tracing import event


class TwoAgentResearch:
    def __init__(
        self, runtime: AgentRuntime, memory: PersistentMemory | None = None
    ) -> None:
        self.runtime = runtime
        self.memory = memory or PersistentMemory()

    def run(
        self,
        query: str | None = None,
        *,
        as_of: date | None = None,
        use_cache: bool = True,
    ) -> ResearchRun:
        ticker = self.runtime.executor.tools.ticker
        as_of = as_of or datetime.now(timezone.utc).date()
        query = query or QUESTION.format(ticker=ticker)
        if use_cache:
            cached = self.memory.load(ticker, as_of, "multi")
            if cached:
                return cached
        self.runtime.executor.start_session(as_of, refresh=not use_cache)

        def analyst(state: MultiState) -> dict[str, Any]:
            result = self.runtime.run(
                query,
                role="analyst",
                stage="analyst_brief",
                output_schema=QuantitativeBrief,
                as_of=as_of,
            )
            trace = result.trace
            if result.output:
                trace += [
                    event(
                        "analyst",
                        "structured_handoff",
                        {"to": "writer", "brief": result.output.model_dump()},
                    )
                ]
            return {
                "brief": result.output,
                "analyst_observations": result.observations,
                "trace": trace,
                "error": result.error,
            }

        def review(state: MultiState) -> dict[str, Any]:
            result = self.runtime.run(
                query,
                role="writer",
                stage="writer_review",
                output_schema=ClarificationRequest,
                as_of=as_of,
                context={"brief": state["brief"].model_dump()},
            )
            trace = state["trace"] + result.trace
            if result.output:
                trace += [
                    event("writer", "critique_request", result.output.model_dump())
                ]
            return {
                "critique_request": result.output,
                "writer_observations": result.observations,
                "trace": trace,
                "error": result.error,
            }

        def clarify(state: MultiState) -> dict[str, Any]:
            result = self.runtime.run(
                query,
                role="analyst",
                stage="analyst_clarify",
                output_schema=ClarificationResponse,
                as_of=as_of,
                context={
                    "brief": state["brief"].model_dump(),
                    "request": state["critique_request"].model_dump(),
                },
                observations=state["analyst_observations"],
            )
            trace = state["trace"] + result.trace
            if result.output:
                trace += [
                    event(
                        "analyst", "clarification_response", result.output.model_dump()
                    )
                ]
            return {
                "clarification": result.output,
                "analyst_observations": result.observations,
                "trace": trace,
                "error": result.error,
            }

        def write(state: MultiState) -> dict[str, Any]:
            result = self.runtime.run(
                query,
                role="writer",
                stage="writer_final",
                output_schema=ResearchReport,
                as_of=as_of,
                context={
                    "brief": state["brief"].model_dump(),
                    "clarification": state["clarification"].model_dump(),
                    "request": state["critique_request"].model_dump(),
                },
                observations=state["writer_observations"],
            )
            return {
                "report": result.output,
                "writer_observations": result.observations,
                "trace": state["trace"] + result.trace,
                "error": result.error,
            }

        graph = StateGraph(MultiState)
        for name, node in (
            ("analyst", analyst),
            ("review", review),
            ("clarify", clarify),
            ("write", write),
        ):
            graph.add_node(name, node)
        graph.add_edge(START, "analyst")
        for name, next_name, output_key in (
            ("analyst", "review", "brief"),
            ("review", "clarify", "critique_request"),
            ("clarify", "write", "clarification"),
        ):

            def route(state: MultiState, key: str = output_key) -> str:
                return (
                    "stop" if state.get("error") or not state.get(key) else "continue"
                )

            graph.add_conditional_edges(
                name, route, {"stop": END, "continue": next_name}
            )
        graph.add_edge("write", END)
        result = graph.compile().invoke({"trace": [], "error": None})
        run = ResearchRun(
            ticker=ticker,
            as_of=as_of,
            workflow="multi",
            report=result.get("report"),
            brief=result.get("brief"),
            critique_request=result.get("critique_request"),
            clarification=result.get("clarification"),
            observations=result.get("analyst_observations", [])
            + result.get("writer_observations", []),
            trace=result["trace"],
            error=result.get("error"),
        )
        if run.report:
            run.trace.append(
                event(
                    "memory",
                    "persistent_cache_save",
                    {"workflow": "multi", "as_of": as_of},
                )
            )
            try:
                self.memory.save(run)
            except OSError:
                run.trace.append(
                    event(
                        "memory",
                        "cache_write_failed",
                        {"reason": "storage unavailable"},
                    )
                )
        return run
