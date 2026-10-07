"""Shared LangGraph decide -> tool -> observe -> replan execution loop."""

import math
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from task1_financial.json_utils import json_payload
from task1_financial.llm import (
    CompletionClient,
    LLMTransportError,
    error_category,
    parse_output,
)
from task3_agentic.prompts import AGENT_SYSTEM, TOOL_DESCRIPTIONS
from task3_agentic.schemas import (
    METRIC_PATHS,
    AgentAction,
    ClarificationResponse,
    QuantitativeBrief,
    QuantMetric,
    ResearchReport,
    Role,
    ToolObservation,
    TraceEvent,
)
from task3_agentic.state import AgentState, observation_view
from task3_agentic.tools import (
    ARGUMENT_SCHEMAS,
    ROLE_TOOLS,
    ToolExecutor,
    tool_call_key,
)
from task3_agentic.tracing import event, redact

DEFAULT_DECISION_BUDGET = 12
MAX_UNCLASSIFIED_TRANSPORT_FAILURES = 2


def evidence_coverage(state: AgentState) -> dict[str, bool]:
    """Require price/volatility AND news/search, accepting validated A handoff."""
    quantitative = any(
        item.success and item.tool_name in {"get_price_data", "calculate_volatility"}
        for item in state["observations"]
    )
    brief = state["context"].get("brief", {})
    quantitative = quantitative or any(
        metric.get("value") is not None and metric.get("name") != "sentiment_score"
        for metric in brief.get("metrics", [])
    )
    return {
        "quantitative": quantitative,
        "qualitative": any(
            item.success and item.tool_name in {"get_news", "web_search"}
            for item in state["observations"]
        ),
    }


@dataclass
class StageResult:
    output: BaseModel | None
    observations: list[ToolObservation]
    trace: list[TraceEvent]
    error: str | None


def validate_metric(metric: QuantMetric, observations: list[ToolObservation]) -> None:
    """Validate numeric handoff values against successful tool output paths."""
    sources = {item.evidence_id: item for item in observations if item.success}
    if metric.evidence_id not in sources or metric.path != METRIC_PATHS[metric.name]:
        raise ValueError("Metric must cite a canonical successful observation")
    actual = sources[metric.evidence_id].output
    for key in metric.path.split("."):
        actual = actual[key]
    if actual is None:
        if metric.value is not None:
            raise ValueError("Missing metric cannot be assigned a numeric value")
    elif (
        metric.value is None
        or isinstance(actual, bool)
        or not math.isclose(float(actual), metric.value, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise ValueError("Metric does not match retrieved value")


def validate_output(output: BaseModel, state: AgentState, as_of: date) -> None:
    observations = state["observations"]
    context = state["context"]
    if isinstance(output, QuantitativeBrief):
        if output.ticker != state["ticker"]:
            raise ValueError("Incorrect brief ticker")
        for metric in output.metrics:
            validate_metric(metric, observations)
    elif isinstance(output, ClarificationResponse):
        request = context["request"]
        if output.question != request["question"]:
            raise ValueError("Answer must address the exact clarification request")
        if output.metric:
            if output.metric.name != request["requested_metric"]:
                raise ValueError("Clarification must supply the requested metric")
            validate_metric(output.metric, observations)
    elif isinstance(output, ResearchReport):
        if output.ticker != state["ticker"] or output.as_of != as_of:
            raise ValueError("Report ticker/date must match run")
        allowed = {item.evidence_id for item in observations if item.success}
        if not all(evidence_coverage(state).values()):
            raise ValueError(
                "Report requires quantitative and qualitative observations"
            )
        quantitative = {
            item.evidence_id
            for item in observations
            if item.success
            and item.tool_name in {"get_price_data", "calculate_volatility"}
        }
        if "brief" in context:
            allowed.add("analyst_brief")
            if any(
                metric["value"] is not None and metric["name"] != "sentiment_score"
                for metric in context["brief"]["metrics"]
            ):
                quantitative.add("analyst_brief")
        if "clarification" in context:
            allowed.add("analyst_clarification")
            clarification = context["clarification"]
            if output.clarification_used != clarification["answer"]:
                raise ValueError("Final report must explicitly incorporate the answer")
            if clarification["metric"] and clarification["metric"]["value"] is not None:
                quantitative.add("analyst_clarification")
                citations = [
                    source
                    for risk in output.top_three_risks
                    for source in risk.evidence_ids
                ]
                citations += output.hedge_strategy_recommendation.evidence_ids
                if "analyst_clarification" not in citations:
                    raise ValueError(
                        "The clarification must support the final analysis"
                    )
        citations = [
            source for risk in output.top_three_risks for source in risk.evidence_ids
        ]
        citations += output.hedge_strategy_recommendation.evidence_ids
        if not set(citations).issubset(allowed):
            raise ValueError("Report cites missing or failed evidence")
        if not set(output.hedge_strategy_recommendation.evidence_ids) & quantitative:
            raise ValueError("Hedge must cite quantitative evidence")


class AgentRuntime:
    def __init__(
        self,
        client: CompletionClient,
        executor: ToolExecutor,
        max_decisions: int = DEFAULT_DECISION_BUDGET,
    ) -> None:
        if not 1 <= max_decisions <= 30:
            raise ValueError("Decision budget must be 1–30")
        self.client, self.executor, self.max_decisions = client, executor, max_decisions

    def run(
        self,
        query: str,
        *,
        role: Role,
        stage: str,
        output_schema: type[BaseModel],
        as_of: date,
        context: dict[str, Any] | None = None,
        observations: list[ToolObservation] | None = None,
    ) -> StageResult:
        """The LLM chooses tools or finish; graph edges only enforce the loop."""
        allowed = ROLE_TOOLS[role]

        def decide(state: AgentState) -> dict[str, Any]:
            if state["steps"] >= self.max_decisions:
                return {
                    "error": "Decision budget exhausted; no fabricated report",
                    "trace": state["trace"]
                    + [event(role, "stopped", {"reason": "decision budget"})],
                }
            payload = redact(
                {
                    "query": state["query"],
                    "ticker": state["ticker"],
                    "as_of": as_of,
                    "role": role,
                    "stage": stage,
                    "remaining_decisions": self.max_decisions - state["steps"],
                    "failed_tool_calls": state["failed_calls"],
                    "report_evidence_coverage": evidence_coverage(state),
                    "allowed_tools": {
                        name: {
                            "description": TOOL_DESCRIPTIONS[name],
                            "arguments_schema": ARGUMENT_SCHEMAS[
                                name
                            ].model_json_schema(),
                        }
                        for name in sorted(allowed)
                    },
                    "action_schema": AgentAction.model_json_schema(),
                    "finish_output_schema": output_schema.model_json_schema(),
                    "allowed_evidence_ids": [
                        item.evidence_id
                        for item in state["observations"]
                        if item.success
                    ]
                    + (["analyst_brief"] if "brief" in state["context"] else [])
                    + (
                        ["analyst_clarification"]
                        if "clarification" in state["context"]
                        else []
                    ),
                    "canonical_metric_paths": METRIC_PATHS,
                    "observations": [
                        observation_view(item) for item in state["observations"]
                    ],
                    "available_headlines": [
                        vars(item)
                        for item in self.executor.tools.known_headlines.values()
                    ],
                    "handoff_context": state["context"],
                    "feedback": state["feedback"],
                }
            )
            user = json_payload(payload)
            trace = state["trace"] + [
                event(role, "llm_request", {"system": AGENT_SYSTEM, "user": payload})
            ]
            try:
                text = self.client.complete(AGENT_SYSTEM, user)
            except Exception as exc:
                category = error_category(exc)
                trace.append(event(role, "llm_failure", {"category": category}))
                failures = state["transport_failures"] + 1
                # GroqClient has already exhausted its bounded transport backoff.
                # An injected legacy client has a separate two-failure safety bound.
                terminal = (
                    isinstance(exc, (LLMTransportError, ValueError))
                    or failures >= MAX_UNCLASSIFIED_TRANSPORT_FAILURES
                )
                if not terminal:
                    time.sleep(1.0)
                return {
                    "transport_failures": failures,
                    "decision": None,
                    "trace": trace,
                    "error": f"LLM transport stopped: {category}" if terminal else None,
                    "feedback": f"LLM transport failure category={category}.",
                }
            try:
                trace.append(event(role, "llm_response", {"text": text}))
                action = parse_output(text, AgentAction)
                trace.append(event(role, "decision", action.model_dump()))
                if (
                    action.kind == "tool"
                    and tool_call_key(action.tool_name, action.arguments)
                    in state["failed_calls"]
                ):
                    trace.append(
                        event(
                            role,
                            "duplicate_failed_call_blocked",
                            {
                                "tool_name": action.tool_name,
                                "arguments": action.arguments,
                            },
                        )
                    )
                    return {
                        "steps": state["steps"] + 1,
                        "decision": None,
                        "trace": trace,
                        "feedback": "This exact tool call already failed and is blocked. "
                        "Choose another source/tool or change the arguments.",
                    }
                return {
                    "steps": state["steps"] + 1,
                    "decision": action,
                    "feedback": None,
                    "trace": trace,
                }
            except Exception:
                trace.append(
                    event(
                        role,
                        "llm_failure",
                        {"category": "validation"},
                    )
                )
                return {
                    "steps": state["steps"] + 1,
                    "decision": None,
                    "trace": trace,
                    "feedback": "Request failed or action JSON invalid. Correct or choose another approach.",
                }

        def call_tool(state: AgentState) -> dict[str, Any]:
            action = state["decision"]
            trace = state["trace"] + [
                event(
                    role,
                    "tool_call",
                    {"tool_name": action.tool_name, "arguments": action.arguments},
                )
            ]
            try:
                observation = self.executor.invoke(
                    role, action.tool_name, action.arguments
                )
                return {"pending": observation, "trace": trace}
            except Exception:
                # A trace storage failure is visible and stops unobservable execution.
                return {
                    "error": "Tool dispatch/trace storage unavailable",
                    "trace": trace,
                }

        def observe(state: AgentState) -> dict[str, Any]:
            if state["error"]:
                return {}
            observation = state["pending"]
            return {
                "failed_calls": state["failed_calls"]
                + (
                    [tool_call_key(observation.tool_name, observation.arguments)]
                    if not observation.success
                    else []
                ),
                "observations": state["observations"] + [observation],
                "trace": state["trace"]
                + [
                    event(role, "observation", observation_view(observation)),
                    event(
                        role,
                        "replan",
                        {
                            "after": observation.evidence_id,
                            "success": observation.success,
                        },
                    ),
                ],
                "pending": None,
            }

        def finish(state: AgentState) -> dict[str, Any]:
            try:
                output = output_schema.model_validate(redact(state["decision"].output))
                validate_output(output, state, as_of)
                return {
                    "output": output,
                    "trace": state["trace"]
                    + [event(role, "stage_output", output.model_dump(mode="json"))],
                }
            except Exception:
                return {
                    "feedback": "Final output invalid or ungrounded. Correct schema, ticker/date, metric paths and evidence IDs. "
                    f"Report evidence coverage: {evidence_coverage(state)}; "
                    "a report needs quantitative price/volatility AND qualitative news/search evidence.",
                    "trace": state["trace"]
                    + [
                        event(
                            role,
                            "output_rejected",
                            {"reason": "schema or evidence validation"},
                        )
                    ],
                }

        graph = StateGraph(AgentState)
        graph.add_node("decide", decide)
        graph.add_node("tool", call_tool)
        graph.add_node("observe", observe)
        graph.add_node("finish", finish)
        graph.add_edge(START, "decide")
        graph.add_conditional_edges(
            "decide",
            lambda state: (
                "stop"
                if state["error"]
                else "decide"
                if state["decision"] is None
                else "tool"
                if state["decision"].kind == "tool"
                else "finish"
            ),
            {"stop": END, "decide": "decide", "tool": "tool", "finish": "finish"},
        )
        graph.add_edge("tool", "observe")
        graph.add_conditional_edges(
            "observe",
            lambda state: "stop" if state["error"] else "decide",
            {"stop": END, "decide": "decide"},
        )
        graph.add_conditional_edges(
            "finish",
            lambda state: "stop" if state["output"] is not None else "decide",
            {"stop": END, "decide": "decide"},
        )
        initial: AgentState = {
            "query": redact(query),
            "ticker": self.executor.tools.ticker,
            "role": role,
            "stage": stage,
            "steps": 0,
            "transport_failures": 0,
            "failed_calls": [
                tool_call_key(item.tool_name, item.arguments)
                for item in (observations or [])
                if not item.success
            ],
            "decision": None,
            "observations": observations or [],
            "pending": None,
            "trace": [],
            "output": None,
            "feedback": None,
            "error": None,
            "context": redact(context or {}),
        }
        result = graph.compile().invoke(
            initial, {"recursion_limit": 5 * self.max_decisions + 10}
        )
        return StageResult(
            result["output"], result["observations"], result["trace"], result["error"]
        )
