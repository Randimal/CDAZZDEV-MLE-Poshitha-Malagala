"""Shared LangGraph decide -> tool -> observe -> replan execution loop."""

import json
import math
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from task1_financial.json_utils import json_safe
from task1_financial.llm import (
    CompletionClient,
    GroqClient,
    LLMTransportError,
    error_category,
    parse_output,
    safe_provider_details,
)
from task3_agentic.prompts import (
    AGENT_SYSTEM,
    CRITIQUE_INSTRUCTIONS,
    FINANCIAL_GROUNDING,
    SINGLE_PLANNER_SYSTEM,
    TOOL_DESCRIPTIONS,
)
from task3_agentic.schemas import (
    METRIC_PATHS,
    AgentAction,
    ClarificationRequest,
    ClarificationResponse,
    QuantitativeBrief,
    QuantMetric,
    ResearchDecision,
    ResearchReport,
    Role,
    ToolObservation,
    TraceEvent,
    WriterResearchReport,
)
from task3_agentic.state import AgentState, compact_schema, observation_view
from task3_agentic.synthesis import final_synthesis
from task3_agentic.tools import (
    ARGUMENT_SCHEMAS,
    ROLE_TOOLS,
    ToolExecutor,
    tool_call_key,
)
from task3_agentic.tracing import event, redact
from task3_agentic.validation import (
    GroundingError,
    annualized_values,
    grounding_facts,
    safe_validation_feedback,
    validate_report_language,
    validate_sentiment_counts,
)

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
        raise GroundingError(
            "metrics.evidence_id/path: cite a successful observation and canonical metric path"
        )
    actual = sources[metric.evidence_id].output
    for key in metric.path.split("."):
        actual = actual[key]
    if actual is None:
        if metric.value is not None:
            raise GroundingError(
                "metrics.value: retrieved missing value must remain null"
            )
    elif (
        metric.value is None
        or isinstance(actual, bool)
        or not math.isclose(float(actual), metric.value, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise GroundingError(
            "metrics.value: copy the exact retrieved value at the cited path"
        )


def validate_output(output: BaseModel, state: AgentState, as_of: date) -> None:
    observations = state["observations"]
    context = state["context"]
    if isinstance(output, QuantitativeBrief):
        if output.ticker != state["ticker"]:
            raise GroundingError("ticker: must match research ticker")
        for metric in output.metrics:
            validate_metric(metric, observations)
    elif isinstance(output, ClarificationRequest):
        known = {
            metric["name"]
            for metric in context["brief"]["metrics"]
            if metric["value"] is not None
        }
        if output.requested_metric in known:
            raise GroundingError(
                "requested_metric: already supplied in analyst brief; request one missing quantitative/sentiment analysis"
            )
        if not evidence_coverage(state)["qualitative"]:
            raise GroundingError(
                "clarification evidence: retrieve news/search before identifying missing analyst analysis"
            )
    elif isinstance(output, ClarificationResponse):
        request = context["request"]
        if output.question != request["question"]:
            raise GroundingError(
                "question: copy the exact clarification request question"
            )
        if output.metric:
            if output.metric.name != request["requested_metric"]:
                raise GroundingError("metric.name: must match request.requested_metric")
            validate_metric(output.metric, observations)
        if request["requested_metric"] == "sentiment_score":
            attempts = [
                item for item in observations if item.tool_name == "llm_sentiment"
            ]
            if not attempts:
                raise GroundingError(
                    "metric: sentiment_score requires an actual llm_sentiment call on supplied headlines"
                )
            if (
                any(
                    item.success
                    and item.output["aggregate"]["overall_score"] is not None
                    for item in attempts
                )
                and output.metric is None
            ):
                raise GroundingError(
                    "metric: copy the available validated sentiment_score; do not replace it with null"
                )
            validate_sentiment_counts(
                "answer",
                output.answer,
                grounding_facts(observations, context)["sentiment"],
            )
    elif isinstance(output, ResearchReport):
        if output.ticker != state["ticker"] or output.as_of != as_of:
            raise GroundingError(
                "ticker/as_of: must match research ticker and run date"
            )
        allowed = {item.evidence_id for item in observations if item.success}
        if not all(evidence_coverage(state).values()):
            raise GroundingError(
                "evidence coverage: need successful price/volatility AND news/search observations"
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
                raise GroundingError(
                    "clarification_used: must be non-empty and equal handoff_context.clarification.answer verbatim; an evidence citation alone is insufficient"
                )
            if clarification["metric"] and clarification["metric"]["value"] is not None:
                quantitative.add("analyst_clarification")
                citations = [
                    source
                    for risk in output.top_three_risks
                    for source in risk.evidence_ids
                ]
                citations += output.hedge_strategy_recommendation.evidence_ids
                if "analyst_clarification" not in citations:
                    raise GroundingError(
                        "evidence_ids: cite analyst_clarification in a risk or hedge when its metric is available"
                    )
        citations = [
            source for risk in output.top_three_risks for source in risk.evidence_ids
        ]
        citations += output.hedge_strategy_recommendation.evidence_ids
        if not set(citations).issubset(allowed):
            raise GroundingError(
                "evidence_ids: report cites missing/failed evidence; use allowed_evidence_ids only"
            )
        if not set(output.hedge_strategy_recommendation.evidence_ids) & quantitative:
            raise GroundingError(
                "hedge_strategy_recommendation.evidence_ids: must cite quantitative price/volatility or validated analyst evidence"
            )
        validate_report_language(
            output,
            annualized_values(observations, context),
            grounding_facts(observations, context),
        )


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
        separate_synthesis = stage == "single_research"
        action_schema = ResearchDecision if separate_synthesis else AgentAction
        system = (
            SINGLE_PLANNER_SYSTEM
            if separate_synthesis
            else AGENT_SYSTEM + CRITIQUE_INSTRUCTIONS
        )
        if stage == "writer_final":
            output_schema = WriterResearchReport
            system += FINANCIAL_GROUNDING

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
                            "arguments_schema": compact_schema(
                                ARGUMENT_SCHEMAS[name].model_json_schema()
                            ),
                        }
                        for name in sorted(allowed)
                    },
                    "action_schema": compact_schema(action_schema.model_json_schema()),
                    "finish_output_schema": compact_schema(
                        output_schema.model_json_schema()
                    ),
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
                    "canonical_metric_paths": METRIC_PATHS
                    if output_schema in {QuantitativeBrief, ClarificationResponse}
                    else {},
                    "observations": [
                        observation_view(item) for item in state["observations"]
                    ],
                    "available_headlines": [
                        {"title": item.title}
                        for item in self.executor.tools.known_headlines.values()
                    ]
                    if "llm_sentiment" in allowed
                    else [],
                    "handoff_context": {
                        key: value
                        for key, value in state["context"].items()
                        if key != "grounding_facts"
                    },
                    "grounding_facts": grounding_facts(
                        state["observations"], state["context"]
                    ),
                    "feedback": state["feedback"],
                }
            )
            if separate_synthesis:
                # The report schema is supplied once to synthesis, never to the planner.
                payload.pop("finish_output_schema")
            user = json.dumps(
                json_safe(payload),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            trace = state["trace"] + [
                event(
                    role,
                    "llm_request",
                    {
                        "system": system,
                        "user": payload,
                        "prompt_chars": len(system) + len(user),
                    },
                )
            ]
            try:
                text = self.client.complete(system, user)
            except Exception as exc:
                if (
                    stage in {"analyst_clarify", "writer_final"}
                    and isinstance(self.client, GroqClient)
                    and isinstance(exc, LLMTransportError)
                    and exc.status_code == 400
                    and exc.error_code == "json_validate_failed"
                    and state["json_repairs"] == 0
                ):
                    # This is an application JSON repair, not a transport retry
                    # or another planning step. The graph still executes tools.
                    state = {**state, "json_repairs": 1}
                    payload = {
                        **payload,
                        "feedback": "output: provider rejected JSON generation; return one valid JSON action object matching action_schema. Finish output must match finish_output_schema; use exact supplied metric values.",
                    }
                    user = json.dumps(
                        json_safe(payload), allow_nan=False, separators=(",", ":")
                    )
                    trace += [
                        event(
                            role,
                            "llm_failure",
                            {"category": "provider_json", **safe_provider_details(exc)},
                        ),
                        event(
                            role,
                            "llm_request",
                            {
                                "system": system,
                                "user": payload,
                                "json_mode": False,
                                "prompt_chars": len(system) + len(user),
                            },
                        ),
                    ]
                    try:
                        text = self.client.complete(system, user, json_mode=False)
                    except Exception as repair_exc:
                        return {
                            "json_repairs": 1,
                            "decision": None,
                            "error": f"JSON repair stopped: {error_category(repair_exc)}; no fabricated output",
                            "trace": trace
                            + [
                                event(
                                    role,
                                    "llm_failure",
                                    {
                                        "category": error_category(repair_exc),
                                        **safe_provider_details(repair_exc),
                                    },
                                )
                            ],
                        }
                else:
                    return transport_failure(state, trace, exc)
            try:
                trace.append(event(role, "llm_response", {"text": text}))
                action = parse_output(text, action_schema)
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
                        "json_repairs": state["json_repairs"],
                        "decision": None,
                        "trace": trace,
                        "feedback": "This exact tool call already failed and is blocked. Choose another source/tool or change the arguments.",
                    }
                return {
                    "steps": state["steps"] + 1,
                    "json_repairs": state["json_repairs"],
                    "decision": action,
                    "feedback": None,
                    "trace": trace,
                }
            except Exception as exc:
                trace.append(event(role, "llm_failure", {"category": "validation"}))
                return {
                    "steps": state["steps"] + 1,
                    "json_repairs": state["json_repairs"],
                    "decision": None,
                    "trace": trace,
                    "feedback": safe_validation_feedback(exc),
                    "error": "JSON repair validation exhausted; no fabricated output"
                    if state["json_repairs"]
                    else None,
                }

        def transport_failure(
            state: AgentState, trace: list[TraceEvent], exc: Exception
        ) -> dict[str, Any]:
            category = error_category(exc)
            trace.append(
                event(
                    role,
                    "llm_failure",
                    {"category": category, **safe_provider_details(exc)},
                )
            )
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
            if stage == "writer_final":
                return final_synthesis(
                    self.client,
                    state,
                    as_of,
                    validate_output,
                    initial_output=state["decision"].output,
                )
            try:
                if separate_synthesis:
                    if not all(evidence_coverage(state).values()):
                        raise GroundingError(
                            "evidence coverage: need successful price/volatility AND news/search observations before final_synthesis"
                        )
                    return {
                        "trace": state["trace"]
                        + [event(role, "research_complete", evidence_coverage(state))]
                    }
                output = output_schema.model_validate(redact(state["decision"].output))
                validate_output(output, state, as_of)
                if isinstance(output, WriterResearchReport):
                    # Validate with the writer contract, then keep the canonical
                    # stored model identical to the cache's ResearchReport type.
                    output = ResearchReport.model_validate(output.model_dump())
                return {
                    "output": output,
                    "trace": state["trace"]
                    + [event(role, "stage_output", output.model_dump(mode="json"))],
                }
            except Exception as exc:
                feedback = safe_validation_feedback(exc)
                failures = state["output_failures"] + 1
                return {
                    "feedback": feedback,
                    "output_failures": failures,
                    "error": f"JSON repair validation exhausted: {feedback}; no fabricated output"
                    if state["json_repairs"]
                    else None,
                    "trace": state["trace"]
                    + [
                        event(
                            role,
                            "output_rejected",
                            {"reason": feedback, "attempt": failures},
                        )
                    ],
                }

        graph = StateGraph(AgentState)
        graph.add_node("decide", decide)
        graph.add_node("tool", call_tool)
        graph.add_node("observe", observe)
        graph.add_node("finish", finish)
        if separate_synthesis:
            graph.add_node(
                "final_synthesis",
                lambda state: final_synthesis(
                    self.client, state, as_of, validate_output
                ),
            )
            graph.add_edge("final_synthesis", END)
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
            lambda state: (
                "stop"
                if state["output"] is not None or state["error"]
                else "synthesize"
                if separate_synthesis and state["feedback"] is None
                else "decide"
            ),
            {
                "stop": END,
                "decide": "decide",
                **({"synthesize": "final_synthesis"} if separate_synthesis else {}),
            },
        )
        initial: AgentState = {
            "query": redact(query),
            "ticker": self.executor.tools.ticker,
            "role": role,
            "stage": stage,
            "steps": 0,
            "transport_failures": 0,
            "json_repairs": 0,
            "output_failures": 0,
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
