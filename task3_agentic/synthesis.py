"""One final-report generation plus at most one targeted, locally validated repair."""

from collections.abc import Callable
from datetime import date
from typing import Any

from pydantic import BaseModel, ValidationError

from task1_financial.json_utils import json_payload
from task1_financial.llm import (
    CompletionClient,
    GroqClient,
    LLMTransportError,
    error_category,
    parse_output,
    safe_provider_details,
)
from task3_agentic.prompts import SYNTHESIS_SYSTEM
from task3_agentic.schemas import AgentAction, ResearchReport, WriterResearchReport
from task3_agentic.state import AgentState, compact_schema, observation_view
from task3_agentic.tracing import event, redact
from task3_agentic.validation import grounding_facts, safe_validation_feedback

SYNTHESIS_ATTEMPTS = 2


def final_synthesis(
    client: CompletionClient,
    state: AgentState,
    as_of: date,
    validate: Callable[[BaseModel, AgentState, date], None],
    *,
    initial_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """No tools or planner loop are reachable from this synthesis node."""
    role = state["role"]
    writer = state["stage"] == "writer_final"
    schema = WriterResearchReport if writer else ResearchReport
    payload = redact(
        {
            "stage": "writer_final" if writer else "final_synthesis",
            "query": state["query"],
            "ticker": state["ticker"],
            "as_of": as_of,
            "finish_output_schema" if writer else "report_schema": compact_schema(
                schema.model_json_schema()
            ),
            "allowed_evidence_ids": [
                item.evidence_id for item in state["observations"] if item.success
            ]
            + (["analyst_brief", "analyst_clarification"] if writer else []),
            "observations": [observation_view(item) for item in state["observations"]],
            "grounding_facts": grounding_facts(state["observations"], state["context"]),
            **(
                {
                    "handoff_context": {
                        key: value
                        for key, value in state["context"].items()
                        if key != "grounding_facts"
                    },
                }
                if writer
                else {}
            ),
        }
    )
    trace = state["trace"] + [
        event(
            role, "synthesis_started", {"evidence_ids": payload["allowed_evidence_ids"]}
        )
    ]
    json_mode = True
    feedback = ""
    attempts = 1 if writer and state["json_repairs"] else SYNTHESIS_ATTEMPTS
    for attempt in range(1, attempts + 1):
        # The writer's initial finish output is validated here. Its only repair
        # stays inside this node and consumes no planner steps or tool calls.
        supplied_output = writer and attempt == 1
        user = json_payload(payload)
        if not supplied_output:
            trace.append(
                event(
                    role,
                    "llm_request",
                    {
                        "system": SYNTHESIS_SYSTEM,
                        "user": payload,
                        "prompt_chars": len(SYNTHESIS_SYSTEM) + len(user),
                    },
                )
            )
        try:
            text = (
                ""
                if supplied_output
                else (
                    client.complete(SYNTHESIS_SYSTEM, user, json_mode=False)
                    if not json_mode and isinstance(client, GroqClient)
                    else client.complete(SYNTHESIS_SYSTEM, user)
                )
            )
        except Exception as exc:
            details = safe_provider_details(exc)
            trace.append(
                event(
                    role,
                    "llm_failure",
                    {"category": error_category(exc), **details},
                )
            )
            if (
                isinstance(exc, LLMTransportError)
                and exc.status_code == 400
                and exc.error_code == "json_validate_failed"
                and isinstance(client, GroqClient)
                and not getattr(client, "handles_provider_json_fallback", False)
            ):
                json_mode = False
                feedback = "output: provider rejected JSON syntax; return one valid JSON object"
                category = "provider_json"
            else:
                return {
                    "output": None,
                    "error": f"Final synthesis transport stopped: {error_category(exc)}",
                    "trace": trace,
                }
        else:
            if not supplied_output:
                trace.append(event(role, "llm_response", {"text": text}))
            try:
                if supplied_output:
                    output = schema.model_validate(redact(initial_output))
                else:
                    try:
                        output = parse_output(text, schema)
                    except ValidationError as report_error:
                        if not writer:
                            raise
                        # Accept the existing finish envelope as well as the
                        # requested report object, with both schemas enforced.
                        try:
                            action = parse_output(text, AgentAction)
                        except ValidationError:
                            raise report_error
                        if action.kind != "finish":
                            raise ValueError("Report repair cannot invoke tools")
                        output = schema.model_validate(action.output)
            except ValidationError as exc:
                feedback, category = safe_validation_feedback(exc), "schema_validation"
            except (ValueError, TypeError):
                feedback = "output: invalid JSON syntax; return one valid JSON object"
                category = "json_parse"
            else:
                try:
                    validate(output, state, as_of)
                except Exception as exc:
                    feedback, category = safe_validation_feedback(exc), "grounding"
                else:
                    return {
                        "output": ResearchReport.model_validate(output.model_dump()),
                        "trace": trace
                        + [event(role, "stage_output", output.model_dump(mode="json"))],
                    }
        trace.append(
            event(
                role,
                "output_rejected",
                {"reason": feedback, "category": category, "attempt": attempt},
            )
        )
        payload = {**payload, "repair_feedback": feedback, "feedback": feedback}
    return {
        "output": None,
        "error": f"{'Writer output' if writer else 'Final synthesis'} validation exhausted: {feedback}; no fabricated report",
        "trace": trace,
    }
