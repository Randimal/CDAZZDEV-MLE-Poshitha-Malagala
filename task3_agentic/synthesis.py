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
from task3_agentic.schemas import ResearchReport
from task3_agentic.state import AgentState, compact_schema, observation_view
from task3_agentic.tracing import event, redact
from task3_agentic.validation import safe_validation_feedback

SYNTHESIS_ATTEMPTS = 2


def final_synthesis(
    client: CompletionClient,
    state: AgentState,
    as_of: date,
    validate: Callable[[BaseModel, AgentState, date], None],
) -> dict[str, Any]:
    """No tools or planner loop are reachable from this synthesis node."""
    role = state["role"]
    payload = redact(
        {
            "stage": "final_synthesis",
            "query": state["query"],
            "ticker": state["ticker"],
            "as_of": as_of,
            "report_schema": compact_schema(ResearchReport.model_json_schema()),
            "allowed_evidence_ids": [
                item.evidence_id for item in state["observations"] if item.success
            ],
            "observations": [observation_view(item) for item in state["observations"]],
        }
    )
    trace = state["trace"] + [
        event(
            role, "synthesis_started", {"evidence_ids": payload["allowed_evidence_ids"]}
        )
    ]
    json_mode = True
    feedback = ""
    for attempt in range(1, SYNTHESIS_ATTEMPTS + 1):
        user = json_payload(payload)
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
                client.complete(SYNTHESIS_SYSTEM, user, json_mode=False)
                if not json_mode and isinstance(client, GroqClient)
                else client.complete(SYNTHESIS_SYSTEM, user)
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
            trace.append(event(role, "llm_response", {"text": text}))
            try:
                output = parse_output(text, ResearchReport)
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
                        "output": output,
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
        payload = {**payload, "repair_feedback": feedback}
    return {
        "output": None,
        "error": f"Final synthesis validation exhausted: {feedback}; no fabricated report",
        "trace": trace,
    }
