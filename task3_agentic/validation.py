"""Small conservative report checks; current tools do not retrieve audited accounts."""

import math
import re
from typing import Any

from pydantic import ValidationError

from task3_agentic.schemas import ResearchReport
from task3_agentic.tools import horizon_volatility

MOVE_ROUNDING_TOLERANCE_PCT = 0.02


class GroundingError(ValueError):
    """A static, safe field/evidence explanation suitable for planner feedback."""


def safe_validation_feedback(exc: Exception) -> str:
    """Use field/error types only, never model input, raw errors or response bodies."""
    if isinstance(exc, GroundingError):
        return str(exc)
    if isinstance(exc, ValidationError):
        issues = []
        for issue in exc.errors(include_input=False, include_context=False)[:4]:
            # Do not echo arbitrary extra-field names supplied by the model.
            fields = set(ResearchReport.model_fields) | {
                "title",
                "supporting_evidence",
                "evidence_ids",
                "strategy",
                "data_driven_rationale",
                "question",
                "answer",
                "metric",
                "metrics",
                "name",
                "value",
                "path",
                "summary",
                "requested_metric",
                "reason",
            }
            location = (
                ".".join(
                    str(part) if isinstance(part, int) or part in fields else "field"
                    for part in issue["loc"]
                )
                or "output"
            )
            detail = (
                "required non-empty string; copy handoff_context.clarification.answer verbatim"
                if location == "clarification_used"
                else issue["type"]
            )
            issues.append(f"{location}: {detail}")
        return "; ".join(issues)
    return "output: invalid structure or unavailable evidence path; use supplied schema and evidence IDs"


FUNDAMENTAL_TERMS = re.compile(
    r"balance[- ]sheet|solven(?:cy|t)|cash[- ]flow|earnings[- ]quality", re.I
)
FUNDAMENTAL_ASSERTION = re.compile(
    r"\b(?:is|are|has|shows|remains|solid|strong|healthy|robust|sound|stable|weak|poor|high|low|improving|deteriorating|insolvent|solvent)\b",
    re.I,
)


def _discloses_unavailability(sentence: str, match: re.Match[str]) -> bool:
    """Permit explicit limitations, not claims followed by a vague caveat."""
    prefix, suffix = sentence[: match.start()], sentence[match.end() :]
    return bool(
        re.search(
            r"(?:cannot|can't|does not|do not|not|no|insufficient|unavailable)\b[^.;]*$",
            prefix,
            re.I,
        )
        or re.match(
            r"(?:\s+health|\s+strength)?\s+(?:is |are )?(?:not assessed|unavailable|unknown|cannot be assessed|not established)\b",
            suffix,
            re.I,
        )
    )


def validate_report_language(report: ResearchReport, annualized: list[float]) -> None:
    """Reject unsupported fundamentals and ungrounded hedge optimization/move math.

    News/search snippets are not audited accounts or an option chain. Fundamental
    conclusions are therefore unavailable in this assessment; disclose that limit.
    This deliberately simple sentence check is not comprehensive NLP moderation.
    """
    fields = {
        "financial_health_summary": report.financial_health_summary,
        "market_sentiment_summary": report.market_sentiment_summary,
        **{
            f"top_three_risks.{i}": f"{risk.title}. {risk.supporting_evidence}"
            for i, risk in enumerate(report.top_three_risks)
        },
        "hedge_strategy_recommendation": " ".join(
            (
                report.hedge_strategy_recommendation.strategy,
                report.hedge_strategy_recommendation.data_driven_rationale,
                report.hedge_strategy_recommendation.limitations,
            )
        ),
        "clarification_used": report.clarification_used or "",
        "limitations": " ".join(report.limitations),
    }
    for field, text in fields.items():
        # Split sentences without splitting decimal percentages.
        for sentence in re.split(r"(?<!\d)[.!?;]\s+|(?<=\d)[.!?;](?!\d)\s+", text):
            for match in FUNDAMENTAL_TERMS.finditer(sentence):
                if FUNDAMENTAL_ASSERTION.search(
                    sentence
                ) and not _discloses_unavailability(sentence, match):
                    raise GroundingError(
                        f"{field}: unsupported fundamental-health claim; no audited balance-sheet/cash-flow/earnings evidence. Describe market/technical condition or explicitly state unavailable."
                    )
            optimal = re.search(r"\boptimal\b", sentence, re.I)
            if (
                optimal
                and re.search(r"put|strike|premium|cost", sentence, re.I)
                and not _discloses_unavailability(sentence, optimal)
            ):
                raise GroundingError(
                    "hedge_strategy_recommendation: optimal strike/cost cannot be established without option-chain/implied-volatility data; give a hedge concept with execution limitations"
                )
    hedge = fields["hedge_strategy_recommendation"]
    patterns = (
        r"90[- ]day\s+(?:one[- ]standard[- ]deviation\s+|expected\s+|1[- ]sigma\s+)?(?:move|volatility)(?:\s+(?:is|of|estimate|approximately|about|equals))*\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%",
        r"(\d+(?:\.\d+)?)\s*%\s+(?:as\s+)?(?:a\s+)?90[- ]day\s+(?:expected\s+|one[- ]standard[- ]deviation\s+)?move",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, hedge, re.I):
            percent = float(match.group(1))
            if not any(
                math.isclose(
                    percent,
                    100 * horizon_volatility(value),
                    abs_tol=MOVE_ROUNDING_TOLERANCE_PCT,
                )
                for value in annualized
            ):
                raise GroundingError(
                    "hedge_strategy_recommendation: 90-day move must equal annualized_volatility * sqrt(90 / 252), under a 90-trading-day historical scaling assumption; annualized volatility itself is not a 90-day expected move"
                )


def annualized_values(observations: list[Any], context: dict[str, Any]) -> list[float]:
    values = [
        item.output["annualized_volatility"]
        for item in observations
        if item.success and item.tool_name == "calculate_volatility"
    ]
    metrics = list(context.get("brief", {}).get("metrics", []))
    metric = context.get("clarification", {}).get("metric")
    if metric:
        metrics.append(metric)
    values.extend(
        item["value"]
        for item in metrics
        if item["name"] == "annualized_volatility" and item["value"] is not None
    )
    return values
