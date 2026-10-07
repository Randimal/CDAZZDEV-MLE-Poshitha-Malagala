"""Small conservative report checks; current tools do not retrieve audited accounts."""

import math
import re
from typing import Any

from pydantic import ValidationError

from task3_agentic.schemas import ResearchReport
from task3_agentic.tools import horizon_volatility

MOVE_ROUNDING_TOLERANCE_PCT = 0.06  # Percentage points; permits one-decimal rounding.


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
    r"\b(?:is|are|has|shows|remains|solid|strong|healthy|good|robust|sound|stable|weak|poor|high|low|improving|deteriorating|insolvent|solvent)\b",
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


def grounding_facts(observations: list[Any], context: dict[str, Any]) -> dict[str, Any]:
    """Compact deterministic facts shared by prompts and local claim checks.

    Sentiment total is successes + failures, not the number of successful results.
    The writer receives these numeric facts from A without acquiring A's tools.
    """
    inherited = context.get("grounding_facts", {})
    volatility = list(inherited.get("volatility", []))
    sentiment = list(inherited.get("sentiment", []))
    for item in observations:
        if not item.success:
            continue
        if item.tool_name == "llm_sentiment":
            aggregate = item.output["aggregate"]
            sentiment.append(
                {
                    "evidence_id": item.evidence_id,
                    "total_headline_count": aggregate["successful_count"]
                    + aggregate["failed_count"],
                    "requested_headline_count": len(item.arguments.get("headlines", []))
                    or aggregate["successful_count"] + aggregate["failed_count"],
                    **{
                        key: aggregate[key]
                        for key in (
                            "successful_count",
                            "failed_count",
                            "positive_count",
                            "negative_count",
                            "neutral_count",
                            "overall_score",
                        )
                    },
                }
            )
    for value in annualized_values(observations, context):
        if not any(
            math.isclose(value, item["annualized_volatility"]) for item in volatility
        ):
            volatility.append(
                {
                    "annualized_volatility": value,
                    "historical_90_trading_day_sigma": horizon_volatility(value),
                    "historical_90_trading_day_sigma_pct": 100
                    * horizon_volatility(value),
                    "formula": "annualized_volatility * sqrt(90 / 252)",
                    "interpretation": "Historical one-standard-deviation return scale over 90 trading days; not a forecast. Do not scale this again.",
                }
            )
    return {"volatility": volatility, "sentiment": sentiment}


def validate_report_language(
    report: ResearchReport, annualized: list[float], facts: dict[str, Any] | None = None
) -> None:
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
            option_valuation = re.search(
                r"\b(?:implied volatility\s+(?:is\s+)?(?:cheap|expensive)|"
                r"(?:option\s+)?premium\s+(?:is\s+)?attractive)\b",
                sentence,
                re.I,
            )
            if option_valuation and not _discloses_unavailability(
                sentence, option_valuation
            ):
                raise GroundingError(
                    "hedge_strategy_recommendation: premium/implied-volatility valuation requires unavailable option-chain data; state execution limitations"
                )
            valuation = re.search(
                r"\b(?:higher|lower|above|below|premium|discount|expensive|cheap)\b",
                sentence,
                re.I,
            )
            if (
                valuation
                and re.search(r"histor(?:y|ical)|sector", sentence, re.I)
                and re.search(
                    r"\bP/?E\b|valuation|valued|ratio|expensive|cheap", sentence, re.I
                )
                and not _discloses_unavailability(sentence, valuation)
            ):
                raise GroundingError(
                    f"{field}: historical/sector PE comparison requires benchmark evidence not supplied; report the PE value only"
                )
            if field == "hedge_strategy_recommendation":
                sizing = re.search(
                    r"\b(?:offset|hedge|protect|cover|reduce|cut|sell|allocate)\s+(?:(?:about|approximately|around|roughly|up to)\s+)?[~≈]?\s*\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\s*%\s+(?:of\s+(?:the\s+)?(?:equity\s+)?(?:exposure|position|portfolio)|hedge|stop[- ]loss)|\b(?:stop[- ]loss|hedge ratio)\b[^.;]{0,30}\d+(?:\.\d+)?\s*%|\b(?:strike|premium|option price)\b\s+(?:is|at|of)\s*\$?\s*\d+(?:\.\d+)?|\b(?:strike|premium|option price)\b[^.;]{0,20}\$\s*\d+(?:\.\d+)?",
                    sentence,
                    re.I,
                )
                instrument = re.search(
                    rf"\b{re.escape(report.ticker)}\s+futures\b", sentence, re.I
                )
                claim = sizing or instrument
                if claim and not _discloses_unavailability(sentence, claim):
                    raise GroundingError(
                        "hedge_strategy_recommendation: unsupported numerical sizing/stop-loss/option price or ticker futures instrument; exact sizing needs beta/correlation/exposure or option-chain data"
                    )
            validate_sentiment_counts(
                field, sentence, (facts or {}).get("sentiment", [])
            )
    patterns = (
        r"90(?:[- ]trading)?[- ]day\s+(?:historical\s+)?(?:one[- ]standard[- ]deviation\s+|expected\s+|1[- ]sigma\s+)?(?:return scale|move|volatility)(?:\s+(?:is|of|estimate|approximately|about|equals))*\s*[:=≈~]?\s*(\d+(?:\.\d+)?)\s*%",
        r"(\d+(?:\.\d+)?)\s*%\s+(?:as\s+)?(?:a\s+)?(?:historical\s+)?(?:90(?:[- ]trading)?[- ]day\s+)?(?:one[- ]sigma|1[- ]sigma|one[- ]standard[- ]deviation)(?:\s+(?:move|return scale))?",
        r"(?:one[- ]sigma|1[- ]sigma|one[- ]standard[- ]deviation)\s+(?:return scale|move)(?:\s+(?:is|of|approximately|about))*\s*[:=≈~]?\s*(\d+(?:\.\d+)?)\s*%",
        r"(\d+(?:\.\d+)?)\s*%\s+(?:as\s+)?(?:a\s+)?90[- ]day\s+(?:expected\s+)?move",
    )
    for pattern in patterns:
        for field, text in fields.items():
            for match in re.finditer(pattern, text, re.I):
                # Explicit annualized/daily sigma is not a horizon-scale claim.
                if re.search(
                    r"(?:annualized|daily)\s+(?:historical\s+)?$",
                    text[max(0, match.start() - 35) : match.start()],
                    re.I,
                ):
                    continue
                percent = float(match.group(1))
                if any(
                    math.isclose(
                        percent,
                        100 * horizon_volatility(value),
                        abs_tol=MOVE_ROUNDING_TOLERANCE_PCT,
                    )
                    for value in annualized
                ):
                    continue
                raise GroundingError(
                    f"{field}: 90-day one-sigma return scale must equal annualized_volatility * sqrt(90 / 252); use supplied historical_90_trading_day_sigma_pct without scaling again, not a forecast"
                )


def validate_sentiment_counts(
    field: str, text: str, aggregates: list[dict[str, Any]]
) -> None:
    """Check explicit numerical coverage claims, not general sentiment language."""
    patterns = {
        "successful_count": (
            r"\b(\d+)\s+of\s+\d+\s+(?:recent\s+)?headlines\s+(?:were\s+)?(?:successfully\s+)?analy[sz]ed",
            r"\b(?:analysis|analy[sz]ed)\s+(?:of\s+)?(\d+)\s+(?:recent\s+)?headlines",
            r"\b(\d+)\s+(?:recent\s+)?headlines\s+(?:were\s+)?(?:successfully\s+)?analy[sz]ed",
        ),
        "failed_count": (r"\b(\d+)\s+(?:headline\s+)?(?:analyses\s+)?failed\b",),
        "positive_count": (r"\b(\d+)\s+positive\b",),
        "negative_count": (r"\b(\d+)\s+negative\b",),
        "neutral_count": (r"\b(\d+)\s+neutral\b",),
        "total_headline_count": (r"\b\d+\s+of\s+(\d+)\s+(?:recent\s+)?headlines",),
    }
    for key, expressions in patterns.items():
        for expression in expressions:
            for match in re.finditer(expression, text, re.I):
                if key == "successful_count" and re.search(
                    r"\d+\s+of\s+$", text[: match.start()]
                ):
                    # In "19 of 20 headlines analyzed", 20 is the denominator,
                    # not a second independent analyzed-count claim.
                    continue
                count = int(match.group(1))
                if not any(count == aggregate[key] for aggregate in aggregates):
                    raise GroundingError(
                        f"{field}: {key} must match supplied sentiment coverage; failures are not successful or neutral analyses"
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
    values.extend(
        item["annualized_volatility"]
        for item in context.get("grounding_facts", {}).get("volatility", [])
    )
    return values
