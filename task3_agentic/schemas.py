"""Structured decisions, grounded handoffs and final research results."""

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["researcher", "analyst", "writer"]
MetricName = Literal[
    "current_price",
    "pe_ratio",
    "ytd_return_pct",
    "SMA_50",
    "SMA_200",
    "RSI",
    "MACD",
    "MACD_signal",
    "MACD_histogram",
    "BB_upper",
    "BB_middle",
    "BB_lower",
    "annualized_volatility",
    "sentiment_score",
]

METRIC_PATHS = {
    "current_price": "summary.current_price",
    "pe_ratio": "summary.pe_ratio",
    "ytd_return_pct": "summary.ytd_return_pct",
    "annualized_volatility": "annualized_volatility",
    "sentiment_score": "aggregate.overall_score",
    **{
        name: f"latest_indicators.{name}"
        for name in (
            "SMA_50",
            "SMA_200",
            "RSI",
            "MACD",
            "MACD_signal",
            "MACD_histogram",
            "BB_upper",
            "BB_middle",
            "BB_lower",
        )
    },
}


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AgentAction(Model):
    kind: Literal["tool", "finish"]
    reason: str = Field(min_length=1, max_length=500)
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] | None = None

    @model_validator(mode="after")
    def check_action(self) -> "AgentAction":
        if self.kind == "tool" and (not self.tool_name or self.output is not None):
            raise ValueError("Tool action needs a name and no final output")
        if self.kind == "finish" and (
            self.output is None or self.tool_name or self.arguments
        ):
            raise ValueError("Finish needs an output and no tool arguments")
        return self


class ResearchDecision(Model):
    """Single-agent planner chooses evidence or readiness, never writes a report."""

    kind: Literal["tool", "finish"]
    reason: str = Field(min_length=1, max_length=500)
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_action(self) -> "ResearchDecision":
        if self.kind == "tool" and not self.tool_name:
            raise ValueError("Tool action needs a name")
        if self.kind == "finish" and (self.tool_name or self.arguments):
            raise ValueError("Finish signals readiness without tool arguments")
        return self


class ToolObservation(Model):
    evidence_id: str
    role: Role
    tool_name: str
    arguments: dict[str, Any]
    output: Any = None
    success: bool
    error: str | None = None
    cache_hit: bool = False


class TraceEvent(Model):
    timestamp: str
    role: str
    event: str
    content: dict[str, Any]


class QuantMetric(Model):
    name: MetricName
    value: float | None = Field(allow_inf_nan=False, strict=True)
    evidence_id: str
    path: str


class QuantitativeBrief(Model):
    ticker: str
    summary: str = Field(min_length=1, max_length=700)
    metrics: list[QuantMetric]
    limitations: list[str] = Field(min_length=1)


class ClarificationRequest(Model):
    question: str = Field(min_length=1, max_length=300)
    requested_metric: MetricName
    reason: str = Field(min_length=1, max_length=300)


class ClarificationResponse(Model):
    question: str
    answer: str = Field(min_length=1, max_length=500)
    metric: QuantMetric | None
    limitations: list[str] = Field(min_length=1)


class SharePriceRisk(Model):
    title: str = Field(min_length=1, max_length=100)
    supporting_evidence: str = Field(min_length=1, max_length=400)
    evidence_ids: list[str] = Field(min_length=1)


class HedgeStrategy(Model):
    strategy: str = Field(min_length=1, max_length=300)
    data_driven_rationale: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(min_length=1)
    limitations: str = Field(min_length=1, max_length=400)


class ResearchReport(Model):
    ticker: str
    as_of: date
    horizon_days: Literal[90]
    financial_health_summary: str = Field(min_length=1, max_length=800)
    market_sentiment_summary: str = Field(min_length=1, max_length=600)
    top_three_risks: list[SharePriceRisk] = Field(min_length=3, max_length=3)
    hedge_strategy_recommendation: HedgeStrategy
    limitations: list[str] = Field(min_length=1)
    clarification_used: str | None = None


class WriterResearchReport(ResearchReport):
    """Writer-only contract; single-agent research has no clarification handoff."""

    clarification_used: str = Field(
        min_length=1,
        max_length=500,
        description="Required: copy handoff_context.clarification.answer verbatim. "
        "Citing analyst_clarification alone does not satisfy this field.",
    )


class FollowupAnswer(Model):
    answer: str = Field(min_length=1, max_length=1000)
    evidence_ids: list[str]


class ResearchRun(Model):
    ticker: str
    as_of: date
    workflow: Literal["single", "multi"]
    report: ResearchReport | None = None
    brief: QuantitativeBrief | None = None
    critique_request: ClarificationRequest | None = None
    clarification: ClarificationResponse | None = None
    observations: list[ToolObservation] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    error: str | None = None
    cached: bool = False
