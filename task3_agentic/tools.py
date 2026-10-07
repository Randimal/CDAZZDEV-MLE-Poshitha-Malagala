"""Five callable tools, shared Task 1 services and enforced role permissions."""

import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import date
from typing import Any

import numpy as np
import yfinance as yf
from ddgs import DDGS
from pydantic import BaseModel, ConfigDict, Field

from shared.config import PipelineConfig
from task1_financial.data_pipeline import run_pipeline
from task1_financial.json_utils import json_safe
from task1_financial.llm import CompletionClient
from task1_financial.models import NewsHeadline, PipelineResult
from task1_financial.news import fetch_news
from task3_agentic.schemas import Role, ToolObservation
from task3_agentic.sentiment import TASK3_HEADLINE_LIMIT, analyze_batch
from task3_agentic.tracing import ToolTracer, redact

TRADING_DAYS_PER_YEAR = 252
HEDGE_HORIZON_TRADING_DAYS = 90


def horizon_volatility(annualized: float) -> float:
    """Historical 90-trading-day 1-sigma scale, not a forecast or option price."""
    return float(
        annualized * np.sqrt(HEDGE_HORIZON_TRADING_DAYS / TRADING_DAYS_PER_YEAR)
    )


ROLE_TOOLS: dict[str, frozenset[str]] = {
    "researcher": frozenset(
        {
            "get_price_data",
            "get_news",
            "calculate_volatility",
            "llm_sentiment",
            "web_search",
        }
    ),
    "analyst": frozenset({"get_price_data", "calculate_volatility", "llm_sentiment"}),
    "writer": frozenset({"get_news", "web_search"}),
}


def normalize_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.^=_-]{1,24}", ticker):
        raise ValueError("Invalid ticker format")
    return ticker


def _pipeline(config: PipelineConfig) -> PipelineResult:
    return run_pipeline(config, include_news=False)


def _search(query: str) -> list[dict[str, Any]]:
    return list(DDGS(timeout=10).text(query, max_results=5, backend="duckduckgo"))


class FinancialTools:
    """Dependencies are injectable; no credentials are accepted as tool arguments."""

    def __init__(
        self,
        ticker: str,
        client: CompletionClient,
        *,
        pipeline: Callable[[PipelineConfig], PipelineResult] = _pipeline,
        news_client_factory: Callable[[str], Any] = yf.Ticker,
        search: Callable[[str], list[dict[str, Any]]] = _search,
        seed_headlines: list[NewsHeadline] | None = None,
    ) -> None:
        self.ticker = normalize_ticker(ticker)
        self.client = client
        self.pipeline = pipeline
        self.news_client_factory = news_client_factory
        self.search = search
        self._prices: dict[tuple[str, str], PipelineResult] = {}
        self.known_headlines = {item.title: item for item in (seed_headlines or [])}
        self._seed_headlines = dict(self.known_headlines)

    def reset_session(self) -> None:
        self._prices.clear()
        self.known_headlines = dict(self._seed_headlines)

    def _history(self, ticker: str, period: str = "2y") -> PipelineResult:
        ticker = normalize_ticker(ticker)
        key = (ticker, period)
        if key not in self._prices:
            self._prices[key] = self.pipeline(
                PipelineConfig(ticker=ticker, period=period)
            )
        return self._prices[key]

    def get_price_data(self, ticker: str, period: str = "2y") -> dict[str, Any]:
        result = self._history(ticker, period)
        data = result.data.copy()
        data.insert(0, "date", data.index)
        return json_safe(
            {
                "ticker": result.ticker,
                "period": period,
                "price_basis": "adjusted daily OHLC",
                "summary": result.summary,
                "latest_indicators": result.data.iloc[-1].to_dict(),
                "rows": data.to_dict(orient="records"),
            }
        )

    def get_news(self, ticker: str, n: int = TASK3_HEADLINE_LIMIT) -> dict[str, Any]:
        if not 1 <= n <= TASK3_HEADLINE_LIMIT:
            raise ValueError("Task 3 news requests must contain 1–10 headlines")
        ticker = normalize_ticker(ticker)
        items = fetch_news(
            self.news_client_factory(ticker), count=TASK3_HEADLINE_LIMIT, ticker=ticker
        )[:n]
        self.known_headlines.update({item.title: item for item in items})
        return {"ticker": ticker, "headlines": [asdict(item) for item in items]}

    def calculate_volatility(self, ticker: str, window: int = 60) -> dict[str, Any]:
        """Sample std of latest n simple daily returns * sqrt(252).

        r_t = Close_t/Close_(t-1)-1, no fill; ddof=1. The result is an
        annualized fraction, not a percentage or a forward forecast.
        """
        if not 2 <= window <= TRADING_DAYS_PER_YEAR:
            raise ValueError("Volatility window must be 2–252 daily returns")
        result = self._history(ticker)
        returns = result.data["Close"].pct_change(fill_method=None).iloc[-window:]
        if len(returns) != window or not np.isfinite(returns).all():
            raise ValueError("Insufficient consecutive daily returns")
        volatility = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        return {
            "ticker": result.ticker,
            "window": window,
            "annualized_volatility": volatility,
            "horizon_trading_days": HEDGE_HORIZON_TRADING_DAYS,
            "horizon_volatility": horizon_volatility(volatility),
            "horizon_formula": "annualized_volatility * sqrt(90 / 252)",
            "unit": "fraction",
            "observations": len(returns),
            "as_of": result.data.index[-1].isoformat(),
            "formula": "sample_std(simple_daily_returns, ddof=1) * sqrt(252)",
        }

    def llm_sentiment(self, headlines: list[dict[str, Any]]) -> dict[str, Any]:
        if not 1 <= len(headlines) <= TASK3_HEADLINE_LIMIT:
            raise ValueError("Task 3 sentiment requires 1–10 retrieved headlines")
        titles = [item.get("title") for item in headlines]
        if not titles or any(title not in self.known_headlines for title in titles):
            raise ValueError("Use only supplied or previously retrieved headlines")
        items = [self.known_headlines[title] for title in dict.fromkeys(titles)]
        return analyze_batch(self.ticker, items, self.client).model_dump()

    def web_search(self, query: str) -> dict[str, Any]:
        results = []
        for item in self.search(query):
            if not isinstance(item, dict):
                continue
            title, url = item.get("title"), item.get("href", item.get("url"))
            if (
                isinstance(title, str)
                and title.strip()
                and isinstance(url, str)
                and url.startswith(("https://", "http://"))
            ):
                results.append(
                    {
                        "title": title,
                        "url": url,
                        "snippet": str(item.get("body", item.get("snippet", "")))[
                            :1500
                        ],
                    }
                )
        return {"query": query, "results": results[:5]}


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PriceArguments(Arguments):
    ticker: str
    period: str = "2y"


class NewsArguments(Arguments):
    ticker: str
    n: int = Field(default=TASK3_HEADLINE_LIMIT, ge=1, le=TASK3_HEADLINE_LIMIT)


class VolatilityArguments(Arguments):
    ticker: str
    window: int = Field(default=60, ge=2, le=252)


class SentimentArguments(Arguments):
    headlines: list[dict[str, Any]] = Field(
        min_length=1, max_length=TASK3_HEADLINE_LIMIT
    )


class SearchArguments(Arguments):
    query: str = Field(min_length=1, max_length=500)


ARGUMENT_SCHEMAS = {
    "get_price_data": PriceArguments,
    "get_news": NewsArguments,
    "calculate_volatility": VolatilityArguments,
    "llm_sentiment": SentimentArguments,
    "web_search": SearchArguments,
}


def tool_call_key(name: str, arguments: dict[str, Any]) -> str:
    """Canonicalize defaults, ticker case and JSON ordering for failure tracking."""
    try:
        args = ARGUMENT_SCHEMAS[name].model_validate(arguments).model_dump()
    except (KeyError, ValueError):
        args = dict(arguments)
    if isinstance(args.get("ticker"), str):
        args["ticker"] = args["ticker"].strip().upper()
    return json.dumps({"name": name, "arguments": args}, sort_keys=True)


def empty_result(tool: str, output: dict[str, Any]) -> bool:
    key = {
        "get_news": "headlines",
        "web_search": "results",
        "get_price_data": "rows",
    }.get(tool)
    if key:
        return not output.get(key)
    if tool == "llm_sentiment":
        return output["aggregate"]["successful_count"] == 0
    return False


class ToolExecutor:
    """Enforce permissions before lookup, cache successes and trace every call."""

    def __init__(self, tools: FinancialTools, tracer: ToolTracer | None = None) -> None:
        self.tools = tools
        self.tracer = tracer or ToolTracer()
        self.cache: dict[str, Any] = {}
        self.counter = 0
        self.session_date: date | None = None

    def start_session(self, as_of: date, *, refresh: bool = False) -> None:
        """Avoid carrying provider snapshots into a new date or forced refresh."""
        if refresh or self.session_date != as_of:
            self.cache.clear()
            self.tools.reset_session()
            self.session_date = as_of

    def invoke(
        self, role: Role, name: str, arguments: dict[str, Any]
    ) -> ToolObservation:
        started = time.perf_counter()
        self.counter += 1
        output, error, cached = None, None, False
        try:
            if name not in ROLE_TOOLS[role]:
                raise PermissionError("Tool not allowed for role")
            args = ARGUMENT_SCHEMAS[name].model_validate(arguments).model_dump()
            if (
                "ticker" in args
                and normalize_ticker(args["ticker"]) != self.tools.ticker
            ):
                raise ValueError("Tool ticker must match research ticker")
            key = json.dumps({"name": name, "arguments": args}, sort_keys=True)
            if key in self.cache:
                output, cached = self.cache[key], True
            else:
                output = redact(getattr(self.tools, name)(**args))
                if empty_result(name, output):
                    raise ValueError("No usable results; try another approach")
                self.cache[key] = output
        except Exception as exc:
            # Provider errors may contain credentials; never retain raw text.
            error = f"{type(exc).__name__}: tool unavailable, empty, invalid, or not permitted"
        success = error is None
        self.tracer.record(
            role=role,
            tool_name=name,
            arguments=arguments,
            output=output if success else {"error": error},
            duration_ms=(time.perf_counter() - started) * 1000,
            success=success,
            cache_hit=cached,
        )
        return ToolObservation(
            evidence_id=f"obs-{self.counter}",
            role=role,
            tool_name=name,
            arguments=redact(arguments),
            output=output if success else None,
            success=success,
            error=error,
            cache_hit=cached,
        )
