"""Daily OHLCV ingestion, validation, features and price-derived summary."""

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from shared.config import INDICATORS, OHLCV_COLUMNS, PipelineConfig
from task1_financial.indicators import add_indicators
from task1_financial.models import (
    DataFetchError,
    InvalidHistoryError,
    MomentumResult,
    PipelineResult,
)
from task1_financial.momentum import classify_momentum
from task1_financial.news import fetch_news

logger = logging.getLogger(__name__)


def validate_history(raw: pd.DataFrame) -> pd.DataFrame:
    """Sort/deduplicate dates and reject incomplete or invalid OHLCV rows.

    Never fill market prices. At least 200 consecutive valid final rows
    are required so the latest SMA200 is defined. Earlier gaps stay NaN.
    """
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise DataFetchError(
            "No price history returned; check ticker or provider availability"
        )
    if not isinstance(raw.index, pd.DatetimeIndex) or raw.index.hasnans:
        raise InvalidHistoryError("History must have a valid DatetimeIndex")
    if any(col not in raw.columns for col in OHLCV_COLUMNS):
        raise InvalidHistoryError("History is missing required OHLCV columns")
    data = raw.loc[:, list(OHLCV_COLUMNS)].copy().sort_index()
    data = data.loc[~data.index.duplicated(keep="last")]
    data = data.apply(pd.to_numeric, errors="coerce").astype(float)
    valid = np.isfinite(data).all(axis=1)
    valid &= (data[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
    valid &= data["Volume"] >= 0
    valid &= data["High"] >= data[["Open", "Low", "Close"]].max(axis=1)
    valid &= data["Low"] <= data[["Open", "High", "Close"]].min(axis=1)
    if not valid.all():
        logger.warning("%d invalid OHLCV rows retained as missing", (~valid).sum())
        data.loc[~valid, :] = np.nan
    if len(data) < INDICATORS.sma_long or not valid.iloc[-INDICATORS.sma_long :].all():
        raise InvalidHistoryError(
            "Need at least 200 consecutive valid latest daily OHLCV observations"
        )
    return data


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def build_summary(
    ticker: str,
    data: pd.DataFrame,
    metadata: Mapping[str, Any] | None,
    momentum: MomentumResult,
    as_of: pd.Timestamp | None = None,
) -> dict[str, str | float | None]:
    """Price-derived trailing 52-week extrema and calendar YTD return.

    YTD uses the last valid close before Jan 1 of the as-of year. If
    either baseline or current-year prices are missing, return null.
    Default as-of is today in the history's exchange timezone.
    """
    now = (
        pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz=data.index.tz)
    )
    if data.index.tz is not None:
        now = (
            now.tz_localize(data.index.tz)
            if now.tzinfo is None
            else now.tz_convert(data.index.tz)
        )
    elif now.tzinfo is not None:
        now = now.tz_localize(None)
    usable = data.loc[data.index <= now].dropna(subset=["Close"])
    if usable.empty:
        raise InvalidHistoryError("No valid prices on or before summary date")
    price = float(usable["Close"].iloc[-1])
    recent = usable.loc[usable.index >= now - pd.Timedelta(weeks=52)]
    start = pd.Timestamp(year=now.year, month=1, day=1, tz=data.index.tz)
    baseline = usable.loc[usable.index < start, "Close"]
    this_year = usable.loc[usable.index >= start, "Close"]
    ytd = None
    if not baseline.empty and not this_year.empty and baseline.iloc[-1] > 0:
        ytd = (price / float(baseline.iloc[-1]) - 1) * 100
    else:
        logger.warning(
            "YTD return unavailable: missing prior-year baseline or current-year prices"
        )
    return {
        "ticker": ticker,
        "current_price": price,
        "52_week_high": _finite_number(recent["High"].max()),
        "52_week_low": _finite_number(recent["Low"].min()),
        "pe_ratio": _finite_number((metadata or {}).get("trailingPE")),
        "ytd_return_pct": ytd,
        "momentum_signal": momentum.signal,
    }


def run_pipeline(
    config: PipelineConfig | None = None,
    *,
    client: Any = None,
    include_news: bool = True,
) -> PipelineResult:
    """Fetch daily adjusted prices; essential failures raise PipelineError.

    Metadata and news are optional and log warnings on failure. Callers
    (including the notebook) should catch PipelineError for user feedback.
    A supplied client allows deterministic tests without Yahoo requests.
    """
    config = config or PipelineConfig()
    ticker = config.ticker.strip().upper()
    try:
        client = client if client is not None else yf.Ticker(ticker)
        raw = client.history(
            period=config.period,
            interval="1d",
            auto_adjust=True,
            actions=False,
            timeout=config.timeout_seconds,
        )
    except Exception as exc:
        raise DataFetchError(
            f"Unable to fetch daily history for {ticker}: {exc}"
        ) from exc
    data = add_indicators(validate_history(raw))
    momentum = classify_momentum(data.iloc[-1].to_dict())
    try:
        metadata = client.info
        if not isinstance(metadata, dict):
            metadata = {}
    except Exception as exc:
        logger.warning("Metadata unavailable for %s: %s", ticker, exc, exc_info=True)
        metadata = {}
    summary = build_summary(ticker, data, metadata, momentum)
    return PipelineResult(
        ticker,
        data,
        fetch_news(client, config.news_count, ticker=ticker) if include_news else [],
        summary,
        momentum,
    )
