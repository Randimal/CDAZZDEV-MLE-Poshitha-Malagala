"""Small result models and recoverable domain errors."""

from dataclasses import dataclass
from typing import Literal

import pandas as pd

Signal = Literal["BULLISH", "BEARISH", "NEUTRAL"]


class PipelineError(Exception):
    """Expected financial data pipeline failure, safe to report to a user."""


class DataFetchError(PipelineError):
    """Provider request failed or returned no prices."""


class InvalidHistoryError(PipelineError):
    """OHLCV data is malformed or insufficient."""


@dataclass(frozen=True)
class NewsHeadline:
    title: str
    publisher: str | None = None
    published_at: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class MomentumResult:
    signal: Signal
    score: int
    reasons: list[str]


@dataclass
class PipelineResult:
    ticker: str
    data: pd.DataFrame
    news: list[NewsHeadline]
    summary: dict[str, str | float | None]
    momentum: MomentumResult
