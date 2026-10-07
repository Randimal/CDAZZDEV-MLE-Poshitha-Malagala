"""Explicit, immutable Task 1A settings."""

from dataclasses import dataclass

OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True)
class IndicatorConfig:
    sma_short: int = 50
    sma_long: int = 200
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0

    def __post_init__(self) -> None:
        windows = (
            self.sma_short,
            self.sma_long,
            self.rsi_period,
            self.macd_fast,
            self.macd_slow,
            self.macd_signal,
            self.bb_window,
        )
        if any(not isinstance(w, int) or isinstance(w, bool) or w < 1 for w in windows):
            raise ValueError("Indicator windows must be positive integers")
        if self.macd_fast >= self.macd_slow or self.bb_std <= 0:
            raise ValueError("Invalid MACD spans or Bollinger multiplier")


@dataclass(frozen=True)
class PipelineConfig:
    ticker: str = "NVDA"
    period: str = "2y"
    news_count: int = 10
    timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        if not self.ticker.strip() or self.news_count < 10:
            raise ValueError(
                "A ticker and at least 10 requested headlines are required"
            )
        if self.period not in {"2y", "5y", "10y", "max"}:
            raise ValueError("Use a history period of at least two years")
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")


INDICATORS = IndicatorConfig()
