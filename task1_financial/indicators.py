"""Technical indicators implemented only with pandas/numpy."""

import numpy as np
import pandas as pd

from shared.config import INDICATORS, IndicatorConfig


def rsi(close: pd.Series, period: int = INDICATORS.rsi_period) -> pd.Series:
    """Wilder RSI: arithmetic seed of n deltas, then EMA alpha=1/n.

    Gain/loss averages recurse with adjust=False. No losses -> 100;
    no gains -> 0; a flat series -> 50 (neutral). Missing prices reset
    the seed, requiring n fresh consecutive changes before another value.
    """
    if period < 1:
        raise ValueError("RSI period must be positive")
    values = pd.to_numeric(close, errors="coerce").astype(float)
    values = values.where(np.isfinite(values))
    result = pd.Series(np.nan, index=close.index, dtype=float, name="RSI")
    groups = values.notna().ne(values.notna().shift()).cumsum()
    for _, segment in values.groupby(groups):
        if segment.isna().any() or len(segment) <= period:
            continue
        delta = segment.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        gain.iloc[period] = gain.iloc[1 : period + 1].mean()
        loss.iloc[period] = loss.iloc[1 : period + 1].mean()
        avg_gain = gain.iloc[period:].ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.iloc[period:].ewm(alpha=1 / period, adjust=False).mean()
        strength = 100 - 100 / (1 + avg_gain / avg_loss)
        strength = strength.mask((avg_loss == 0) & (avg_gain > 0), 100)
        strength = strength.mask((avg_loss == 0) & (avg_gain == 0), 50)
        result.loc[strength.index] = strength
    return result


def add_indicators(
    data: pd.DataFrame,
    config: IndicatorConfig = INDICATORS,
) -> pd.DataFrame:
    """Return a copy with full-window SMAs, EMA MACD, RSI and bands.

    EMA_t = alpha*x_t + (1-alpha)*EMA_(t-1), alpha=2/(span+1),
    seeded by the first price (adjust=False). MACD is fast minus slow;
    signal is an EMA of defined MACD values. Bands use population std
    (ddof=0): rolling mean +/- multiplier*std. Warm-up values stay NaN.
    Missing closes reset EMA state; rolling windows require all values.
    """
    output = data.copy()
    close = pd.to_numeric(output["Close"], errors="coerce").astype(float)
    close = close.where(np.isfinite(close))
    output["Close"] = close
    output["SMA_50"] = close.rolling(
        config.sma_short, min_periods=config.sma_short
    ).mean()
    output["SMA_200"] = close.rolling(
        config.sma_long, min_periods=config.sma_long
    ).mean()
    output["RSI"] = rsi(close, config.rsi_period)
    groups = close.isna().cumsum()

    def ema(series: pd.Series, span: int) -> pd.Series:
        return (
            series.groupby(groups)
            .transform(
                lambda part: part.ewm(span=span, adjust=False, min_periods=span).mean()
            )
            .where(series.notna())
        )

    output["MACD"] = ema(close, config.macd_fast) - ema(close, config.macd_slow)
    output["MACD_signal"] = ema(output["MACD"], config.macd_signal)
    output["MACD_histogram"] = output["MACD"] - output["MACD_signal"]
    middle = close.rolling(config.bb_window, min_periods=config.bb_window).mean()
    deviation = close.rolling(config.bb_window, min_periods=config.bb_window).std(
        ddof=0
    )
    output["BB_middle"] = middle
    output["BB_upper"] = middle + config.bb_std * deviation
    output["BB_lower"] = middle - config.bb_std * deviation
    return output
