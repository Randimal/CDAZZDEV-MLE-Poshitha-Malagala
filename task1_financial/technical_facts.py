"""Compact deterministic relationships over already-computed daily indicators."""

import math
from collections.abc import Mapping

# Conventional RSI extremes; a five-point band around 50 avoids describing
# a small oscillator change as strong/weak. Regimes are descriptive heuristics.
RSI_OVERSOLD = 30.0
RSI_NEUTRAL_LOWER = 45.0
RSI_NEUTRAL_UPPER = 55.0
RSI_OVERBOUGHT = 70.0


def _number(value: object, *, positive: bool = False) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and (not positive or number > 0) else None


def _relative(left: float | None, right: float | None) -> str | None:
    if left is None or right is None:
        return None
    return "above" if left > right else "below" if left < right else "equal"


def _distance(price: float | None, average: float | None) -> float | None:
    """100 * (price / average - 1); invalid denominators remain unavailable."""
    if price is None or average is None:
        return None
    distance = (price / average - 1) * 100
    return distance if math.isfinite(distance) else None


def _rsi_regime(value: float | None) -> str | None:
    if value is None or not 0 <= value <= 100:
        return None
    if value <= RSI_OVERSOLD:
        return "oversold"
    if value < RSI_NEUTRAL_LOWER:
        return "weak"
    if value <= RSI_NEUTRAL_UPPER:
        return "neutral"
    if value < RSI_OVERBOUGHT:
        return "strong"
    return "overbought"


def _band_location(price: float | None, latest: Mapping[str, object]) -> str | None:
    lower, middle, upper = (
        _number(latest.get(key)) for key in ("BB_lower", "BB_middle", "BB_upper")
    )
    if price is None or lower is None or middle is None or upper is None:
        return None
    if not lower <= middle <= upper:
        return None
    if price < lower:
        return "below_lower"
    if price > upper:
        return "above_upper"
    if price == middle:
        return "middle"
    return "lower_half" if price < middle else "upper_half"


def build_technical_facts(
    latest: Mapping[str, object], *, momentum_signal: str | None = None
) -> dict[str, str | float | None]:
    """Describe latest indicator interactions, without predicting or finding patterns.

    Bullish structure means Close > SMA50 > SMA200; bearish reverses that
    ordering. Other fully available orderings are neutral/mixed. Missing,
    nonfinite or invalid inputs yield None, never an invented neutral reading.
    Percent distances use the relevant SMA as denominator. Equal MACD or zero
    histogram remains explicitly equal/zero; band boundaries belong to halves.
    RSI regimes are a policy, not calibrated reversal probabilities.
    """
    price, short, long = (
        _number(latest.get(key), positive=True)
        for key in ("Close", "SMA_50", "SMA_200")
    )
    trend = None
    if price is not None and short is not None and long is not None:
        trend = (
            "bullish"
            if price > short > long
            else "bearish"
            if price < short < long
            else "neutral"
        )
    macd, signal = (_number(latest.get(key)) for key in ("MACD", "MACD_signal"))
    histogram = _number(latest.get("MACD_histogram"))
    histogram_sign = (
        None
        if histogram is None
        else "positive"
        if histogram > 0
        else "negative"
        if histogram < 0
        else "zero"
    )
    return {
        "price_vs_sma50_pct": _distance(price, short),
        "price_vs_sma200_pct": _distance(price, long),
        "sma50_vs_sma200": _relative(short, long),
        "trend_structure": trend,
        "rsi_regime": _rsi_regime(_number(latest.get("RSI"))),
        "macd_vs_signal": _relative(macd, signal),
        "macd_histogram_sign": histogram_sign,
        "bollinger_location": _band_location(price, latest),
        "deterministic_momentum": momentum_signal
        if momentum_signal in {"BULLISH", "BEARISH", "NEUTRAL"}
        else None,
    }
