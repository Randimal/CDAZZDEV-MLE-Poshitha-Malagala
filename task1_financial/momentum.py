"""Transparent multi-indicator directional score; no learned component."""

import math
from collections.abc import Mapping

from task1_financial.models import MomentumResult

RSI_MIDPOINT = 50.0
# Three of five votes is an absolute majority, even if others abstain.
MAJORITY_SCORE = 3


def classify_momentum(latest: Mapping[str, object]) -> MomentumResult:
    """Five equal votes: price/SMA50, SMA50/SMA200, RSI/50,
    MACD/signal and price/BB middle. Equality or missing data abstains.
    A net score >=3 or <=-3 gives a directional signal; else neutral.
    """
    pairs = [
        ("Close", "SMA_50"),
        ("SMA_50", "SMA_200"),
        ("RSI", "RSI midpoint"),
        ("MACD", "MACD_signal"),
        ("Close", "BB_middle"),
    ]
    score = 0
    reasons: list[str] = []
    for left, right in pairs:
        try:
            a = float(latest.get(left))
            b = RSI_MIDPOINT if right == "RSI midpoint" else float(latest.get(right))
            if not (math.isfinite(a) and math.isfinite(b)):
                raise ValueError("Nonfinite indicator")
        except (TypeError, ValueError):
            reasons.append(f"{left} vs {right}: unavailable (0)")
            continue
        vote = int(a > b) - int(a < b)
        score += vote
        reasons.append(
            f"{left} {'above' if vote > 0 else 'below' if vote < 0 else 'equals'} {right} ({vote:+d})"
        )
    signal = (
        "BULLISH"
        if score >= MAJORITY_SCORE
        else "BEARISH"
        if score <= -MAJORITY_SCORE
        else "NEUTRAL"
    )
    return MomentumResult(signal, score, reasons)
