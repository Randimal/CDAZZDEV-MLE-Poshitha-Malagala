import pytest

from task1_financial.momentum import classify_momentum


@pytest.mark.parametrize(
    "row,signal,score",
    [
        (
            {
                "Close": 120,
                "SMA_50": 110,
                "SMA_200": 100,
                "RSI": 60,
                "MACD": 2,
                "MACD_signal": 1,
                "BB_middle": 115,
            },
            "BULLISH",
            5,
        ),
        (
            {
                "Close": 80,
                "SMA_50": 90,
                "SMA_200": 100,
                "RSI": 40,
                "MACD": -2,
                "MACD_signal": -1,
                "BB_middle": 85,
            },
            "BEARISH",
            -5,
        ),
        (
            {
                "Close": 100,
                "SMA_50": 100,
                "SMA_200": 100,
                "RSI": 50,
                "MACD": 0,
                "MACD_signal": 0,
                "BB_middle": 100,
            },
            "NEUTRAL",
            0,
        ),
        ({}, "NEUTRAL", 0),
        ({"Close": None, "RSI": float("nan")}, "NEUTRAL", 0),
        (
            {
                "Close": 120,
                "SMA_50": 110,
                "SMA_200": 100,
                "RSI": 40,
                "MACD": 0,
                "MACD_signal": 1,
                "BB_middle": 100,
            },
            "NEUTRAL",
            1,
        ),
    ],
)
def test_votes(row: dict[str, object], signal: str, score: int) -> None:
    result = classify_momentum(row)
    assert (result.signal, result.score) == (signal, score)
    assert len(result.reasons) == 5
