import numpy as np
import pandas as pd
import pytest

from task1_financial.indicators import add_indicators, rsi


def test_sma_windows_and_warmup(prices: pd.DataFrame) -> None:
    result = add_indicators(prices)
    for window in (50, 200):
        values = result[f"SMA_{window}"]
        assert values.iloc[: window - 1].isna().all()
        assert values.iloc[window - 1] == pytest.approx(
            prices.Close.iloc[:window].mean()
        )
        assert values.iloc[-1] == pytest.approx(prices.Close.iloc[-window:].mean())


@pytest.mark.parametrize("step,expected", [(1, 100), (-1, 0), (0, 50)])
def test_rsi_edges(step: int, expected: int) -> None:
    result = rsi(pd.Series(100 + step * np.arange(60)))
    assert result.iloc[:14].isna().all()
    np.testing.assert_allclose(result.iloc[14:], expected)


def test_rsi_recurrence_and_range() -> None:
    close = pd.Series([100, 102, 101, 104, 102, 106, 105], dtype=float)
    result = rsi(close, period=3)
    gain, loss = 5 / 3, 1 / 3
    assert result.iloc[3] == pytest.approx(100 * gain / (gain + loss))
    gain, loss = gain * 2 / 3, loss * 2 / 3 + 2 / 3
    assert result.iloc[4] == pytest.approx(100 * gain / (gain + loss))
    assert result.dropna().between(0, 100).all()


def test_macd_recurrence(prices: pd.DataFrame) -> None:
    def recursive_ema(values: np.ndarray, span: int) -> np.ndarray:
        out = np.empty(len(values))
        out[0] = values[0]
        for i in range(1, len(values)):
            out[i] = 2 / (span + 1) * values[i] + (1 - 2 / (span + 1)) * out[i - 1]
        return out

    close = prices.Close.to_numpy()
    expected = recursive_ema(close, 12) - recursive_ema(close, 26)
    result = add_indicators(prices)
    assert result.MACD.iloc[:25].isna().all()
    np.testing.assert_allclose(result.MACD.iloc[25:], expected[25:])
    signal = recursive_ema(expected[25:], 9)
    assert result.MACD_signal.iloc[:33].isna().all()
    np.testing.assert_allclose(result.MACD_signal.iloc[33:], signal[8:])
    np.testing.assert_allclose(
        result.MACD_histogram.iloc[33:], expected[33:] - signal[8:]
    )


def test_bands_and_missing_data(prices: pd.DataFrame) -> None:
    result = add_indicators(prices)
    bands = result.dropna(subset=["BB_middle"])
    assert (bands.BB_upper >= bands.BB_middle).all()
    assert (bands.BB_middle >= bands.BB_lower).all()
    last = prices.Close.iloc[-20:].to_numpy()
    assert result.BB_upper.iloc[-1] == pytest.approx(last.mean() + 2 * last.std(ddof=0))
    prices.loc[prices.index[250], "Close"] = np.nan
    result = add_indicators(prices)
    assert result.RSI.iloc[250:265].isna().all()
    assert result.MACD.iloc[250:276].isna().all()
    assert result.SMA_50.iloc[250:300].isna().all()
    assert np.isfinite(result.RSI.iloc[265])
