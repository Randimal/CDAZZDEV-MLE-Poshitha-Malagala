import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def offline_news_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy pipeline tests remain offline when Yahoo triggers the new fallback."""
    monkeypatch.setattr("task1_financial.news.fetch_rss_news", lambda *args: [])


@pytest.fixture
def prices() -> pd.DataFrame:
    close = np.linspace(100, 250, 600)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": np.full(600, 1000),
        },
        index=pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=600),
    )
