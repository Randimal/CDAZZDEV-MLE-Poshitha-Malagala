"""Mocked Yahoo/RSS shortfall recovery; no live news requests."""

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import Mock, patch

import pytest

from task1_financial.models import NewsHeadline
from task1_financial.news import fetch_news, fetch_rss_news


@pytest.mark.parametrize("failure", [True, False])
def test_primary_failure_or_shortfall_uses_fallback(failure: bool, caplog) -> None:
    client = Mock()
    if failure:
        client.get_news.side_effect = ConnectionError("secret raw error")
    else:
        client.get_news.return_value = [{"title": "Primary headline"}]
    with patch("task1_financial.news.fetch_rss_news") as rss:
        rss.return_value = [
            NewsHeadline("Primary headline"),
            NewsHeadline(" PRIMARY   headline "),
            NewsHeadline("Fallback headline", "Wire", url="https://example.com/news"),
        ]
        with caplog.at_level("INFO"):
            results = fetch_news(client, count=2, ticker="NVDA")
    assert [item.title for item in results] == ["Primary headline", "Fallback headline"]
    rss.assert_called_once_with("NVDA", 2)
    assert "google_news_rss" in caplog.text
    assert "secret raw error" not in caplog.text


def test_full_primary_skips_rss() -> None:
    client = Mock()
    client.get_news.return_value = [{"title": "Actual headline"}]
    with patch("task1_financial.news.fetch_rss_news") as rss:
        assert len(fetch_news(client, count=1, ticker="NVDA")) == 1
    rss.assert_not_called()


def test_rss_parser_keeps_recent_real_metadata() -> None:
    now = datetime.now(timezone.utc)
    recent = format_datetime(now - timedelta(hours=1))
    stale = format_datetime(now - timedelta(days=30))
    payload = f"""<rss><channel>
    <item><title>Actual news - Wire</title><source>Wire</source>
    <pubDate>{recent}</pubDate><link>https://example.com/news</link></item>
    <item><title> ACTUAL   news - Wire</title><source>Wire</source>
    <pubDate>{recent}</pubDate><link>https://example.com/duplicate</link></item>
    <item><title>Second real headline</title><source>Desk</source>
    <pubDate>{recent}</pubDate><link>https://example.com/second</link></item>
    <item><title>Old news</title><pubDate>{stale}</pubDate>
    <link>https://example.com/old</link></item>
    <item><title>Bad date</title><pubDate>invalid</pubDate></item>
    </channel></rss>""".encode()
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = payload
    with patch("task1_financial.news.urlopen", return_value=response) as request:
        items = fetch_rss_news("NVDA", count=2)
    assert len(items) == 2 and items[1].title == "Second real headline"
    assert items[0].title == "Actual news" and items[0].publisher == "Wire"
    assert items[0].published_at and items[0].url == "https://example.com/news"
    assert "when%3A7d" in request.call_args.args[0].full_url
    assert request.call_args.kwargs == {"timeout": 10}


def test_fallback_failure_preserves_primary(caplog) -> None:
    client = Mock()
    client.get_news.return_value = [{"title": "Primary"}]
    with patch("task1_financial.news.fetch_rss_news", side_effect=OSError("private")):
        assert fetch_news(client, 10, ticker="NVDA") == [NewsHeadline("Primary")]
    assert "private" not in caplog.text
