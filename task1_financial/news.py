"""Defensive normalization of old and nested yfinance news payloads."""

import logging
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from task1_financial.models import NewsHeadline

logger = logging.getLogger(__name__)
NEWS_LOOKBACK_DAYS = 7
RSS_TIMEOUT_SECONDS = 10
MAX_RSS_BYTES = 1_000_000


def fetch_rss_news(ticker: str, count: int = 10) -> list[NewsHeadline]:
    """Read recent Google News RSS search results with normal HTTPS verification.

    Search matches are headlines, not verified company facts. Retain actual
    publisher/date/link fields and skip undated, stale or malformed items.
    """
    query = urlencode(
        {
            "q": f'"{ticker}" stock when:{NEWS_LOOKBACK_DAYS}d',
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        }
    )
    request = Request(
        f"https://news.google.com/rss/search?{query}",
        headers={"User-Agent": "CDAZZDEV-financial-research/1.0"},
    )
    with urlopen(request, timeout=RSS_TIMEOUT_SECONDS) as response:
        payload = response.read(MAX_RSS_BYTES + 1)
    if len(payload) > MAX_RSS_BYTES or any(
        token in payload.upper() for token in (b"<!DOCTYPE", b"<!ENTITY")
    ):
        raise ValueError("Unsupported RSS payload")
    now = datetime.now(timezone.utc)
    items = []
    for item in ElementTree.fromstring(payload).findall("./channel/item"):
        title, url = _text(item.findtext("title")), _text(item.findtext("link"))
        try:
            published = parsedate_to_datetime(item.findtext("pubDate", ""))
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            published = published.astimezone(timezone.utc)
        except (TypeError, ValueError, OverflowError):
            continue
        if (
            not title
            or not url
            or not url.startswith("https://")
            or not now - timedelta(days=NEWS_LOOKBACK_DAYS) <= published <= now
        ):
            continue
        publisher = _text(item.findtext("source"))
        # Google appends the publisher to the RSS title; normalize for deduplication.
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(f" - {publisher}")].strip()
        if title:
            items.append(NewsHeadline(title, publisher, published.isoformat(), url))
    items.sort(key=lambda item: item.published_at or "", reverse=True)
    # Deduplicate before truncating so repeated feed entries cannot consume the
    # requested quota while other usable recent headlines remain in the feed.
    unique: list[NewsHeadline] = []
    seen: set[str] = set()
    for item in items:
        key = " ".join(item.title.casefold().split())
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique[:count]


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def parse_news(records: object) -> list[NewsHeadline]:
    """Ignore malformed/duplicate entries; never fabricate titles."""
    if not isinstance(records, list):
        return []
    headlines: list[NewsHeadline] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        content = record.get("content", record)
        if not isinstance(content, dict):
            continue
        title = _text(content.get("title"))
        title_key = " ".join(title.casefold().split()) if title else ""
        if title is None or title_key in seen:
            continue
        seen.add(title_key)
        provider = content.get("provider")
        publisher = _text(content.get("publisher"))
        if isinstance(provider, dict):
            publisher = _text(provider.get("displayName")) or publisher
        url = _text(content.get("link"))
        for key in ("canonicalUrl", "clickThroughUrl"):
            candidate = content.get(key)
            if isinstance(candidate, dict):
                url = url or _text(candidate.get("url"))
        published = content.get("pubDate", content.get("providerPublishTime"))
        timestamp = None
        try:
            if isinstance(published, (int, float)) and not isinstance(published, bool):
                timestamp = datetime.fromtimestamp(published, timezone.utc).isoformat()
            elif isinstance(published, str):
                parsed = datetime.fromisoformat(published.replace("Z", "+00:00"))
                timestamp = parsed.isoformat()
        except (ValueError, OverflowError, OSError):
            pass
        headlines.append(NewsHeadline(title, publisher, timestamp, url))
    return headlines


def fetch_news(
    client: Any, count: int = 10, *, ticker: str | None = None
) -> list[NewsHeadline]:
    """Prefer Yahoo; supplement shortfalls with real RSS news when ticker is known."""
    try:
        try:
            records = client.get_news(count=count)
        except (AttributeError, TypeError):
            records = client.news
        headlines = parse_news(records)[:count]
    except Exception:
        logger.warning("News source=yfinance unavailable")
        headlines = []
    logger.info("News source=yfinance usable=%d", len(headlines))
    symbol = ticker or getattr(client, "ticker", None)
    if len(headlines) < count and isinstance(symbol, str) and symbol.strip():
        logger.info("News fallback=google_news_rss requested=%d", count)
        try:
            fallback = fetch_rss_news(symbol, count)
            seen = {" ".join(item.title.casefold().split()) for item in headlines}
            for item in fallback:
                key = " ".join(item.title.casefold().split())
                if key not in seen:
                    headlines.append(item)
                    seen.add(key)
            headlines = headlines[:count]
            logger.info(
                "News source=yfinance+google_news_rss usable=%d", len(headlines)
            )
        except Exception:
            logger.warning("News fallback=google_news_rss unavailable")
    if len(headlines) < count:
        logger.warning(
            "Only %d of %d requested usable headlines available", len(headlines), count
        )
    return headlines
