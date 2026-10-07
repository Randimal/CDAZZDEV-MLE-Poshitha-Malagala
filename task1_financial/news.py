"""Defensive normalization of old and nested yfinance news payloads."""

import logging
from datetime import datetime, timezone
from typing import Any

from task1_financial.models import NewsHeadline

logger = logging.getLogger(__name__)


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
        if title is None or title.casefold() in seen:
            continue
        seen.add(title.casefold())
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


def fetch_news(client: Any, count: int = 10) -> list[NewsHeadline]:
    """Request at least ten headlines; optional provider failures degrade to []."""
    try:
        try:
            records = client.get_news(count=count)
        except (AttributeError, TypeError):
            records = client.news
        headlines = parse_news(records)[:count]
    except Exception as exc:
        logger.warning("News request failed: %s", exc, exc_info=True)
        headlines = []
    if len(headlines) < count:
        logger.warning(
            "Only %d of %d requested usable headlines available", len(headlines), count
        )
    return headlines
