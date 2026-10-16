"""Google News: fetch the RSS feed for front-page and search URLs, else the page.

The page shows little without JavaScript; the feed has every story.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from wesearch.fetch import FetchSession, PolicyParams, RequestParams, fetch


__all__ = [
    "NewsPayload",
    "fetch_google_news",
    "matches",
]

type NewsPayload = Literal["rss", "html"]


def matches(url: str) -> bool:
    """Return whether ``url`` is on ``news.google.com``."""
    return urlparse(url).hostname == "news.google.com"


def fetch_google_news(
    url: str,
    *,
    policy: PolicyParams,
    session: FetchSession | None = None,
) -> tuple[bytes, NewsPayload]:
    """Fetch a Google News URL via RSS when the path has a rewrite, else HTML.

    Args:
      url: A ``news.google.com`` URL.
      policy: How to fetch.
      session: Connections to reuse; ``None`` uses the process's shared session.

    Returns:
      body: The RSS XML or the raw HTML bytes.
      payload: ``"rss"`` when the fetch used an RSS endpoint, else ``"html"``.

    """
    target = _rewrite(url) or url
    body = fetch(target, request=RequestParams(policy=policy), session=session)
    payload: NewsPayload = "rss" if urlparse(target).path.startswith("/rss") else "html"
    return body, payload


def _rewrite(url: str) -> str | None:
    """Return the RSS-equivalent URL, or None for paths left unrewritten."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    # Paths whose feed is the top stories; article and topic pages have no feed.
    if path in {"", "/home", "/topstories", "/foryou"}:
        return parsed._replace(path="/rss").geturl()
    if path == "/search":
        return parsed._replace(path="/rss/search").geturl()
    return None
