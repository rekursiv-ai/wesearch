"""Google News provider: rewrite SPA URLs to their public RSS endpoints.

The ``news.google.com`` single-page app server-renders only a sparse "Your
briefing" block; the bulk is hydrated by JS. The public RSS feed at the same
hostname serves the full set of story clusters as structured XML, which a feed
formatter renders cleanly. This provider maps the known front-page and search
paths to their RSS equivalents; article-detail and topic URLs have no RSS
analogue and pass through unchanged.
"""

from __future__ import annotations

from typing import Final, Literal
from urllib.parse import urlparse

from wesearch.fetch import RequestParams, Transport, ValidatedHosts, fetch


__all__ = [
    "NewsPayload",
    "fetch_google_news",
    "matches",
]

# The payload shape a fetch produced, so the caller selects the right renderer.
type NewsPayload = Literal["rss", "html"]

# Front-page paths that route to the top-stories RSS feed. Article-detail
# (``/articles/...``) and topic (``/topics/...``) URLs fall through unrewritten:
# topic ids don't map cleanly between the SPA and RSS URL spaces, and article
# pages have no RSS equivalent.
_TOP_PATHS: Final = frozenset({"", "/home", "/topstories", "/foryou"})


def matches(url: str) -> bool:
    """Whether ``url`` is the ``news.google.com`` host (no subdomains)."""
    return urlparse(url).hostname == "news.google.com"


def fetch_google_news(
    url: str,
    *,
    transport: Transport = "auto",
    validated_hosts: ValidatedHosts | None = None,
) -> tuple[bytes, NewsPayload]:
    """Fetch a Google News URL via RSS when the path has a rewrite, else HTML.

    Args:
      url: A ``news.google.com`` URL.
      transport: Retrieval transport forwarded to the HTTP layer.
      validated_hosts: Optional SSRF resolver pinning the connect IP per host;
        ``None`` leaves the fetch unpinned.

    Returns:
      body: The RSS XML or the raw HTML bytes.
      payload: ``"rss"`` when the fetch used an RSS endpoint, else ``"html"``.

    """
    target = _rewrite(url) or url
    body, _session = fetch(
        target,
        request=RequestParams(transport=transport, validated_hosts=validated_hosts),
    )
    payload: NewsPayload = "rss" if urlparse(target).path.startswith("/rss") else "html"
    return body, payload


def _rewrite(url: str) -> str | None:
    """Return the RSS-equivalent URL, or None for paths left unrewritten."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if path.startswith("/rss"):
        return None
    if path in _TOP_PATHS:
        return parsed._replace(path="/rss").geturl()
    if path == "/search":
        return parsed._replace(path="/rss/search").geturl()
    return None
