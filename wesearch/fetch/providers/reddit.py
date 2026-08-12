"""Reddit provider: anonymous RSS, or richer OAuth JSON when configured.

Reddit bot-walls datacenter-IP requests to its JSON API, HTML pages, and
``old.reddit.com`` with a 403. The public ``.rss`` feed still serves
anonymously: a thread feed carries top-level comments as Atom entries, so titles,
bodies, and comment text survive.

:func:`fetch_reddit` returns raw bytes plus a tag identifying the payload shape;
rendering those bytes to text is the caller's concern, exactly as the other feed
formats are rendered by the caller.
"""

from __future__ import annotations

from typing import Final, Literal
from urllib.parse import urlparse

import re

from wesearch.fetch import (
    PolicyParams,
    RequestParams,
    fetch,
)


__all__ = [
    "RedditPayload",
    "fetch_reddit",
    "rss_url",
]

# The payload shape a fetch produced, so the caller selects the right renderer.
type RedditPayload = Literal["rss", "thread_json", "listing_json"]

_THREAD_RE: Final = re.compile(r"/r/[^/?#]+/comments/\w+")


def matches(url: str) -> bool:
    """Whether ``url`` is a Reddit URL (``reddit.com`` or any subdomain)."""
    hostname = urlparse(url).hostname or ""
    return hostname == "reddit.com" or hostname.endswith(".reddit.com")


def fetch_reddit(url: str, *, policy: PolicyParams) -> tuple[bytes, RedditPayload]:
    """Fetch a Reddit URL; return its bytes and the payload shape.

    Args:
      url: A Reddit URL (thread, subreddit, or user page).
      policy: Transport and trust forwarded to the HTTP layer.

    Returns:
      body: The raw feed/JSON bytes.
      payload: Which shape ``body`` is -- ``"rss"``, ``"thread_json"``, or
        ``"listing_json"``.

    Raises:
      FetchError: On an HTTP failure after any token refresh.

    """
    body, _session = fetch(rss_url(url), request=RequestParams(policy=policy))
    return body, "rss"


def rss_url(raw_url: str) -> str:
    """Return the ``.rss`` feed URL for any Reddit URL.

    Rewrites the path to end in ``.rss`` (dropping a trailing ``.json`` if
    present) while preserving the query string, which carries feed options like
    ``?limit=100`` and ``?sort=top``. A path already ending in ``.rss`` is
    returned unchanged.

    Args:
      raw_url: Any Reddit URL.

    Returns:
      feed_url: The ``.rss`` feed URL.

    """
    parsed = urlparse(raw_url)
    if parsed.path.endswith(".rss"):
        return raw_url
    path = re.sub(r"/?\.json$", "", parsed.path).rstrip("/")
    return parsed._replace(path=f"{path}/.rss").geturl()
