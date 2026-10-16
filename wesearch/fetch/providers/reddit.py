"""Reddit: fetch a page as its RSS feed, which Reddit serves to datacenter IPs.

Its pages and JSON API refuse them with a 403.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

import re

from wesearch.fetch import (
    FetchSession,
    PolicyParams,
    RequestParams,
    fetch,
)
from wesearch.fetch.session import default_session


__all__ = [
    "RedditPayload",
    "fetch_reddit",
    "rss_url",
]

type RedditPayload = Literal["rss", "thread_json", "listing_json"]


def matches(url: str) -> bool:
    """Return whether ``url`` is on ``reddit.com`` or a subdomain."""
    hostname = urlparse(url).hostname
    if hostname is None:
        return False
    return hostname == "reddit.com" or hostname.endswith(".reddit.com")


def fetch_reddit(
    url: str,
    *,
    policy: PolicyParams,
    session: FetchSession | None = None,
) -> tuple[bytes, RedditPayload]:
    """Fetch a Reddit URL; return its bytes and the payload shape.

    Args:
      url: A Reddit URL (thread, subreddit, or user page).
      policy: How to fetch.
      session: Connections and the Reddit token to reuse; ``None`` uses the
        process's shared session.

    Returns:
      body: The raw feed/JSON bytes.
      payload: Which shape ``body`` is -- ``"rss"``, ``"thread_json"``, or
        ``"listing_json"``.

    Raises:
      FetchError: On an HTTP failure after any token refresh.

    """
    session = default_session() if session is None else session
    return fetch(
        rss_url(url),
        request=RequestParams(policy=policy),
        session=session,
    ), "rss"


def rss_url(raw_url: str) -> str:
    """Return the ``.rss`` feed URL for any Reddit URL.

    Rewrites the path to end in ``.rss`` (dropping a trailing ``.json`` if
    present) while preserving the query string, which carries feed options like
    ``?limit=100`` and ``?sort=top``. A path already ending in ``.rss`` is
    returned unchanged.

    Args:
      raw_url: object Reddit URL.

    Returns:
      feed_url: The ``.rss`` feed URL.

    """
    parsed = urlparse(raw_url)
    if parsed.path.endswith(".rss"):
        return raw_url
    path = re.sub(r"/?\.json$", "", parsed.path)
    path = re.sub(r"/+$", "", path)
    return parsed._replace(path=f"{path}/.rss").geturl()
