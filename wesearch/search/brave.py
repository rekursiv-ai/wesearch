"""The Brave Search API backend: a keyed JSON API, so nothing to bot-wall.

Requires ``BRAVE_API_KEY``. The only backend whose throughput is bought
rather than borrowed from the egress IP's reputation -- and billed per request,
which is why the default chain tries it last.
"""

from __future__ import annotations

from typing import Final
from urllib.parse import urlencode

import html
import os
import re

from treekle import from_plain, json_loads

from wesearch.fetch import (
    ContentParams,
    PolicyParams,
    RequestParams,
    RetryParams,
    Transport,
    fetch,
)
from wesearch.search.custom_types import SearchError, SearchResult, clean_text


__all__ = ["BRAVE_API_KEY_ENV", "brave"]

BRAVE_API_KEY_ENV: Final = "BRAVE_API_KEY"


def brave(
    query: str,
    num_results: int = 10,
    headers: dict[str, str] | None = None,
    *,
    timeout_sec: float = 15.0,
    connect_timeout_sec: float = 3.0,
    retries: int = 1,
    transport: Transport = "auto",
    max_count: int = 20,
) -> list[SearchResult]:
    """Query the Brave Search web endpoint and return parsed results.

    Args:
      query: Search query string.
      num_results: Maximum results to return.
      headers: Optional override headers forwarded to fetch.
      timeout_sec: HTTP ceiling per attempt.
      connect_timeout_sec: Ceiling on the handshake alone.
      retries: Retry attempts for a transient failure.
      transport: Retrieval transport.
      max_count: Brave's per-request ``count`` ceiling.

    Returns:
      results: Parsed search results.

    Raises:
      SearchError: When ``BRAVE_API_KEY`` is unset.

    """
    if num_results < 0:
        raise ValueError(f"'num_results' must be >= 0, got {num_results}.")
    if num_results == 0:
        return []
    key = os.environ.get(BRAVE_API_KEY_ENV, "").strip()
    if not key:
        raise SearchError(f"{BRAVE_API_KEY_ENV} must be set to use Brave search")
    params = urlencode({"q": query, "count": min(num_results, max_count)})
    body, _ = fetch(
        f"https://api.search.brave.com/res/v1/web/search?{params}",
        request=RequestParams(
            content=ContentParams(
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": key,
                    **(headers or {}),
                },
            ),
            retry=RetryParams(
                retries=retries,
                timeout_sec=timeout_sec,
                connect_timeout_sec=connect_timeout_sec,
            ),
            policy=PolicyParams(transport=transport),
        ),
    )
    return _brave_parse(body, num_results)


def _brave_parse(body: bytes, num_results: int) -> list[SearchResult]:
    """Map Brave's ``web.results`` onto :class:`SearchResult`, dropping malformed rows."""
    tree = json_loads(body, dict[str, object])
    web = from_plain(tree.get("web"), dict[str, object], default={})
    rows = from_plain(web.get("results"), list[dict[str, object]], default=[])
    results = [
        SearchResult(
            url=from_plain(row.get("url"), str, default=""),
            title=_plain(from_plain(row.get("title"), str, default="")),
            snippet=_plain(from_plain(row.get("description"), str, default="")),
        )
        for row in rows
    ]
    return [r for r in results if r.url.startswith(("http://", "https://"))][
        :num_results
    ]


# Brave marks query terms with ``<strong>`` and escapes entities in titles and
# descriptions; a model reading the snippet wants the text.
def _plain(text: str) -> str:
    """Strip inline markup and unescape entities."""
    return clean_text(html.unescape(re.sub(r"<[^>]+>", "", text)))
