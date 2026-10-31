"""Retry a fetch blocked for this IP through the reader proxy, from another IP.

Only a failure another IP can clear retries: a proven bot block, a rate limit,
a dropped connection, a firewall's bare HTML refusal (``is_unmarked_block``),
or a browser that would not start. An API's JSON error is the site's own,
which another IP will not clear.
"""

from __future__ import annotations

from wesearch.fetch import FetchSession, PolicyParams, RequestParams, fetch
from wesearch.fetch.challenge import is_unmarked_block
from wesearch.fetch.providers.reader_proxy import fetch_reader_proxy
from wesearch.types.errors import BotDetectionError, FetchError


__all__ = ["fetch_with_reader_fallback"]


def fetch_with_reader_fallback(
    url: str,
    *,
    policy: PolicyParams,
    session: FetchSession | None = None,
    egress_bound_statuses: frozenset[int] = frozenset({0, 429}),
) -> tuple[bytes, bool]:
    """Fetch ``url``; on a failure bound to this IP, retry through the reader proxy.

    Args:
      url: The target URL (GET only benefits from the fallback).
      policy: How to fetch, for both the page and the proxy.
      session: Connections to reuse; ``None`` uses the process's shared session.
      egress_bound_statuses: Statuses that another IP can clear; ``0`` is a
        connection that failed, which a site dropping a flagged IP causes.

    Returns:
      body: The page, or the proxy's Markdown of it.
      via_reader_proxy: Whether ``body`` is the proxy's Markdown.

    Raises:
      FetchError: The page's error, when the proxy is not tried or also fails.
      OSError: The browser's failure to start, when the proxy also fails.

    """
    try:
        body = fetch(url, request=RequestParams(policy=policy), session=session)
        return body, False
    except BotDetectionError as e:
        primary: FetchError | OSError = e
    except FetchError as e:
        if e.status not in egress_bound_statuses and not is_unmarked_block(
            e,
            requested=url,
        ):
            raise
        primary = e
    except OSError as e:
        # ``BrowserUnavailableError``: the browser leg could not start here.
        primary = e
    try:
        return fetch_reader_proxy(url, policy=policy, session=session), True
    except (FetchError, ValueError, OSError) as e:
        raise primary from e
