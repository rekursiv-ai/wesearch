"""Bot-wall fallback: retry a blocked fetch through the reader proxy.

The primary fetch (Chrome TLS/HTTP-2 impersonation via
:func:`wesearch.fetch.fetch`) clears most sites. A 403/429/503 on a GET is
the signature of edge-side bot detection (Fastly, Akamai, Cloudflare); a
same-egress retry would present the identical fingerprint and hit the same wall.
The reader proxy is the one retry with a genuinely different egress, so it is the
sole fallback rung.

Only that specific bot-wall signature engages the rung: other statuses (404,
500), non-GET methods, and connection/DNS errors surface immediately. The rung
is content-kind-agnostic -- it returns the proxy's markdown bytes and lets the
caller decide how to render them.
"""

from __future__ import annotations

from typing import Final

from wesearch.fetch import Policy, RequestParams, fetch
from wesearch.fetch.providers.reader_proxy import fetch_reader_proxy
from wesearch.types.errors import FetchError


__all__ = ["fetch_with_reader_fallback"]

# Edge-side bot-detection statuses a different-egress retry can clear.
_BOT_WALL_STATUSES: Final = frozenset({403, 429, 503})


def fetch_with_reader_fallback(url: str, *, policy: Policy) -> tuple[bytes, bool]:
    """Fetch ``url``; on a bot-wall GET, retry through the reader proxy.

    Args:
      url: The target URL (GET only benefits from the fallback).
      policy: Transport and trust, applied to both the primary fetch and the
        proxy hop.

    Returns:
      body: The response bytes -- the origin's HTML on the primary path, or the
        proxy's rendered markdown on the fallback path.
      via_reader_proxy: True when the body came from the reader proxy, so the
        caller can treat it as already-extracted markdown.

    Raises:
      FetchError: The primary error when it is not a bot-wall status, or when the
        reader-proxy fallback also fails (the original error is re-raised).

    """
    try:
        body, _session = fetch(url, request=RequestParams(policy=policy))
        return body, False
    except FetchError as e:
        if e.status not in _BOT_WALL_STATUSES:
            raise
        primary = e
    try:
        return fetch_reader_proxy(url, policy=policy), True
    except (FetchError, ValueError, OSError) as e:
        raise primary from e
