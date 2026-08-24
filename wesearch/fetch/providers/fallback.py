"""Bot-wall fallback: retry a blocked fetch through the reader proxy.

The primary fetch (Chrome TLS/HTTP-2 impersonation via
:func:`wesearch.fetch.fetch`) clears most sites. When a fetch fails for a
reason attached to THIS EGRESS -- a proven bot wall, or a rate limit keyed to
our address -- a same-egress retry presents the identical fingerprint and hits
the same wall. The reader proxy is the one retry from a genuinely different
address, so it is the sole fallback rung.

What engages the rung is the failure's KIND, not its status. A
``BotDetectionError`` is proven mitigation (see
:func:`wesearch.fetch.challenge.classify_http_error`), and a rate limit is
egress-keyed by definition. A bare 403 or 503 is neither: Cloudflare fronts an
origin's own errors, so an expired API token carries exactly the status a wall
does, and routing it to a third party egresses the URL for a failure no address
change can clear. Other statuses, non-GET methods, and connection/DNS errors
surface immediately. The rung is content-kind-agnostic -- it returns the proxy's
markdown bytes and lets the caller decide how to render them.
"""

from __future__ import annotations

from wesearch.fetch import PolicyParams, RequestParams, fetch
from wesearch.fetch.providers.reader_proxy import fetch_reader_proxy
from wesearch.types.errors import BotDetectionError, FetchError


__all__ = ["fetch_with_reader_fallback"]


def fetch_with_reader_fallback(
    url: str,
    *,
    policy: PolicyParams,
    egress_bound_statuses: frozenset[int] = frozenset({429}),
) -> tuple[bytes, bool]:
    """Fetch ``url``; on a bot-wall GET, retry through the reader proxy.

    Args:
      url: The target URL (GET only benefits from the fallback).
      policy: Transport and trust, applied to both the primary fetch and the
        proxy hop.
      egress_bound_statuses: Statuses whose failure is keyed to THIS egress
        rather than to the request, so a fetch from the proxy's address can
        clear them. A rate limit is the case: it is not a challenge (see
        :func:`wesearch.fetch.challenge.classify_http_error`), yet a
        different egress is exactly what it responds to.

    Returns:
      body: The response bytes -- the origin's HTML on the primary path, or the
        proxy's rendered markdown on the fallback path.
      via_reader_proxy: True when the body came from the reader proxy, so the
        caller can treat it as already-extracted markdown.

    Raises:
      FetchError: The primary error when a different egress could not clear it,
        or when the reader-proxy fallback also fails (the original is re-raised).

    """
    try:
        body, _session = fetch(url, request=RequestParams(policy=policy))
        return body, False
    except BotDetectionError as e:
        primary: FetchError = e
    except FetchError as e:
        # STATUS is not the test: ``challenge.py`` proved 403/503 insufficient,
        # because Cloudflare fronts an origin's own errors and an expired API
        # token is indistinguishable from a wall by status alone. Sending that
        # URL to a third party buys nothing and egresses it for no reason.
        if e.status not in egress_bound_statuses:
            raise
        primary = e
    try:
        return fetch_reader_proxy(url, policy=policy), True
    except (FetchError, ValueError, OSError) as e:
        raise primary from e
