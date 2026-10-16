"""Render a page through r.jina.ai, which returns it as Markdown.

For pages that render only in a browser the local one cannot use. The proxy
sees the URL, so it is off unless ``JINA_AI_API_KEY`` or
``WESEARCH_ALLOW_THIRD_PARTY_RENDER`` is set.
"""

from __future__ import annotations

from urllib.parse import quote

import os
import re

from wesearch.fetch import (
    ContentParams,
    FetchSession,
    PolicyParams,
    RequestParams,
    fetch,
)
from wesearch.types.errors import FetchError


__all__ = [
    "fetch_reader_proxy",
    "third_party_render_allowed",
]


def third_party_render_allowed() -> bool:
    """Return whether ``JINA_AI_API_KEY`` or ``WESEARCH_ALLOW_THIRD_PARTY_RENDER`` is set.

    Returns:
      allowed: True when the key is set, or the variable is 1, true, yes, or on.

    """
    if os.environ.get("JINA_AI_API_KEY", "").strip():
        return True
    value = os.environ.get("WESEARCH_ALLOW_THIRD_PARTY_RENDER")
    return value is not None and value.strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def fetch_reader_proxy(
    url: str,
    *,
    policy: PolicyParams,
    session: FetchSession | None = None,
) -> bytes:
    """Render ``url`` through the reader proxy; return its markdown bytes.

    Args:
      url: The target URL to render.
      policy: How to fetch the proxy.
      session: Connections to reuse; ``None`` uses the process's shared session.

    Returns:
      markdown: The proxy's extracted-markdown response bytes.

    Raises:
      FetchError: When third-party rendering is not permitted, or when the
        proxy signals a soft failure (a ``Warning:`` sentinel in a 200 body).

    """
    if not third_party_render_allowed():
        raise FetchError(
            url,
            0,
            {},
            (
                b"Reader-proxy render requires third-party egress to "
                b"https://r.jina.ai/...; set JINA_AI_API_KEY or "
                b"WESEARCH_ALLOW_THIRD_PARTY_RENDER=1 to allow."
            ),
        )
    proxy_url = f"https://r.jina.ai/{quote(url, safe=':/')}"
    key = os.environ.get("JINA_AI_API_KEY", "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    body = fetch(
        proxy_url,
        request=RequestParams(
            content=ContentParams(headers=headers),
            policy=policy,
        ),
        session=session,
    )
    # The proxy reports its own failure inside a 200 body.
    if re.search(rb"Warning:\s*Target URL returned error \d{3}", body, re.IGNORECASE):
        raise FetchError(url, 502, {}, body[:200])
    return body
