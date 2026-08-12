"""Reader-proxy provider: render a JS-walled page via a third-party service.

Some pages (single-page apps, aggressively bot-walled sites) serve no useful
content to an HTTP client -- the text only exists after a full browser renders
the page. When the local headless-browser transport cannot reach such a page
(a login wall, a residential-IP requirement), a hosted reader proxy renders it
and returns clean markdown.

The proxy receives the target URL as path data and contacts the target itself,
so the fetch egresses to the proxy host, not the target. Because that routes the
user's URL through a third party, the hop is opt-in: the caller must set
``WESEARCH_ALLOW_THIRD_PARTY_RENDER`` to a truthy value, or
:func:`fetch_reader_proxy` raises. An optional ``JINA_API_KEY`` authenticates the
proxy request, which is required when the egress network's anonymous quota is
blocked (the proxy returns HTTP 401 with a network-reputation message otherwise).

The proxy returns HTTP 200 even when ITS backend was bot-walled, embedding the
diagnostic as a ``Warning:`` line in the markdown; that sentinel is detected and
raised as a :class:`~wesearch.types.errors.FetchError` so a caller's fallback
ladder treats it as a soft failure rather than surfacing a proxy diagnostic as
article text.
"""

from __future__ import annotations

from typing import Final
from urllib.parse import quote

import os
import re

from wesearch.fetch import (
    ContentParams,
    PolicyParams,
    RequestParams,
    RetryParams,
    fetch,
)
from wesearch.types.errors import FetchError


__all__ = [
    "fetch_reader_proxy",
    "third_party_render_allowed",
]

_READER_PROXY_TEMPLATE: Final = "https://r.jina.ai/{url}"
_ALLOW_THIRD_PARTY_RENDER_ENV: Final = "WESEARCH_ALLOW_THIRD_PARTY_RENDER"
_API_KEY_ENV: Final = "JINA_AI_API_KEY"

# Sentinel the proxy embeds in a 200 body when its own backend was bot-walled;
# matched on the raw bytes because the failure rides a success status.
_SOFT_FAIL_RE: Final = re.compile(
    rb"Warning:\s*Target URL returned error \d{3}", re.IGNORECASE
)


def third_party_render_allowed() -> bool:
    """Return whether the operator opted into third-party rendering.

    Default: refuse. The reader proxy sees the target URL, which a
    privacy-sensitive operator may not want to disclose, so the hop requires
    explicit consent. Consent is signalled EITHER by a configured
    ``JINA_AI_API_KEY`` (a key means the operator already committed to using the
    proxy) OR by an explicit ``WESEARCH_ALLOW_THIRD_PARTY_RENDER`` for the
    keyless case (e.g. a clean-egress network within the anonymous quota).

    Returns:
      allowed: True when a Jina API key is set, or the env var is a truthy value
        (``1``/``true``/``yes``/``on``).

    """
    if os.environ.get(_API_KEY_ENV, "").strip():
        return True
    return os.environ.get(_ALLOW_THIRD_PARTY_RENDER_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def fetch_reader_proxy(url: str, *, policy: PolicyParams) -> bytes:
    """Render ``url`` through the reader proxy; return its markdown bytes.

    Args:
      url: The target URL to render.
      policy: Transport and trust for the proxy hop. The proxy host is fetched
        like any other URL, so the caller's trust level applies to it.

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
                f"Reader-proxy render requires third-party egress to "
                f"{_READER_PROXY_TEMPLATE.format(url='...')}; set "
                f"{_API_KEY_ENV} or {_ALLOW_THIRD_PARTY_RENDER_ENV}=1 to allow."
            ).encode(),
        )
    proxy_url = _READER_PROXY_TEMPLATE.format(url=quote(url, safe=":/"))
    key = os.environ.get(_API_KEY_ENV, "")
    headers = {"Authorization": f"Bearer {key}"} if key else None
    body, _session = fetch(
        proxy_url,
        request=RequestParams(
            content=ContentParams(headers=headers),
            retry=RetryParams(timeout_sec=30),
            policy=policy,
        ),
    )
    if _SOFT_FAIL_RE.search(body):
        raise FetchError(url, 502, {}, body[:200])
    return body
