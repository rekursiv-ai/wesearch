"""X (Twitter) provider: render the SPA via the reader proxy.

X serves an empty shell to non-JS clients; tweet text appears only after a JS
hydration step, and the logged-out timeline is additionally gated. A local
headless browser reaches the shell but not the content, so this provider
delegates the render to the reader proxy, which returns extracted markdown.

Because the reader proxy is a third party that sees the target URL, the hop is
opt-in via :func:`wesearch.fetch.providers.reader_proxy.third_party_render_allowed`
(a configured ``JINA_AI_API_KEY`` or ``WESEARCH_ALLOW_THIRD_PARTY_RENDER``).
"""

from __future__ import annotations

from urllib.parse import urlparse

from wesearch.fetch import PolicyParams
from wesearch.fetch.providers.reader_proxy import fetch_reader_proxy


__all__ = [
    "fetch_x",
    "matches",
]


def matches(url: str) -> bool:
    """Whether ``url`` is an X/Twitter URL (``x.com``/``twitter.com`` + subs)."""
    hostname = urlparse(url).hostname or ""
    if hostname in ("x.com", "twitter.com"):
        return True
    return hostname.endswith((".x.com", ".twitter.com"))


def fetch_x(url: str, *, policy: PolicyParams) -> bytes:
    """Render an X URL through the reader proxy; return its markdown bytes.

    Args:
      url: An X/Twitter URL.
      policy: Transport and trust for the proxy hop.

    Returns:
      markdown: The proxy's extracted-markdown bytes.

    Raises:
      FetchError: When third-party rendering is not permitted, or the proxy
        fails.

    """
    return fetch_reader_proxy(url, policy=policy)
