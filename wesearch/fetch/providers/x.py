"""X (Twitter): pages render only with JavaScript, so fetch them via the reader proxy."""

from __future__ import annotations

from urllib.parse import urlparse


__all__ = ["matches"]


def matches(url: str) -> bool:
    """Return whether ``url`` is on ``x.com`` or ``twitter.com``, or a subdomain."""
    hostname = urlparse(url).hostname or ""
    return hostname in ("x.com", "twitter.com") or hostname.endswith(
        (".x.com", ".twitter.com"),
    )
