"""Fetch like Chrome with curl_cffi: Chrome's TLS fingerprint and headers."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, cast
from urllib.parse import urlparse

import functools
import importlib
import logging

from wesearch.chrome.headers import (
    chrome_headers_for_host,
    chrome_identity_headers,
    chrome_platform,
)
from wesearch.fetch.challenge import classify_http_error
from wesearch.fetch.common import (
    DEFAULT_PORTS,
    PendingRequest,
    bracket_ipv6,
    merge_headers,
    origin,
    pinned_host,
)
from wesearch.fetch.session import default_session
from wesearch.types.errors import FetchError
from wesearch.types.params import ignore, no_cookies


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from curl_cffi import requests as cc_requests
    from curl_cffi.requests import Response

    import curl_cffi

    from wesearch.fetch.types import FetchSession
    from wesearch.types.params import HttpMethod
else:
    from wrapt import lazy_import

    curl_cffi = lazy_import("curl_cffi")  # 86ms.


__all__ = ["curl_available", "fetch_curl"]

logger = logging.getLogger(__name__)


def fetch_curl(
    url: str,
    *,
    method: HttpMethod = "GET",
    headers: Mapping[str, str | None] = MappingProxyType({}),
    body: bytes = b"",
    timeout_sec: float = 30.0,
    connect_timeout_sec: float = 10.0,
    max_redirects: int = 10,
    on_redirect: Callable[[str], None] = ignore,
    on_response: Callable[[int, dict[str, str], str], None] = ignore,
    allow_private_hosts: bool = False,
    session: FetchSession | None = None,
    cookies_for_host: Callable[[str], Mapping[str, str]] = no_cookies,
) -> bytes:
    """Fetch ``url`` looking like Chrome on this OS; return the response body.

    Sends Chrome's TLS fingerprint and headers, so most sites that refuse
    scripts serve it. Pages that need JavaScript need ``fetch_zendriver``.

    Args:
      url: Fully-qualified URL.
      method: HTTP method.
      headers: Sent over Chrome's headers, names in any case; a ``None`` value
        removes that header. A ``Cookie`` header is the request's cookies.
      body: Request body.
      timeout_sec: Seconds allowed for the whole request.
      connect_timeout_sec: Seconds allowed to connect.
      max_redirects: Redirects to follow; at 0 the redirect is returned.
      on_redirect: Called with each redirect target; raise to stop.
      on_response: Called with ``(status, headers, url)`` for every response.
      allow_private_hosts: Allow loopback, private, and link-local addresses;
        leave ``False`` for any URL you did not write.
      session: Connections to reuse; ``None`` uses the process's shared session.
      cookies_for_host: Returns the cookies to send a host a redirect moves to;
        the caller's ``Cookie`` stays with the first origin.

    Returns:
      body: The response body, decompressed.

    Raises:
      FetchError: On a status of 400 or more, or a failed connection (status 0).

    """
    session = default_session() if session is None else session
    pending = PendingRequest(
        url=url,
        method=method,
        headers=merge_headers(_chrome_headers(url, method), headers),
        body=body,
        remaining=max_redirects,
    )
    # curl_cffi drops an impersonated header whose value is None (measured); the
    # merge alone cannot remove one curl adds itself.
    removed = {name: value for name, value in headers.items() if value is None}
    while True:
        requested = pending.url
        try:
            response = _connection(
                session,
                requested,
                allow_private_hosts=allow_private_hosts,
            ).request(  # pyright: ignore[reportUnknownMemberType] -- curl_cffi's request kwargs have no complete stubs.
                cast("HttpMethod", pending.method),
                requested,
                headers={**pending.headers, **removed},
                data=pending.body or None,
                impersonate="chrome",
                timeout=(connect_timeout_sec, timeout_sec),
                allow_redirects=False,
                discard_cookies=True,
            )
        except curl_cffi.CurlError as error:
            raise FetchError(requested, 0, {}, str(error).encode()) from error
        status = int(response.status_code)  # pyrefly: ignore[unnecessary-type-conversion] -- The stub's type hides the runtime value (numpy/torch scalar or non-str key); the conversion is load-bearing.
        received = {k.lower(): str(v) for k, v in response.headers.items()}
        # ``items()`` joins repeated Set-Cookie with ", ", which cookie values
        # also contain; one per line splits back exactly.
        set_cookies = [
            cookie
            for cookie in response.headers.get_list("set-cookie")
            if cookie is not None
        ]
        if set_cookies:
            received["set-cookie"] = "\n".join(set_cookies)
        content = response.content or b""
        if pending.follow(
            status,
            received,
            on_response=on_response,
            on_redirect=on_redirect,
            cookies_for_host=cookies_for_host,
        ):
            continue
        if status >= 400:
            raise classify_http_error(requested, status, received, content)
        return content


@functools.cache
def curl_available() -> bool:
    """Return whether ``curl_cffi`` and its native library load on this host.

    Returns:
      available: ``False`` when the import fails; ``fetch`` then uses stdlib.

    """
    try:
        importlib.import_module("curl_cffi.requests")
    except ImportError:
        logger.warning("curl_cffi is unavailable; fetching with the stdlib transport.")
        return False
    return True


# curl_cffi sends the macOS Chrome's identity and no ``Origin``; Chrome on this host
# sends its own OS, and ``Origin`` on a non-GET.
def _chrome_headers(url: str, method: str) -> dict[str, str]:
    """Return the Chrome headers curl_cffi's impersonation leaves out."""
    target = curl_cffi.requests.impersonate.REAL_TARGET_MAP["chrome"]
    major = int(target.removeprefix("chrome"))
    platform = chrome_platform(target)
    headers = chrome_identity_headers(major=major, platform=platform)
    if method not in ("GET", "HEAD"):
        headers["Origin"] = origin(url)
    return headers | chrome_headers_for_host(
        urlparse(url).hostname or "",
        major=major,
        platform=platform,
    )


def _connection(
    session: FetchSession,
    url: str,
    *,
    allow_private_hosts: bool,
) -> cc_requests.Session[Response]:
    """Return the session's curl connection for ``url``'s host, opening it once."""
    parsed = urlparse(url)
    pin = pinned_host(url, allow_private_hosts=allow_private_hosts)
    port = parsed.port or DEFAULT_PORTS[parsed.scheme]
    key = (session.egress, parsed.hostname or "", pin, port)
    with session.lock:
        connection = session.curl.get(key)
        if isinstance(connection, curl_cffi.requests.Session):
            return cast("cc_requests.Session[Response]", connection)
        # Pinned per connection, so curl cannot re-resolve the host to a private
        # address. Session-level ``curl_options``: a handle-level pin misses
        # curl_cffi's per-thread handles (measured: DNS fails).
        opened = cast(
            "cc_requests.Session[Response]",
            curl_cffi.requests.Session(
                impersonate="chrome",
                curl_options={
                    curl_cffi.CurlOpt.RESOLVE: [
                        f"{pin.host}:{port}:{bracket_ipv6(pin.ip)}",
                    ],
                }
                if pin is not None
                else {},
            ),
        )
        session.curl[key] = opened
        return opened
