"""Fetch a URL: the one way ``wesearch`` reaches the network.

Usage::

    from wesearch.fetch import fetch

    html = fetch(url).decode()
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlencode, urlparse, urlsplit, urlunsplit

import base64
import ipaddress
import json
import logging
import time

from wesearch.fetch.common import cookie_header, merge_headers
from wesearch.fetch.gate import gated
from wesearch.fetch.session import default_session
from wesearch.fetch.transport import transport_routing
from wesearch.fetch.transport.curl import curl_available, fetch_curl
from wesearch.fetch.transport.stdlib import fetch_stdlib
from wesearch.profile import Profile, parse_set_cookie
from wesearch.types.errors import BotDetectionError, FetchError
from wesearch.types.params import (
    NO_BODY,
    ContentParams,
    RequestParams,
    RetryParams,
    Transport,
    no_cookies,
)


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from wesearch.fetch.transport import zendriver
    from wesearch.fetch.types import Fetch, FetchSession
    from wesearch.types.params import HttpMethod
else:
    from wrapt import lazy_import

    zendriver = lazy_import("wesearch.fetch.transport.zendriver")  # 62ms.


__all__ = [
    "egress_ip",
    "fetch",
    "resolve_transport",
]

logger = logging.getLogger(__name__)


def fetch(
    url: str,
    *,
    request: RequestParams | None = None,
    session: FetchSession | None = None,
) -> bytes:
    """Fetch ``url``; return the response body.

    Cookies a site sets are kept, per this host's public IP and the site, and
    sent back on later fetches, also from other processes.

    Args:
      url: Fully-qualified URL.
      request: What to send and how; ``None`` is a plain GET.
      session: Connections to reuse across fetches; ``None`` uses the
        process's shared session.

    Returns:
      body: The response body, decompressed.

    Raises:
      FetchError: On a status of 400 or more after the retries, or a failed
        connection (status 0).
      BotDetectionError: When the site served a block instead of the page.
      CooldownError: When a recent block on this IP holds the site shut; no
        request was sent.
      ValueError: When the host is not public and ``allow_private_hosts`` is
        not set, or the body cannot be decompressed.

    """
    params = RequestParams() if request is None else request
    session = default_session() if session is None else session
    # Raw requests probe the egress itself, so they must not wait on one.
    egress = session.egress if params.content.raw_headers else egress_ip(session)
    with gated(url, egress=egress):
        if not params.content.raw_headers:
            return _fetch_as_profile(url, params, session=session)
        return _send(url, params, session=session, profile_cookies=False)


def resolve_transport(
    transport: Transport,
    *,
    method: HttpMethod = "GET",
    raw_headers: bool = False,
    has_body: bool = False,
    domain: str = "",
) -> Transport:
    """Return the transport ``auto`` and an unavailable curl stand for.

    Args:
      transport: The requested transport.
      method: HTTP method; only a GET can go to the browser.
      raw_headers: Whether the request must go out byte for byte.
      has_body: Whether the request has a body.
      domain: The request's host; a host known to block curl goes to the browser.

    Returns:
      transport: A transport that can send this request.

    """
    if transport == "auto":
        browsable = method == "GET" and not raw_headers and not has_body
        transport = "curl-then-zendriver" if browsable else "curl"
    if (
        transport == "curl-then-zendriver"
        and domain in transport_routing.zendriver_domains()
    ):
        return "zendriver"
    if transport == "curl" and not curl_available():
        return "stdlib"
    return transport


def egress_ip(
    session: FetchSession,
    *,
    refresh: bool = False,
    ipv6: bool = False,
    v4_echoes: Sequence[str] = ("https://ipv4.icanhazip.com", "https://api.ipify.org"),
    v6_echoes: Sequence[str] = (
        "https://ipv6.icanhazip.com",
        "https://api64.ipify.org",
    ),
    timeout_sec: float = 5.0,
    retry_after_sec: float = 60.0,
) -> str:
    """Return this host's public IP, or ``""`` when none resolves.

    The IPv4 answer is kept on ``session``. When it changes, every curl
    connection opened on another address, or before it was known, closes. When no echo
    answers, the next probe waits ``retry_after_sec`` unless ``refresh``.

    Args:
      session: Holds the last IPv4 answer.
      refresh: Probe even when ``session`` already knows it.
      ipv6: Return the IPv6 address instead; never kept.
      v4_echoes: Services that answer with the caller's IPv4 address.
      v6_echoes: Services that answer with the caller's IPv6 address.
      timeout_sec: Seconds allowed per probe.
      retry_after_sec: Seconds an IPv4 probe that found nothing is not repeated.

    Returns:
      ip: The public address, or ``""``.

    """
    if (
        not refresh
        and not ipv6
        and (session.egress or time.monotonic() < session.egress_retry_at)
    ):
        return session.egress
    for echo in v6_echoes if ipv6 else v4_echoes:
        try:
            ip = (
                fetch(
                    echo,
                    request=RequestParams(
                        content=ContentParams(raw_headers=True),
                        retry=RetryParams(timeout_sec=timeout_sec),
                    ),
                    session=session,
                )
                .decode()
                .strip()
            )
            if ipaddress.ip_address(ip).version != (6 if ipv6 else 4):
                continue
        except (FetchError, ValueError):
            continue
        if not ipv6 and ip != session.egress:
            with session.lock:
                session.egress = ip
                stale = [k for k in session.curl if k[0] != ip]
                closing = [session.curl.pop(k) for k in stale]
            for connection in closing:
                connection.close()
        return ip
    if not ipv6:
        session.egress_retry_at = time.monotonic() + retry_after_sec
    return ""


def _fetch_as_profile(
    url: str,
    params: RequestParams,
    *,
    session: FetchSession,
) -> bytes:
    """Fetch with this egress's stored cookies for the host, saving what it sets."""
    transport = resolve_transport(
        params.policy.transport,
        method=params.content.method,
        has_body=params.content.has_body,
        domain=urlparse(url).hostname or "",
    )
    if transport == "curl-then-zendriver":
        # The browser clears blocks curl cannot; anything else it would not help.
        try:
            return _fetch_as_profile(url, _on(params, "curl"), session=session)
        except BotDetectionError as block:
            domain = urlparse(url).hostname or ""
            if domain and block.browser_clears:
                transport_routing.remember_zendriver_domain(domain)
            return _fetch_as_profile(url, _on(params, "zendriver"), session=session)
    params = _on(params, transport)
    try:
        return _send(url, params, session=session, profile_cookies=True)
    except BotDetectionError:
        # A stored identity may be what got blocked: forget it and retry once
        # from a freshly probed egress. The browser's wall is final.
        domain = urlparse(url).hostname or ""
        if (
            transport == "zendriver"
            or session.profiles.load(session.egress, domain) is None
        ):
            raise
        session.profiles.discard(session.egress, domain)
        egress_ip(session, refresh=True)
        return _send(url, params, session=session, profile_cookies=True)


def _on(params: RequestParams, transport: Transport) -> RequestParams:
    """Return ``params`` with ``transport``."""
    return replace(params, policy=replace(params.policy, transport=transport))


def _send(
    url: str,
    params: RequestParams,
    *,
    session: FetchSession,
    profile_cookies: bool,
) -> bytes:
    """Send ``params`` once, with retries, filing the cookies each response sets."""
    url, credentials = _without_userinfo(url)
    parts = urlsplit(url)
    if params.content.params:
        query = urlencode(params.content.params)
        url = urlunsplit(
            parts._replace(query=f"{parts.query}&{query}" if parts.query else query),
        )
    content = params.content
    body, content_type = b"", ""
    if content.data is not None:
        body, content_type = (
            urlencode(content.data).encode(),
            "application/x-www-form-urlencoded",
        )
    elif content.json is not NO_BODY:
        body, content_type = json.dumps(content.json).encode(), "application/json"
    headers: dict[str, str | None] = {}
    if content_type and not content.raw_headers:
        headers["Content-Type"] = content_type
    if credentials:
        headers["Authorization"] = credentials

    def stored_cookies(host: str) -> Mapping[str, str]:
        if not profile_cookies:
            return no_cookies(host)
        profile = session.profiles.load(session.egress, host)
        return no_cookies(host) if profile is None else profile.cookies

    headers.update(content.headers)
    cookies = {**stored_cookies(parts.hostname or ""), **content.cookies}
    if cookies:
        # In any case: "cookie" beside "Cookie" sends two headers, a bot tell.
        cookie = cookie_header(
            {k: v for k, v in headers.items() if v is not None},
            cookies,
        )
        headers = merge_headers({}, {**headers, "Cookie": cookie}) | {
            k: v for k, v in headers.items() if v is None
        }
    transports: dict[Transport, Fetch] = {
        "curl": fetch_curl,
        "stdlib": fetch_stdlib,
        "zendriver": zendriver.fetch_zendriver,
    }
    transport = resolve_transport(
        params.policy.transport,
        method=content.method,
        raw_headers=content.raw_headers,
        has_body=content.has_body,
    )
    landed: dict[str, dict[str, str]] = {}

    def on_response(status: int, received: dict[str, str], hop: str) -> None:
        if (set_cookie := received.get("set-cookie", "")) and (
            host := urlparse(hop).hostname or ""
        ):
            landed.setdefault(host, {}).update(parse_set_cookie(set_cookie))
        params.observe.on_response(status, received)

    attempt = 0
    while True:
        try:
            body_out = transports[transport](
                url,
                method=content.method,
                headers=headers,
                body=body,
                timeout_sec=params.retry.timeout_sec,
                connect_timeout_sec=params.retry.connect_timeout_sec,
                max_redirects=params.retry.max_redirects,
                on_redirect=params.observe.on_redirect,
                on_response=on_response,
                allow_private_hosts=params.policy.allow_private_hosts,
                session=session,
                cookies_for_host=stored_cookies,
            )
            break
        except FetchError as error:
            retryable = error.status in (0, 429, 500, 502, 503, 504)
            if not retryable or attempt == params.retry.retries:
                raise
            delay_sec = params.retry.backoff_delay(attempt, error.headers)
            logger.debug("fetch %s -> %d, retry in %.1fs", url, error.status, delay_sec)
            attempt += 1
            time.sleep(delay_sec)
    # Per attempt, so a 200 that is really a block sends ``auto`` to the browser.
    params.observe.body_validator(body_out)
    if profile_cookies and session.egress:
        for host, cookies_set in landed.items():
            # The body already arrived: losing a cookie must not fail the fetch.
            try:
                if session.profiles.load(session.egress, host) is None:
                    session.profiles.save(
                        session.egress,
                        host,
                        Profile(cookies=cookies_set),
                    )
                else:
                    session.profiles.update_cookies(session.egress, host, cookies_set)
            except OSError:
                logger.warning("could not save cookies for %s", host, exc_info=True)
    return body_out


def _without_userinfo(url: str) -> tuple[str, str]:
    """Return ``url`` without ``user:pass@`` and the Basic credential it carried."""
    parsed = urlparse(url)
    username, password = parsed.username or "", parsed.password or ""
    if not (username or password):
        return url, ""
    pair = f"{unquote(username)}:{unquote(password)}"
    netloc = parsed.netloc[parsed.netloc.rfind("@") + 1 :]
    return (
        parsed._replace(netloc=netloc).geturl(),
        f"Basic {base64.b64encode(pair.encode()).decode()}",
    )
