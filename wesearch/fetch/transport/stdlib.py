"""Fetch with only the standard library: Chrome's headers, Python's TLS."""

from __future__ import annotations

from http import client
from types import MappingProxyType
from typing import TYPE_CHECKING, cast, override
from urllib.parse import urlparse

import socket
import ssl
import time

from wesearch.chrome.headers import (
    chrome_headers_for_host,
    chrome_navigation_headers,
    chrome_platform,
)
from wesearch.chrome.useragents import newest_chrome_major
from wesearch.fetch.challenge import classify_http_error
from wesearch.fetch.common import (
    PendingRequest,
    bracket_ipv6,
    decompress,
    decompress_error_body,
    host_header,
    join_headers,
    merge_headers,
    origin,
    pinned_host,
)
from wesearch.types.errors import FetchError
from wesearch.types.params import ignore, no_cookies


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from wesearch.fetch.types import FetchSession
    from wesearch.types.params import HttpMethod


__all__ = ["fetch_stdlib"]


def fetch_stdlib(
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
    """Fetch ``url`` with only the standard library; return the response body.

    Sends Chrome's headers but Python's TLS fingerprint, so sites that check
    the fingerprint refuse it. Use it where ``curl_cffi`` is not installed.

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
      session: Unused: every request opens its own connection.
      cookies_for_host: Returns the cookies to send a host a redirect moves to;
        the caller's ``Cookie`` stays with the first origin.

    Returns:
      body: The response body, decompressed.

    Raises:
      FetchError: On a status of 400 or more, or a failed connection (status 0).

    """
    del session
    platform = chrome_platform("chrome")
    major = newest_chrome_major(platform)
    content_type = next(
        (
            v
            for k, v in headers.items()
            if k.lower() == "content-type" and v is not None
        ),
        "",
    )
    pending = PendingRequest(
        url=url,
        method=method,
        headers=merge_headers(
            chrome_navigation_headers(
                major=major,
                platform=platform,
                method=method,
                content_type=content_type,
                origin=origin(url),
            )
            | chrome_headers_for_host(
                urlparse(url).hostname or "",
                major=major,
                platform=platform,
            ),
            headers,
        ),
        body=body,
        remaining=max_redirects,
    )
    deadline = time.monotonic() + timeout_sec
    while True:
        requested = pending.url
        parsed = urlparse(requested)
        pin = pinned_host(requested, allow_private_hosts=allow_private_hosts)
        hostname = parsed.hostname or ""
        # First, as Chrome sends it: servers that check order 403 a trailing Host.
        sent = (
            {"Host": host_header(pin.host, parsed.port, parsed.scheme)} if pin else {}
        )
        sent |= pending.headers
        connection = _connect(
            parsed.scheme,
            hostname,
            port=parsed.port,
            ip=pin.ip if pin else "",
            connect_timeout_sec=connect_timeout_sec,
        )
        try:
            connection.request(
                pending.method,
                f"{parsed.path or '/'}{f'?{parsed.query}' if parsed.query else ''}",
                body=pending.body or None,
                headers=sent,
            )
            # Held here: ``getresponse`` drops ``connection.sock`` when the
            # server will close, yet the response still reads through it.
            sock = connection.sock
            chunks: list[bytes] = []
            response: client.HTTPResponse | None = None
            # A socket timeout bounds each read, so a server trickling bytes
            # would never time out; the deadline bounds the whole request.
            # The response closes the socket after the last byte, so stop there.
            while response is None or not response.isclosed():
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"no response within {timeout_sec}s")
                if sock is not None:
                    sock.settimeout(left)
                if response is None:
                    response = connection.getresponse()
                elif chunk := response.read(65_536):
                    chunks.append(chunk)
                else:
                    break
            received = join_headers(response.getheaders())
            content = b"".join(chunks)
        except (OSError, client.HTTPException) as error:
            raise FetchError(requested, 0, {}, str(error).encode()) from error
        finally:
            connection.close()
        if pending.follow(
            response.status,
            received,
            on_response=on_response,
            on_redirect=on_redirect,
            cookies_for_host=cookies_for_host,
        ):
            continue
        if response.status >= 400:
            raise classify_http_error(
                requested,
                response.status,
                received,
                decompress_error_body(content, received),
            )
        return decompress(content, received.get("content-encoding", "identity"))


def _connect(
    scheme: str,
    hostname: str,
    *,
    port: int | None,
    ip: str,
    connect_timeout_sec: float,
) -> client.HTTPConnection:
    """Return an unopened connection to ``ip`` if given, else to ``hostname``."""
    host = bracket_ipv6(ip or hostname)
    if scheme != "https":
        return client.HTTPConnection(host, port=port, timeout=connect_timeout_sec)
    if ip:
        return _PinnedHTTPSConnection(
            host,
            port=port,
            hostname=hostname,
            timeout=connect_timeout_sec,
        )
    return client.HTTPSConnection(
        host,
        port=port,
        timeout=connect_timeout_sec,
        context=ssl.create_default_context(),
    )


class _PinnedHTTPSConnection(client.HTTPSConnection):
    """An HTTPS connection to a fixed IP that verifies the certificate by name."""

    def __init__(
        self,
        ip: str,
        *,
        port: int | None,
        hostname: str,
        timeout: float,
    ) -> None:
        context = ssl.create_default_context()
        super().__init__(ip, port=port, timeout=timeout, context=context)
        self._hostname = hostname
        self._context = context

    @override
    def connect(self) -> None:
        client.HTTPConnection.connect(self)
        self.sock = self._context.wrap_socket(
            cast(socket.socket, self.sock),
            server_hostname=self._hostname,
        )
