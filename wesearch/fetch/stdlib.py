"""http.client reference transport for :mod:`wesearch.fetch`."""

from __future__ import annotations

from collections.abc import Callable
from typing import override
from urllib.parse import urlparse

import http.client
import ssl

from wesearch.fetch.challenge import classify_http_error
from wesearch.fetch.common import (
    _REDIRECT_STATUSES,
    Observer,
    apply_redirect,
    bracket_ipv6,
    decompress,
    decompress_error_body,
    host_header,
    join_headers,
    pinned_host,
    redirect_target,
)
from wesearch.types.params import Trust


__all__ = ["fetch_stdlib"]

HTTPConn = http.client.HTTPConnection | http.client.HTTPSConnection


class _ValidatedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        host: str,
        *,
        port: int | None = None,
        server_hostname: str,
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        super().__init__(host, port=port, timeout=timeout, context=context)
        self._server_hostname = server_hostname
        self._ssl_context = context

    @override
    def connect(self) -> None:
        http.client.HTTPConnection.connect(self)
        assert self.sock is not None
        self.sock = self._ssl_context.wrap_socket(
            self.sock,
            server_hostname=self._server_hostname,
        )


def _open_connection(
    scheme: str,
    hostname: str,
    timeout_sec: float,
    *,
    connect_timeout_sec: float | None = None,
    port: int | None = None,
    resolved_ip: str = "",
) -> HTTPConn:
    """Open a new HTTP/HTTPS connection; pin to ``resolved_ip`` when given.

    ``http.client`` keeps ONE timeout for the handshake and every later socket
    read. The connection is therefore built with the CONNECT budget, and the
    caller widens the live socket to ``timeout_sec`` once connected
    (:func:`_widen_after_connect`). Matches curl_cffi's ``(connect, read)``
    pair, which is what keeps the two backends from disagreeing on ``Retry``.
    """
    handshake_sec = (
        connect_timeout_sec if connect_timeout_sec is not None else timeout_sec
    )
    connect_host = bracket_ipv6(resolved_ip or hostname)
    if scheme == "https":
        ctx = ssl.create_default_context()
        if resolved_ip:
            return _ValidatedHTTPSConnection(
                connect_host,
                port=port,
                server_hostname=hostname,
                timeout=handshake_sec,
                context=ctx,
            )
        return http.client.HTTPSConnection(
            connect_host,
            port=port,
            timeout=handshake_sec,
            context=ctx,
        )
    return http.client.HTTPConnection(connect_host, port=port, timeout=handshake_sec)


def _widen_after_connect(conn: HTTPConn, timeout_sec: float) -> None:
    """Restore the full read budget on an already-connected socket.

    The connection was built with the narrower CONNECT budget, which would
    otherwise also cap every response read -- turning a slow page into a
    spurious timeout. No-op before the socket exists; ``http.client`` connects
    lazily, so the caller invokes this right after the first request.
    """
    if conn.sock is not None:
        conn.sock.settimeout(timeout_sec)


def fetch_stdlib(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout_sec: float,
    connect_timeout_sec: float | None = None,
    max_redirects: int,
    impersonate: str,
    on_redirect: Callable[[str], None] | None,
    on_response: Observer | None,
    trust: Trust = "untrusted",
    session: object | None = None,
    reseat: Callable[[str], object | None] | None = None,
) -> bytes:
    """Stdlib transport: http.client with manual redirect following.

    A drop-in peer of ``curl.fetch_curl`` with the identical signature, so the
    core dispatcher can select either transport. This backend has no TLS
    impersonation and no pooled connection, so ``impersonate``, ``session``, and
    ``reseat`` are accepted for interface parity and ignored; the coherent Chrome
    header set is instead hand-built upstream, in ``fetch``'s own header assembly.
    """
    del impersonate, session, reseat  # No impersonation or pooling here.
    # ``connect_timeout_sec`` is honored via _open_connection/_widen_after_connect.
    # The connection is owned entirely here: opened locally and closed in the
    # finally on every exit (success, HTTP error, redirect/decompress failure),
    # so no socket leaks. Nothing escapes to the caller.
    parsed = urlparse(url)
    scheme = parsed.scheme
    hostname = parsed.hostname or parsed.netloc
    port = parsed.port
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    validated = pinned_host(url, trust)
    connect_host = validated.ip if validated is not None else ""
    request_headers = headers
    if validated is not None:
        # Host first: real browsers and http.client's own auto-generated
        # Host header both place it before User-Agent/Accept/etc. Servers
        # that observe header order return 403 when Host is trailing. The
        # resolver returns the bare host (its contract), so host_header
        # re-appends any non-default port.
        request_headers = {
            "Host": host_header(validated.host, port, scheme),
            **headers,
        }

    raw_conn = _open_connection(
        scheme,
        hostname,
        timeout_sec,
        connect_timeout_sec=connect_timeout_sec,
        port=port,
        resolved_ip=connect_host,
    )
    try:
        current_url = url
        remaining = max_redirects
        while True:
            raw_conn.request(method, path, body=body, headers=request_headers)
            _widen_after_connect(raw_conn, timeout_sec)
            response = raw_conn.getresponse()
            resp_headers = join_headers(response.getheaders())
            if on_response is not None:
                on_response(response.status, resp_headers, current_url)

            is_redirect = response.status in _REDIRECT_STATUSES
            if is_redirect and remaining > 0:
                remaining -= 1
                response.read()  # Drain the socket before advancing the hop.
                redirect_url = redirect_target(
                    current_url, response.status, resp_headers
                )
                redir = urlparse(redirect_url)
                redir_scheme = redir.scheme or scheme
                if on_redirect is not None:
                    on_redirect(redirect_url)
                redir_parsed = urlparse(redirect_url)
                redir_hostname = redir_parsed.hostname or hostname
                redir_port = redir_parsed.port
                # Per-hop transform (Origin rewrite, 303 -> bodyless GET) via the
                # ONE shared helper, so the stdlib and curl paths cannot drift.
                # Applied to request_headers (which carries any validated Host).
                request_headers, method, body = apply_redirect(
                    current_url,
                    request_headers,
                    method,
                    body=body,
                    status=response.status,
                    redirect_url=redirect_url,
                )
                if (
                    redir_hostname != hostname
                    or redir_port != port
                    or redir_scheme != scheme
                ):
                    raw_conn.close()
                    scheme = redir_scheme
                    hostname = redir_hostname
                    port = redir_port
                    validated = pinned_host(redirect_url, trust)
                    connect_host = validated.ip if validated is not None else ""
                    # New host: replace the Host header (drop any prior first).
                    # HTTP field names are case-insensitive, so drop any casing.
                    request_headers = {
                        k: v for k, v in request_headers.items() if k.lower() != "host"
                    }
                    if validated is not None:
                        request_headers = {
                            "Host": host_header(validated.host, port, scheme),
                            **request_headers,
                        }
                    raw_conn = _open_connection(
                        scheme,
                        hostname,
                        timeout_sec,
                        connect_timeout_sec=connect_timeout_sec,
                        port=port,
                        resolved_ip=connect_host,
                    )
                path = redir.path or "/"
                if redir.query:
                    path = f"{path}?{redir.query}"
                current_url = redirect_url
                continue

            raw_body = response.read()
            if response.status >= 400:
                raise classify_http_error(
                    current_url,
                    response.status,
                    resp_headers,
                    decompress_error_body(raw_body, resp_headers),
                )
            encoding = resp_headers.get("content-encoding", "identity")
            return decompress(raw_body, encoding)
    finally:
        raw_conn.close()
