"""Tests for what only the stdlib transport does: its sockets and its pins.

The behavior it shares with curl is in ``transport_test.py``.
"""

from __future__ import annotations

from http import client
from typing import cast
from unittest.mock import Mock, patch

import ssl

from wesearch.fetch.common import ValidatedHost
from wesearch.fetch.transport.stdlib import (
    _open_connection,
    _ValidatedHTTPSConnection,
    _widen_after_connect,
    fetch_stdlib,
)


class _Conn:
    """An ``http.client`` connection that records requests and replays responses."""

    def __init__(self, *responses: Mock) -> None:
        self.sock = None
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        self.closed = False
        self._responses = list(responses)

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        headers: dict[str, str],
    ) -> None:
        del body
        self.requests.append((method, path, dict(headers)))

    def getresponse(self) -> Mock:
        return self._responses.pop(0)

    def close(self) -> None:
        self.closed = True


def _response(
    status: int = 200,
    body: bytes = b"ok",
    headers: list[tuple[str, str]] | None = None,
) -> Mock:
    resp = Mock(spec=client.HTTPResponse)
    resp.status = status
    resp.read.return_value = body
    resp.getheaders.return_value = headers or []
    return resp


def _fetch(url: str, *, headers: dict[str, str]) -> bytes:
    return fetch_stdlib(
        url,
        method="GET",
        headers=headers,
        body=None,
        timeout_sec=5.0,
        max_redirects=10,
        impersonate="chrome",
        on_redirect=None,
        on_response=None,
    )


def test_an_untrusted_request_dials_the_pin_and_names_the_host_first() -> None:
    conn = _Conn(_response())
    pin = ValidatedHost(host="example.com", ip="93.184.216.34")
    with (
        patch("wesearch.fetch.transport.stdlib.pinned_host", return_value=pin),
        patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=conn,
        ) as opened,
    ):
        _fetch("https://example.com:8443/p?q=1", headers={"Accept": "*/*"})
    assert opened.call_args.kwargs["resolved_ip"] == "93.184.216.34"
    [(method, path, headers)] = conn.requests
    assert (method, path) == ("GET", "/p?q=1")
    assert list(headers) == ["Host", "Accept"]
    assert headers["Host"] == "example.com:8443"


def test_a_cross_host_redirect_reopens_and_re_pins() -> None:
    first = _Conn(
        _response(302, b"", [("Location", "https://other.example/next?x=1")]),
    )
    second = _Conn(_response(body=b"final"))
    pins = iter(
        [
            ValidatedHost(host="example.com", ip="93.184.216.34"),
            ValidatedHost(host="other.example", ip="93.184.216.35"),
        ],
    )

    def pin(url: str, trust: str) -> ValidatedHost:
        del url, trust
        return next(pins)

    with (
        patch("wesearch.fetch.transport.stdlib.pinned_host", pin),
        patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            side_effect=[first, second],
        ) as opened,
    ):
        assert _fetch("https://example.com/start", headers={"host": "stale"}) == (
            b"final"
        )
    assert first.closed
    assert opened.call_args.kwargs["resolved_ip"] == "93.184.216.35"
    [(_, path, headers)] = second.requests
    assert path == "/next?x=1"
    assert [k for k in headers if k.lower() == "host"] == ["Host"]
    assert headers["Host"] == "other.example"


def test_a_pinned_https_connection_verifies_the_hostname() -> None:
    conn = _open_connection(
        "https",
        "example.com",
        5.0,
        connect_timeout_sec=1.0,
        port=8443,
        resolved_ip="2606:4700::1",
    )
    assert isinstance(conn, _ValidatedHTTPSConnection)
    assert conn.host == "2606:4700::1"
    assert conn.timeout == 1.0
    assert conn._server_hostname == "example.com"


def test_an_unpinned_connection_dials_the_hostname() -> None:
    https = _open_connection("https", "example.com", 5.0)
    http = _open_connection("http", "example.com", 5.0, port=8080)
    assert type(https) is client.HTTPSConnection
    assert https.host == "example.com"
    assert https.timeout == 5.0
    assert type(http) is client.HTTPConnection
    assert (http.host, http.port) == ("example.com", 8080)


def test_the_validated_connection_wraps_its_socket_for_the_hostname() -> None:
    context = cast("ssl.SSLContext", Mock(spec=ssl.SSLContext))
    conn = _ValidatedHTTPSConnection(
        "93.184.216.34",
        server_hostname="example.com",
        timeout=1.0,
        context=context,
    )
    raw = Mock()

    def connect(self: client.HTTPConnection) -> None:
        self.sock = raw

    with (
        patch.object(client.HTTPConnection, "connect", connect),
        patch.object(context, "wrap_socket") as wrap,
    ):
        conn.connect()
    wrap.assert_called_once_with(raw, server_hostname="example.com")
    assert conn.sock is wrap.return_value


def test_the_read_budget_widens_only_once_connected() -> None:
    conn = client.HTTPConnection("example.com")
    _widen_after_connect(conn, 30.0)
    sock = Mock()
    conn.sock = sock
    _widen_after_connect(conn, 30.0)
    sock.settimeout.assert_called_once_with(30.0)


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
