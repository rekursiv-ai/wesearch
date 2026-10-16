"""Tests for what only the stdlib transport does: its sockets and its pins.

The behavior it shares with curl is in ``transport_test.py``.
"""

from __future__ import annotations

from http import client
from typing import cast
from unittest.mock import Mock, call, patch

import socket
import ssl

import pytest

from wesearch.fetch.transport.stdlib import (
    _connect,
    _PinnedHTTPSConnection,
    fetch_stdlib,
)
from wesearch.fetch.types import ValidatedHost
from wesearch.types.errors import FetchError


class _Conn:
    """An ``http.client`` connection that records requests and replays responses."""

    def __init__(self, *responses: Mock | Exception, sock: Mock | None = None) -> None:
        self.sock = sock
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
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        self.closed = True


def _response(
    status: int = 200,
    body: bytes = b"ok",
    headers: list[tuple[str, str]] | None = None,
) -> Mock:
    resp = Mock(spec=client.HTTPResponse)
    resp.status = status
    resp.read.side_effect = [body, b""]
    resp.isclosed.return_value = False
    resp.getheaders.return_value = headers or []
    return resp


def test_a_public_only_request_dials_the_pin_and_names_the_host_first() -> None:
    conn = _Conn(_response())
    pin = ValidatedHost(host="example.com", ip="93.184.216.34")
    with (
        patch("wesearch.fetch.transport.stdlib.pinned_host", return_value=pin),
        patch(
            "wesearch.fetch.transport.stdlib._connect",
            return_value=conn,
        ) as opened,
    ):
        fetch_stdlib("https://example.com:8443/p?q=1", headers={"Accept": "*/*"})
    assert opened.call_args.kwargs["ip"] == "93.184.216.34"
    assert opened.call_args.kwargs["port"] == 8443
    [(method, path, headers)] = conn.requests
    assert (method, path) == ("GET", "/p?q=1")
    assert next(iter(headers)) == "Host"
    assert headers["Host"] == "example.com:8443"
    assert headers["Accept"] == "*/*"
    assert conn.closed


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

    def pin(url: str, *, allow_private_hosts: bool) -> ValidatedHost:
        del url
        assert not allow_private_hosts
        return next(pins)

    with (
        patch("wesearch.fetch.transport.stdlib.pinned_host", pin),
        patch(
            "wesearch.fetch.transport.stdlib._connect",
            side_effect=[first, second],
        ) as opened,
    ):
        assert fetch_stdlib("https://example.com/start", headers={"host": "x"}) == (
            b"final"
        )
    assert first.closed
    assert opened.call_args.kwargs["ip"] == "93.184.216.35"
    [(_, path, headers)] = second.requests
    assert path == "/next?x=1"
    assert headers["Host"] == "other.example"


def test_the_read_budget_widens_only_once_connected() -> None:
    sock = Mock()
    conn = _Conn(_response(), sock=sock)
    with (
        patch("wesearch.fetch.transport.stdlib._connect", return_value=conn),
        patch("wesearch.fetch.transport.stdlib.time.monotonic", return_value=0.0),
    ):
        fetch_stdlib(
            "http://127.0.0.1/",
            timeout_sec=30.0,
            connect_timeout_sec=1.0,
            allow_private_hosts=True,
        )
    assert sock.settimeout.call_args_list[0] == call(30.0)


def test_a_slow_body_fails_at_the_deadline_not_per_read() -> None:
    # Each read is under the budget, but together they pass it.
    clock = [0.0]

    def trickle(amt: int) -> bytes:
        del amt
        clock[0] += 4.0
        return b"x"

    resp = _response()
    resp.read.side_effect = trickle
    sock = Mock()
    conn = _Conn(resp, sock=sock)
    with (
        patch("wesearch.fetch.transport.stdlib._connect", return_value=conn),
        patch(
            "wesearch.fetch.transport.stdlib.time.monotonic",
            side_effect=lambda: clock[0],
        ),
        pytest.raises(FetchError) as error,
    ):
        fetch_stdlib("http://127.0.0.1/", timeout_sec=10.0, allow_private_hosts=True)
    assert error.value.status == 0
    assert resp.read.call_count == 3
    assert [c.args[0] for c in sock.settimeout.call_args_list] == [10.0, 10.0, 6.0, 2.0]
    assert conn.closed


@pytest.mark.parametrize(
    "failure",
    [OSError("connection refused"), client.RemoteDisconnected("hung up")],
)
def test_a_failed_connection_is_a_status_zero_fetch_error(failure: Exception) -> None:
    conn = _Conn(failure)
    with (
        patch("wesearch.fetch.transport.stdlib._connect", return_value=conn),
        pytest.raises(FetchError) as error,
    ):
        fetch_stdlib("http://127.0.0.1/x", allow_private_hosts=True)
    assert (error.value.url, error.value.status) == ("http://127.0.0.1/x", 0)
    assert error.value.body == str(failure).encode()
    assert error.value.__cause__ is failure
    assert conn.closed


def test_a_pinned_https_connection_verifies_the_hostname() -> None:
    conn = _connect(
        "https",
        "example.com",
        port=8443,
        ip="2606:4700::1",
        connect_timeout_sec=1.0,
    )
    assert isinstance(conn, _PinnedHTTPSConnection)
    assert (conn.host, conn.port) == ("2606:4700::1", 8443)
    assert conn.timeout == 1.0
    assert conn._hostname == "example.com"


def test_a_pinned_http_connection_dials_the_ip() -> None:
    conn = _connect(
        "http",
        "example.com",
        port=None,
        ip="93.184.216.34",
        connect_timeout_sec=1.0,
    )
    assert type(conn) is client.HTTPConnection
    assert conn.host == "93.184.216.34"


def test_an_unpinned_connection_dials_the_hostname() -> None:
    https = _connect("https", "example.com", port=None, ip="", connect_timeout_sec=5.0)
    http = _connect("http", "example.com", port=8080, ip="", connect_timeout_sec=5.0)
    assert type(https) is client.HTTPSConnection
    assert https.host == "example.com"
    assert https.timeout == 5.0
    assert type(http) is client.HTTPConnection
    assert (http.host, http.port) == ("example.com", 8080)


def test_the_pinned_connection_wraps_its_socket_for_the_hostname() -> None:
    conn = _PinnedHTTPSConnection(
        "93.184.216.34",
        port=None,
        hostname="example.com",
        timeout=1.0,
    )
    context = cast("ssl.SSLContext", Mock(spec=ssl.SSLContext))
    conn._context = context
    raw = Mock(spec=socket.socket)

    def connect(self: client.HTTPConnection) -> None:
        self.sock = raw

    with (
        patch.object(client.HTTPConnection, "connect", connect),
        patch.object(context, "wrap_socket") as wrap,
    ):
        conn.connect()
    wrap.assert_called_once_with(raw, server_hostname="example.com")
    assert conn.sock is wrap.return_value


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
