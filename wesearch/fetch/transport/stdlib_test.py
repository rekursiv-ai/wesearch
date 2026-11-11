"""Unit tests for stdlib requests, response handling, and pinned connections."""

from __future__ import annotations

from functools import partial
from http import client
from typing import cast
from unittest.mock import Mock, call, patch

import gzip
import socket
import ssl

import pytest

from wesearch.chrome.headers import (
    chrome_headers_for_google,
    chrome_navigation_headers,
    chrome_user_agent,
)
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
        self.bodies: list[bytes | None] = []
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
        self.bodies.append(body)
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
    return Mock(
        spec=client.HTTPResponse,
        status=status,
        read=Mock(side_effect=[body, b""]),
        isclosed=Mock(return_value=False),
        getheaders=Mock(return_value=headers or []),
    )


@pytest.mark.parametrize(
    ("scheme", "port", "host", "defaults"),
    [
        ("https", 8443, "example.com:8443", False),
        ("http", 80, "example.com", True),
        ("https", 443, "example.com", True),
    ],
)
def test_a_public_only_request_dials_the_pin_and_names_the_host_first(
    scheme: str,
    port: int,
    host: str,
    defaults: bool,
) -> None:
    response = _response()
    response.read = Mock(side_effect=[b"o", b"k", b""])
    conn = _Conn(response)
    pin = ValidatedHost(host="example.com", ip="93.184.216.34")
    fetch = fetch_stdlib if defaults else partial(fetch_stdlib, connect_timeout_sec=2.5)
    with (
        patch("wesearch.fetch.transport.stdlib.pinned_host", return_value=pin),
        patch(
            "wesearch.fetch.transport.stdlib._connect",
            return_value=conn,
        ) as opened,
    ):
        assert (
            fetch(
                f"{scheme}://example.com:{port}/p?q=1",
                headers={"Accept": "*/*"},
            )
            == b"ok"
        )
    opened.assert_called_once_with(
        scheme,
        "example.com",
        port=port,
        ip="93.184.216.34",
        connect_timeout_sec=10.0 if defaults else 2.5,
    )
    [(method, path, headers)] = conn.requests
    assert (method, path) == ("GET", "/p?q=1")
    assert next(iter(headers)) == "Host"
    assert headers["Host"] == host
    assert headers["Accept"] == "*/*"
    assert conn.bodies == [None]
    assert conn.closed


@pytest.mark.parametrize("host", ["www.google.com", "example.com"])
@pytest.mark.parametrize("content_type", ["", "text/plain"])
def test_a_post_forwards_its_body_headers_and_navigation_policy(
    host: str,
    content_type: str,
) -> None:
    supplied = {"X-Ignore": None, "X-Key": "key"}
    if content_type:
        supplied |= {
            "cOnTeNt-TyPe": content_type,
            "User-Agent": "Android GSA",
            "sec-ch-ua-platform": '"Android"',
            "Accept": None,
        }
    conn = _Conn(_response())
    with (
        patch(
            "wesearch.fetch.transport.stdlib._connect",
            return_value=conn,
        ) as opened,
        patch(
            "wesearch.fetch.transport.stdlib.chrome_platform",
            return_value="macOS",
        ),
        patch(
            "wesearch.fetch.transport.stdlib.newest_chrome_major",
            return_value=151,
        ) as newest,
        patch(
            "wesearch.fetch.transport.stdlib.chrome_navigation_headers",
            wraps=chrome_navigation_headers,
        ) as navigation,
    ):
        assert (
            fetch_stdlib(
                f"https://{host}",
                method="POST",
                body=b"sent",
                headers=supplied,
                allow_private_hosts=True,
            )
            == b"ok"
        )
    assert conn.bodies == [b"sent"]
    [(method, path, headers)] = conn.requests
    sent = {name.lower(): value for name, value in headers.items()}
    assert (method, path) == ("POST", "/")
    assert sent["origin"] == f"https://{host}"
    assert sent["x-key"] == "key"
    navigation_headers = {
        name.lower(): value
        for name, value in chrome_navigation_headers(
            major=151,
            platform="macOS",
            method="POST",
            content_type=content_type,
            origin=f"https://{host}",
        ).items()
    }
    if content_type:
        navigation_headers.pop("accept")
        navigation_headers["user-agent"] = "Android GSA"
        navigation_headers["sec-ch-ua-platform"] = '"Android"'
    assert {name: sent[name] for name in navigation_headers} == navigation_headers
    if content_type:
        assert sent["user-agent"] == "Android GSA"
        assert sent["sec-ch-ua-platform"] == '"Android"'
        assert "accept" not in sent
        assert sent["content-type"] == content_type
        assert navigation.call_args.kwargs["content_type"] == content_type
    else:
        assert sent["user-agent"] == chrome_user_agent(151, "macOS")
        assert sent["accept"]
        assert "content-type" not in sent
    assert "x-ignore" not in sent
    assert navigation.call_args.kwargs["method"] == "POST"
    assert opened.call_args.kwargs["ip"] == ""
    newest.assert_called_once_with("macOS")
    expected = chrome_headers_for_google(major=151, platform="macOS")
    if host == "www.google.com":
        assert {name: sent[name] for name in expected} == expected
    else:
        assert not expected.keys() & sent.keys()
    assert conn.closed


def test_an_already_closed_response_needs_no_read() -> None:
    response = _response()
    response.isclosed = Mock(return_value=True)
    read = Mock(side_effect=AssertionError("closed response read"))
    response.read = read
    conn = _Conn(response)
    with patch("wesearch.fetch.transport.stdlib._connect", return_value=conn):
        assert fetch_stdlib("https://example.com/", allow_private_hosts=True) == b""
    read.assert_not_called()
    assert conn.closed


def test_a_compressed_success_is_decoded() -> None:
    conn = _Conn(
        _response(body=gzip.compress(b"plain"), headers=[("Content-Encoding", "gzip")]),
    )
    with patch("wesearch.fetch.transport.stdlib._connect", return_value=conn):
        assert (
            fetch_stdlib("https://example.com/", allow_private_hosts=True) == b"plain"
        )
    assert conn.closed


def test_a_cross_host_redirect_reopens_and_re_pins() -> None:
    first = _Conn(
        _response(302, b"", [("Location", "https://other.example/next?x=1")]),
    )
    second = _Conn(_response(body=b"final"))
    pin = Mock(
        side_effect=[
            ValidatedHost(host="example.com", ip="93.184.216.34"),
            ValidatedHost(host="other.example", ip="93.184.216.35"),
        ],
    )
    observed, response, redirect = Mock(), Mock(), Mock()
    observed.attach_mock(response, "response")
    observed.attach_mock(redirect, "redirect")
    cookies = Mock(return_value={"B": "2"})
    with (
        patch("wesearch.fetch.transport.stdlib.pinned_host", pin),
        patch(
            "wesearch.fetch.transport.stdlib._connect",
            side_effect=[first, second],
        ) as opened,
    ):
        assert (
            fetch_stdlib(
                "https://example.com/start",
                headers={"host": "x", "Cookie": "A=1", "Authorization": "Bearer x"},
                on_response=response,
                on_redirect=redirect,
                cookies_for_host=cookies,
            )
            == b"final"
        )
    assert first.closed
    assert second.closed
    assert pin.call_args_list == [
        call("https://example.com/start", allow_private_hosts=False),
        call("https://other.example/next?x=1", allow_private_hosts=False),
    ]
    assert observed.mock_calls == [
        (
            "response",
            (
                302,
                {"location": "https://other.example/next?x=1"},
                "https://example.com/start",
            ),
            {},
        ),
        ("redirect", ("https://other.example/next?x=1",), {}),
        ("response", (200, {}, "https://other.example/next?x=1"), {}),
    ]
    cookies.assert_called_once_with("other.example")
    assert opened.call_args.kwargs["ip"] == "93.184.216.35"
    [(_, path, headers)] = second.requests
    assert path == "/next?x=1"
    assert headers["Host"] == "other.example"
    assert headers["Cookie"] == "B=2"
    assert not any(name.lower() == "authorization" for name in headers)


@pytest.mark.parametrize(("defaults", "redirects"), [(False, 0), (True, 10)])
def test_a_redirect_budget_returns_the_body_without_another_connection(
    defaults: bool,
    redirects: int,
) -> None:
    conn = _Conn(
        *(
            _response(302, b"hop", [("Location", "/next")])
            for _ in range(redirects + 1)
        ),
    )
    redirect, response = Mock(), Mock()
    fetch = fetch_stdlib if defaults else partial(fetch_stdlib, max_redirects=0)
    with patch(
        "wesearch.fetch.transport.stdlib._connect",
        side_effect=[conn] * (redirects + 1)
        + [AssertionError("followed beyond redirect budget")],
    ) as opened:
        assert (
            fetch(
                "https://example.com/start",
                on_redirect=redirect,
                on_response=response,
                allow_private_hosts=True,
            )
            == b"hop"
        )
    assert opened.call_count == redirects + 1
    assert (
        response.call_args_list
        == [
            call(302, {"location": "/next"}, "https://example.com/start"),
        ]
        + [call(302, {"location": "/next"}, "https://example.com/next")] * redirects
    )
    assert redirect.call_args_list == [call("https://example.com/next")] * redirects
    assert conn.closed


def test_a_redirect_refusal_escapes_after_the_connection_closes() -> None:
    conn = _Conn(_response(302, b"", [("Location", "/next")]))
    refused = ValueError("refused")

    def refuse(url: str) -> None:
        assert url == "https://example.com/next"
        assert conn.closed
        raise refused

    with (
        patch(
            "wesearch.fetch.transport.stdlib._connect",
            return_value=conn,
        ) as opened,
        pytest.raises(ValueError, match="refused") as error,
    ):
        fetch_stdlib(
            "https://example.com/start",
            on_redirect=refuse,
            allow_private_hosts=True,
        )
    assert error.value is refused
    opened.assert_called_once()
    assert conn.closed


def test_the_read_budget_widens_only_once_connected() -> None:
    settimeout = Mock()
    conn = _Conn(_response(), sock=Mock(settimeout=settimeout))
    with (
        patch("wesearch.fetch.transport.stdlib._connect", return_value=conn),
        patch("wesearch.fetch.transport.stdlib.time.monotonic", return_value=0.0),
    ):
        fetch_stdlib(
            "http://127.0.0.1/",
            connect_timeout_sec=1.0,
            allow_private_hosts=True,
        )
    assert settimeout.call_args_list[0] == call(30.0)


def test_a_slow_body_fails_at_the_deadline_not_per_read() -> None:
    # Each read is under the budget, but together they pass it.
    clock = [0.0]

    def trickle(amt: int | None) -> bytes:
        assert isinstance(amt, int)
        assert amt > 0
        clock[0] += 4.0
        return b"x"

    read = Mock(side_effect=trickle)
    resp = _response()
    resp.read = read
    settimeout = Mock()
    conn = _Conn(resp, sock=Mock(settimeout=settimeout))
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
    cause = error.value.__cause__
    assert isinstance(cause, TimeoutError)
    assert "10" in str(cause)
    assert "response" in str(cause).casefold()
    assert error.value.body == str(cause).encode()
    assert read.call_count == 3
    assert settimeout.call_args_list == [call(10.0), call(10.0), call(6.0), call(2.0)]
    assert conn.closed


@pytest.mark.parametrize("elapsed", [2.5, 3.0])
def test_the_callers_read_deadline_includes_its_exact_boundary(elapsed: float) -> None:
    read = Mock(side_effect=[b"ok", b""])
    response = _response()
    response.read = read
    conn = _Conn(response)
    with (
        patch("wesearch.fetch.transport.stdlib._connect", return_value=conn),
        patch(
            "wesearch.fetch.transport.stdlib.time.monotonic",
            side_effect=[0.0, elapsed, elapsed, elapsed],
        ),
    ):
        if elapsed == 3.0:
            with pytest.raises(FetchError) as error:
                fetch_stdlib(
                    "https://example.com/",
                    timeout_sec=3.0,
                    allow_private_hosts=True,
                )
            assert error.value.status == 0
            read.assert_not_called()
        else:
            assert (
                fetch_stdlib(
                    "https://example.com/",
                    timeout_sec=3.0,
                    allow_private_hosts=True,
                )
                == b"ok"
            )
    assert conn.closed


@pytest.mark.parametrize("status", [400, 404])
def test_an_error_status_raises_a_fetch_error_with_the_body(status: int) -> None:
    conn = _Conn(_response(status, b"nope", [("X-Reason", "refused")]))
    classified = FetchError(
        "http://127.0.0.1/gone",
        status,
        {"x-reason": "refused"},
        b"nope",
    )
    with (
        patch("wesearch.fetch.transport.stdlib._connect", return_value=conn),
        patch(
            "wesearch.fetch.transport.stdlib.classify_http_error",
            return_value=classified,
        ) as classify,
        pytest.raises(FetchError) as error,
    ):
        fetch_stdlib("http://127.0.0.1/gone", allow_private_hosts=True)
    assert error.value is classified
    classify.assert_called_once_with(
        "http://127.0.0.1/gone",
        status,
        {"x-reason": "refused"},
        b"nope",
    )
    assert (error.value.url, error.value.status) == ("http://127.0.0.1/gone", status)
    assert error.value.headers == {"x-reason": "refused"}
    assert error.value.body == b"nope"
    assert conn.closed


def test_an_error_body_is_decompressed_before_it_is_reported() -> None:
    conn = _Conn(
        _response(
            429,
            gzip.compress(b"slow down"),
            [("Content-Encoding", "gzip")],
        ),
    )
    with (
        patch("wesearch.fetch.transport.stdlib._connect", return_value=conn),
        pytest.raises(FetchError) as error,
    ):
        fetch_stdlib("http://127.0.0.1/", allow_private_hosts=True)
    assert error.value.status == 429
    assert error.value.body == b"slow down"


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
    assert error.value.headers == {}
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
    assert conn.timeout == 1.0


def test_an_unpinned_connection_dials_the_hostname() -> None:
    https = _connect("https", "example.com", port=8443, ip="", connect_timeout_sec=5.0)
    http = _connect("http", "example.com", port=8080, ip="", connect_timeout_sec=5.0)
    assert type(https) is client.HTTPSConnection
    assert (https.host, https.port) == ("example.com", 8443)
    assert https.timeout == 5.0
    assert type(http) is client.HTTPConnection
    assert (http.host, http.port) == ("example.com", 8080)
    assert http.timeout == 5.0


def test_the_pinned_connection_wraps_its_socket_for_the_hostname() -> None:
    context = cast("ssl.SSLContext", Mock(spec=ssl.SSLContext))
    raw = Mock(spec=socket.socket)
    wrapped = Mock(spec=ssl.SSLSocket)

    def connect(self: client.HTTPConnection) -> None:
        self.sock = raw

    with (
        patch(
            "wesearch.fetch.transport.stdlib.ssl.create_default_context",
            return_value=context,
        ),
        patch.object(client.HTTPConnection, "connect", connect),
        patch.object(context, "wrap_socket", return_value=wrapped) as wrap,
    ):
        conn = _PinnedHTTPSConnection(
            "93.184.216.34",
            port=None,
            hostname="example.com",
            timeout=1.0,
        )
        conn.connect()
    wrap.assert_called_once_with(raw, server_hostname="example.com")
    assert conn.sock is wrapped


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
