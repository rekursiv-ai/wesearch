"""Shared contract tests: every header transport answers alike on the wire.

Each test drives ``fetch_curl`` and ``fetch_stdlib`` against one real loopback
HTTP server, so a rule is written once and checked against both. Behavior a
single transport alone owns lives in its own test file.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import TYPE_CHECKING, ClassVar, Protocol, cast, override

import gzip
import inspect
import socket

from curl_cffi.requests.impersonate import REAL_TARGET_MAP

import pytest

from wesearch.chrome.headers import (
    chrome_identity_headers,
    chrome_navigation_headers,
    chrome_platform,
)
from wesearch.chrome.useragents import newest_chrome_major
from wesearch.fetch.session import default_session
from wesearch.fetch.transport.curl import fetch_curl
from wesearch.fetch.transport.stdlib import fetch_stdlib
from wesearch.fetch.transport.zendriver import fetch_zendriver
from wesearch.fetch.types import Fetch, FetchSession
from wesearch.lib import zstd_compat
from wesearch.types.errors import CloudflareChallengeError, FetchError
from wesearch.types.params import ignore


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from wesearch.types.params import HttpMethod


# `path` -> (status, headers, body). A request is recorded as (method, path, headers,
# body) so a test can assert what each hop sent.
_Route = tuple[int, list[tuple[str, str]], bytes]


class _Server(BaseHTTPRequestHandler):
    routes: ClassVar[dict[str, _Route]] = {}
    seen: ClassVar[list[tuple[str, str, dict[str, str], bytes]]] = []
    raw: ClassVar[list[list[tuple[str, str]]]] = []

    def _answer(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        headers = {k.lower(): v for k, v in self.headers.items()}
        _Server.seen.append((self.command, self.path, headers, body))
        _Server.raw.append(list(self.headers.items()))
        status, out_headers, out = _Server.routes.get(self.path, (404, [], b"nope"))
        self.send_response(status)
        for name, value in out_headers:
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    do_GET = _answer  # noqa: N815 -- http.server dispatches by these exact names.
    do_POST = _answer  # noqa: N815 -- http.server dispatches by these exact names.

    @override
    def log_message(self, format: str, *args: object) -> None:
        del format, args


@pytest.fixture(autouse=True)
def offline_dns() -> None:
    """Resolve for real: these tests reach only a loopback literal.

    Overrides the package fixture, whose canned answer carries port 443 -- so a
    stdlib socket built from it dials 443 instead of the server's port.
    """


@pytest.fixture(name="base")
def base_fixture() -> Iterator[str]:
    """Serve ``_Server`` on loopback; return its base URL."""
    _Server.routes = {}
    _Server.seen = []
    _Server.raw = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Server)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class _Send(Protocol):
    """One header transport, with the defaults every test shares."""

    def __call__(
        self,
        url: str,
        *,
        method: HttpMethod = "GET",
        headers: Mapping[str, str | None] | None = None,
        body: bytes = b"",
        max_redirects: int = 10,
        on_redirect: Callable[[str], None] = ignore,
        on_response: Callable[[int, dict[str, str], str], None] = ignore,
        allow_private_hosts: bool = True,
    ) -> bytes: ...


def _bind(transport: Fetch) -> _Send:
    """Bind ``transport`` to the defaults every test shares."""

    def send(
        url: str,
        *,
        method: HttpMethod = "GET",
        headers: Mapping[str, str | None] | None = None,
        body: bytes = b"",
        max_redirects: int = 10,
        on_redirect: Callable[[str], None] = ignore,
        on_response: Callable[[int, dict[str, str], str], None] = ignore,
        allow_private_hosts: bool = True,
    ) -> bytes:
        return transport(
            url,
            method=method,
            headers=headers or {},
            body=body,
            timeout_sec=5.0,
            max_redirects=max_redirects,
            on_redirect=on_redirect,
            on_response=on_response,
            allow_private_hosts=allow_private_hosts,
        )

    return send


_TRANSPORTS: dict[str, Fetch] = {"curl": fetch_curl, "stdlib": fetch_stdlib}


@pytest.fixture(name="send", params=list(_TRANSPORTS))
def send_fixture(request: pytest.FixtureRequest) -> _Send:
    """Return one header transport, parametrized over both."""
    return _bind(_TRANSPORTS[cast("str", request.param)])


def _statuses(seen: list[int]) -> Callable[[int, dict[str, str], str], None]:
    """Return an observer appending each response status to ``seen``."""

    def observe(status: int, headers: dict[str, str], url: str) -> None:
        del headers, url
        seen.append(status)

    return observe


def test_a_get_returns_the_body(base: str, send: _Send) -> None:
    _Server.routes["/"] = (200, [], b"hello")
    assert send(f"{base}/") == b"hello"


def test_a_post_sends_its_body(base: str, send: _Send) -> None:
    _Server.routes["/form"] = (200, [], b"ok")
    send(f"{base}/form", method="POST", body=b"a=1", headers={"X-Test": "yes"})
    method, path, headers, body = _Server.seen[-1]
    assert (method, path, body) == ("POST", "/form", b"a=1")
    assert headers["x-test"] == "yes"


@pytest.mark.parametrize(
    ("encoding", "encode"),
    [("gzip", gzip.compress), ("zstd", zstd_compat.compress)],
)
def test_a_compressed_body_is_decoded(
    base: str,
    send: _Send,
    encoding: str,
    encode: Callable[[bytes], bytes],
) -> None:
    _Server.routes["/"] = (200, [("Content-Encoding", encoding)], encode(b"plain"))
    assert send(f"{base}/") == b"plain"


def test_a_redirect_is_followed_and_reported(
    base: str,
    send: _Send,
) -> None:
    _Server.routes["/a"] = (302, [("Location", "/b")], b"")
    _Server.routes["/b"] = (200, [], b"final")
    hops: list[str] = []
    seen: list[int] = []
    assert (
        send(
            f"{base}/a",
            on_redirect=hops.append,
            on_response=_statuses(seen),
        )
        == b"final"
    )
    assert hops == [f"{base}/b"]
    assert seen == [302, 200]


def test_the_redirect_cap_returns_the_3xx_body(
    base: str,
    send: _Send,
) -> None:
    _Server.routes["/a"] = (302, [("Location", "/b")], b"the 3xx body")
    assert send(f"{base}/a", max_redirects=0) == b"the 3xx body"
    assert [path for _, path, _, _ in _Server.seen] == ["/a"]


def test_a_raising_redirect_callback_aborts_the_hop(
    base: str,
    send: _Send,
) -> None:
    _Server.routes["/a"] = (302, [("Location", "/b")], b"")

    def refuse(url: str) -> None:
        raise ValueError(url)

    with pytest.raises(ValueError, match="/b"):
        send(f"{base}/a", on_redirect=refuse)
    assert [path for _, path, _, _ in _Server.seen] == ["/a"]


def test_a_303_turns_a_post_into_a_bodyless_get(
    base: str,
    send: _Send,
) -> None:
    _Server.routes["/form"] = (303, [("Location", "/done")], b"")
    _Server.routes["/done"] = (200, [], b"ok")
    send(
        f"{base}/form",
        method="POST",
        body=b"a=1",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    method, path, headers, body = _Server.seen[-1]
    assert (method, path, body) == ("GET", "/done", b"")
    assert "content-type" not in headers


def test_a_cross_origin_redirect_drops_the_credentials(
    base: str,
    send: _Send,
) -> None:
    other = base.replace("127.0.0.1", "localhost")
    _Server.routes["/a"] = (302, [("Location", f"{other}/b")], b"")
    _Server.routes["/b"] = (200, [], b"ok")
    send(f"{base}/a", headers={"Cookie": "s=1", "Authorization": "Bearer x"})
    _, _, headers, _ = _Server.seen[-1]
    assert "cookie" not in headers
    assert "authorization" not in headers


@pytest.mark.parametrize("transport", list(_TRANSPORTS.values()))
def test_a_cross_origin_redirect_sends_the_targets_cookies(
    base: str,
    transport: Fetch,
) -> None:
    other = base.replace("127.0.0.1", "localhost")
    _Server.routes["/a"] = (302, [("Location", f"{other}/b")], b"")
    _Server.routes["/b"] = (200, [], b"ok")
    asked: list[str] = []

    def stored(host: str) -> Mapping[str, str]:
        asked.append(host)
        return {"B": "2"}

    transport(
        f"{base}/a",
        headers={"Cookie": "A=1"},
        allow_private_hosts=True,
        cookies_for_host=stored,
    )
    _, _, headers, _ = _Server.seen[-1]
    assert asked == ["localhost"]
    assert headers["cookie"] == "B=2"


def test_a_redirect_without_a_location_is_an_error(
    base: str,
    send: _Send,
) -> None:
    _Server.routes["/a"] = (302, [], b"")
    with pytest.raises(FetchError, match="302"):
        send(f"{base}/a")


def test_an_error_status_raises_with_its_body(
    base: str,
    send: _Send,
) -> None:
    _Server.routes["/gone"] = (404, [("Content-Encoding", "gzip")], gzip.compress(b"x"))
    with pytest.raises(FetchError) as error:
        send(f"{base}/gone")
    assert error.value.status == 404
    assert error.value.body == b"x"


def test_a_challenge_status_is_classified(
    base: str,
    send: _Send,
) -> None:
    _Server.routes["/"] = (
        403,
        [("Content-Type", "text/html")],
        b"<html><head><title>Just a moment...</title></head></html>",
    )
    with pytest.raises(CloudflareChallengeError):
        send(f"{base}/")


def test_every_set_cookie_header_reaches_the_observer(
    base: str,
    send: _Send,
) -> None:
    _Server.routes["/"] = (
        200,
        [("Set-Cookie", "a=1, comma; Path=/"), ("Set-Cookie", "b=2")],
        b"ok",
    )
    cookies: list[str] = []

    def observe(status: int, headers: dict[str, str], url: str) -> None:
        del status, url
        cookies.append(headers["set-cookie"])

    send(f"{base}/", on_response=observe)
    assert cookies == ["a=1, comma; Path=/\nb=2"]


def test_a_caller_header_replaces_the_transport_default(
    base: str,
    send: _Send,
) -> None:
    _Server.routes["/"] = (200, [], b"ok")
    send(f"{base}/", headers={"User-Agent": "loop-paper", "X-Test": "yes"})
    _, _, headers, _ = _Server.seen[-1]
    assert headers["user-agent"] == "loop-paper"
    assert headers["x-test"] == "yes"


def test_a_none_header_deletes_the_transport_default(
    base: str,
    send: _Send,
) -> None:
    _Server.routes["/"] = (200, [], b"ok")
    send(f"{base}/")
    defaulted = set(_Server.seen[-1][2])
    send(f"{base}/", headers={"User-Agent": None, "Accept": None})
    _, _, headers, _ = _Server.seen[-1]
    assert {"user-agent", "accept"} <= defaulted
    assert "user-agent" not in headers
    assert "accept" not in headers


@pytest.mark.parametrize(
    ("name", "major"),
    [
        ("curl", int(REAL_TARGET_MAP["chrome"].removeprefix("chrome"))),
        ("stdlib", newest_chrome_major(chrome_platform("chrome"))),
    ],
)
def test_each_transport_presents_its_own_chrome_on_the_host_os(
    base: str,
    name: str,
    major: int,
) -> None:
    _Server.routes["/"] = (200, [], b"ok")
    _bind(_TRANSPORTS[name])(f"{base}/")
    _, _, headers, _ = _Server.seen[-1]
    platform = chrome_platform("chrome")
    identity = chrome_identity_headers(major=major, platform=platform)
    assert {k: headers[k.lower()] for k in identity} == identity


def test_stdlib_sends_the_whole_chrome_navigation_set(base: str) -> None:
    _Server.routes["/"] = (200, [], b"ok")
    _bind(fetch_stdlib)(f"{base}/", headers={"E": "f"})
    _, _, headers, _ = _Server.seen[-1]
    platform = chrome_platform("chrome")
    expected = chrome_navigation_headers(
        major=newest_chrome_major(platform),
        platform=platform,
    ) | {"E": "f"}
    assert {k: headers[k.lower()] for k in expected} == expected


def test_a_post_carries_its_origin(base: str, send: _Send) -> None:
    _Server.routes["/form"] = (200, [], b"ok")
    send(f"{base}/form", method="POST", body=b"a=1")
    _, _, headers, _ = _Server.seen[-1]
    assert headers["origin"] == base


def test_a_caller_overrides_the_browser_identity(base: str, send: _Send) -> None:
    _Server.routes["/"] = (200, [], b"ok")
    send(
        f"{base}/",
        headers={"User-Agent": "Android GSA", "sec-ch-ua-platform": '"Android"'},
    )
    _, _, headers, _ = _Server.seen[-1]
    assert headers["user-agent"] == "Android GSA"
    assert headers["sec-ch-ua-platform"] == '"Android"'


def test_one_cookie_header_carries_the_cookies_each_hop_set(
    base: str,
    send: _Send,
) -> None:
    _Server.routes["/a"] = (302, [("Location", "/b"), ("Set-Cookie", "s=2")], b"")
    _Server.routes["/b"] = (200, [], b"ok")
    send(f"{base}/a", headers={"cookie": "s=1; keep=3"})
    first, second = _Server.raw
    assert [v for k, v in first if k.lower() == "cookie"] == ["s=1; keep=3"]
    assert [v for k, v in second if k.lower() == "cookie"] == ["s=2; keep=3"]


def test_a_refused_connection_is_a_status_zero_fetch_error(send: _Send) -> None:
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{closed.getsockname()[1]}/"
    with pytest.raises(FetchError) as error:
        send(url)
    assert (error.value.url, error.value.status) == (url, 0)
    assert error.value.body


@pytest.mark.parametrize("transport", [fetch_curl, fetch_stdlib, fetch_zendriver])
def test_every_fetch_argument_defaults_like_the_protocol(transport: Fetch) -> None:
    def defaults(function: Callable[..., object]) -> dict[str, object]:
        return {
            name: cast(object, parameter.default)
            for name, parameter in inspect.signature(function).parameters.items()
            if name not in {"self", "url"}
        }

    shared = defaults(Fetch.__call__)
    assert inspect.Parameter.empty not in shared.values()
    assert defaults(transport) == shared


def test_a_public_only_fetch_refuses_a_loopback_address(
    base: str,
    send: _Send,
) -> None:
    with pytest.raises(ValueError, match="non-public"):
        send(f"{base}/", allow_private_hosts=False)
    assert _Server.seen == []


def test_curl_reuses_one_connection_per_host_in_a_session(base: str) -> None:
    _Server.routes["/"] = (200, [], b"ok")
    session = FetchSession(egress="203.0.113.7")
    try:
        fetch_curl(f"{base}/", allow_private_hosts=True, session=session)
        [(key, opened)] = session.curl.items()
        fetch_curl(f"{base}/", allow_private_hosts=True, session=session)
        assert list(session.curl.items()) == [(key, opened)]
        port = int(base.rsplit(":", 1)[1])
        assert key == ("203.0.113.7", "127.0.0.1", None, port)
    finally:
        session.close()
    assert session.curl == {}


def test_curl_without_a_session_reuses_the_default_connection(base: str) -> None:
    _Server.routes["/"] = (200, [], b"ok")
    fetch_curl(f"{base}/", allow_private_hosts=True)
    [opened] = default_session().curl.values()
    fetch_curl(f"{base}/", allow_private_hosts=True)
    assert list(default_session().curl.values()) == [opened]


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
