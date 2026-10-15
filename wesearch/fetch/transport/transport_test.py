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

import pytest

from wesearch.fetch.testing import zstd_compress
from wesearch.fetch.transport.curl import fetch_curl
from wesearch.fetch.transport.stdlib import fetch_stdlib
from wesearch.types.errors import CloudflareChallengeError, FetchError


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from wesearch.fetch.common import Observer
    from wesearch.types.params import Trust


# `path` -> (status, headers, body). A request is recorded as (method, path, headers,
# body) so a test can assert what each hop sent.
_Route = tuple[int, list[tuple[str, str]], bytes]


class _Server(BaseHTTPRequestHandler):
    routes: ClassVar[dict[str, _Route]] = {}
    seen: ClassVar[list[tuple[str, str, dict[str, str], bytes]]] = []

    def _answer(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        headers = {k.lower(): v for k, v in self.headers.items()}
        _Server.seen.append((self.command, self.path, headers, body))
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
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        max_redirects: int = 10,
        on_redirect: Callable[[str], None] | None = None,
        on_response: Observer | None = None,
        trust: Trust = "internal",
    ) -> bytes: ...


class _Transport(Protocol):
    """The keyword surface ``fetch_curl`` and ``fetch_stdlib`` share."""

    def __call__(
        self,
        url: str,
        *,
        method: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout_sec: float,
        max_redirects: int,
        impersonate: str,
        on_redirect: Callable[[str], None] | None,
        on_response: Observer | None,
        trust: Trust = ...,
    ) -> bytes: ...


def _bind(transport: _Transport) -> _Send:
    """Bind ``transport`` to the defaults every test shares."""

    def send(
        url: str,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        max_redirects: int = 10,
        on_redirect: Callable[[str], None] | None = None,
        on_response: Observer | None = None,
        trust: Trust = "internal",
    ) -> bytes:
        return transport(
            url,
            method=method,
            headers=headers or {},
            body=body,
            timeout_sec=5.0,
            max_redirects=max_redirects,
            impersonate="chrome",
            on_redirect=on_redirect,
            on_response=on_response,
            trust=trust,
        )

    return send


_TRANSPORTS: dict[str, _Transport] = {"curl": fetch_curl, "stdlib": fetch_stdlib}


@pytest.fixture(name="send", params=list(_TRANSPORTS))
def send_fixture(request: pytest.FixtureRequest) -> _Send:
    """Return one header transport, parametrized over both."""
    return _bind(_TRANSPORTS[cast("str", request.param)])


def _statuses(seen: list[int]) -> Observer:
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
    [("gzip", gzip.compress), ("zstd", zstd_compress)],
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


def test_untrusted_refuses_a_loopback_address(
    base: str,
    send: _Send,
) -> None:
    with pytest.raises(ValueError, match="non-public"):
        send(f"{base}/", trust="untrusted")
    assert _Server.seen == []


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
