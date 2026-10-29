"""Tests for wesearch.fetch.transport.curl.

The behavior curl shares with stdlib is in ``transport_test.py``; these mock
curl_cffi's ``Session.request`` to check what only curl does: its pooled,
pinned connections and the exact arguments it hands curl_cffi.
"""

from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import TYPE_CHECKING, cast, override
from unittest.mock import Mock, patch

import importlib
import subprocess
import sys

from curl_cffi import (
    CurlError,
    CurlOpt,
    requests as cc_requests,
)
from curl_cffi.requests import Headers

import pytest

from wesearch.chrome.headers import chrome_identity_headers, chrome_platform
from wesearch.fetch import (
    ContentParams,
    FetchSession,
    ObserveParams,
    PolicyParams,
    RequestParams,
    RetryParams,
    fetch,
)
from wesearch.fetch.session import default_session
from wesearch.fetch.transport import curl
from wesearch.fetch.transport.curl import fetch_curl
from wesearch.fetch.types import CurlKey, ValidatedHost
from wesearch.types.errors import FetchError


if TYPE_CHECKING:
    from collections.abc import Generator, Iterator, Mapping


# A public egress, so ``fetch`` never probes for one.
_EGRESS = "203.0.113.1"


def lower_headers(kw: Mapping[str, object]) -> dict[str, str]:
    """Return the request headers of a ``Session.request`` mock call, lower-cased.

    Args:
      kw: The call's keyword arguments.

    Returns:
      headers: Header names lower-cased, mapped to their values.

    """
    headers = cast(dict[str, str | None], kw["headers"])
    return {k.lower(): v for k, v in headers.items() if v is not None}


@pytest.fixture(name="session")
def session_fixture() -> Iterator[FetchSession]:
    """Return a session that already knows its egress; close it after."""
    session = FetchSession(egress=_EGRESS)
    yield session
    session.close()


def _response(
    status: int = 200,
    content: bytes | None = b"ok",
    headers: list[tuple[str, str]] | None = None,
) -> Mock:
    """Return a curl_cffi response stand-in with real ``Headers``."""
    return Mock(status_code=status, content=content, headers=Headers(headers or []))


@contextmanager
def _serve(*responses: Mock | Exception) -> Generator[Mock]:
    """Patch every curl_cffi Session to answer with ``responses`` in order."""
    request = Mock(side_effect=list(responses))
    with patch.object(cc_requests.Session, "request", request):
        yield request


def _resolves(session: FetchSession) -> dict[CurlKey, object]:
    """Return each open connection's ``RESOLVE`` pin, by connection key."""
    pins: dict[CurlKey, object] = {}
    for key, opened in session.curl.items():
        assert isinstance(opened, cc_requests.Session)
        pins[key] = cast("object", opened.curl_options.get(CurlOpt.RESOLVE))
    return pins


def _curl(
    *,
    content: ContentParams | None = None,
    retry: RetryParams | None = None,
    observe: ObserveParams | None = None,
) -> RequestParams:
    """Return request params on the curl transport."""
    return RequestParams(
        content=content or ContentParams(),
        retry=retry or RetryParams(),
        observe=observe or ObserveParams(),
        policy=PolicyParams(transport="curl"),
    )


class TestFetchCurl:
    """The arguments ``fetch_curl`` hands curl_cffi, and how it reads the answer."""

    def test_a_request_hands_curl_cffi_the_whole_contract(
        self,
        session: FetchSession,
    ) -> None:
        with _serve(_response(content=b"body")) as request:
            assert (
                fetch_curl(
                    "https://example.com/path",
                    method="POST",
                    headers={"X-Test": "yes"},
                    body=b"payload",
                    timeout_sec=9.0,
                    connect_timeout_sec=3.0,
                    max_redirects=0,
                    session=session,
                )
                == b"body"
            )
        major = int(cc_requests.impersonate.REAL_TARGET_MAP["chrome"][len("chrome") :])
        request.assert_called_once_with(
            "POST",
            "https://example.com/path",
            headers={
                **chrome_identity_headers(
                    major=major,
                    platform=chrome_platform("chrome"),
                ),
                "Origin": "https://example.com",
                "X-Test": "yes",
            },
            data=b"payload",
            impersonate="chrome",
            timeout=(3.0, 9.0),
            allow_redirects=False,
            discard_cookies=True,
        )

    def test_the_defaults_reach_curl_cffi(self, session: FetchSession) -> None:
        hop = _response(302, b"hop", [("Location", "https://example.com/")])
        with _serve(*[hop] * 12) as request:
            # Ten redirects followed; the eleventh 302 is returned as the body.
            assert fetch_curl("https://example.com/", session=session) == b"hop"
        assert request.call_count == 11
        assert request.call_args.kwargs["timeout"] == (10.0, 30.0)

    def test_a_get_sends_no_origin_and_no_data(self, session: FetchSession) -> None:
        with _serve(_response()) as request:
            fetch_curl("https://example.com/", session=session)
        assert "origin" not in lower_headers(request.call_args.kwargs)
        assert request.call_args.kwargs["data"] is None

    def test_a_none_header_reaches_curl_cffi_so_it_drops_its_own(
        self,
        session: FetchSession,
    ) -> None:
        with _serve(_response()) as request:
            fetch_curl(
                "https://example.com/",
                headers={"Accept-Encoding": None, "User-Agent": None},
                session=session,
            )
        sent = cast(dict[str, str | None], request.call_args.kwargs["headers"])
        assert sent["Accept-Encoding"] is None
        assert sent["User-Agent"] is None

    def test_a_status_of_400_raises(self, session: FetchSession) -> None:
        with _serve(_response(400, b"bad")), pytest.raises(FetchError) as error:
            fetch_curl("https://example.com/path", session=session)
        assert (error.value.status, error.value.body) == (400, b"bad")

    def test_an_error_keeps_the_url_and_an_empty_body(
        self,
        session: FetchSession,
    ) -> None:
        with _serve(_response(400, None)), pytest.raises(FetchError) as error:
            fetch_curl("https://example.com/path", session=session)
        assert error.value.url == "https://example.com/path"
        assert error.value.body == b""

    def test_a_curl_error_is_status_zero_at_the_url_it_reached(
        self,
        session: FetchSession,
    ) -> None:
        with (
            _serve(
                _response(302, b"", [("Location", "https://example.com/fail")]),
                CurlError("broken"),
            ),
            pytest.raises(FetchError) as error,
        ):
            fetch_curl("https://example.com/start", session=session)
        assert (error.value.url, error.value.status) == ("https://example.com/fail", 0)
        assert error.value.body == b"broken"

    def test_every_set_cookie_header_is_reported_one_per_line(
        self,
        session: FetchSession,
    ) -> None:
        seen: list[dict[str, str]] = []
        with _serve(_response(headers=[("Set-Cookie", "A=1"), ("Set-Cookie", "B=2")])):
            fetch_curl(
                "https://example.com/",
                on_response=lambda _s, headers, _u: seen.append(headers),
                session=session,
            )
        assert [headers["set-cookie"] for headers in seen] == ["A=1\nB=2"]

    def test_set_cookie_is_read_by_its_lower_case_name(
        self,
        session: FetchSession,
    ) -> None:
        def get_list(name: str) -> list[str]:
            return ["A=1", "B=2"] if name == "set-cookie" else []

        response = _response()
        response.headers = Mock(
            items=Mock(return_value=[("set-cookie", "A=1, B=2")]),
            get_list=Mock(side_effect=get_list),
        )
        seen: list[dict[str, str]] = []
        with _serve(response):
            fetch_curl(
                "https://example.com/",
                on_response=lambda _s, headers, _u: seen.append(headers),
                session=session,
            )
        assert [headers["set-cookie"] for headers in seen] == ["A=1\nB=2"]

    def test_a_response_without_set_cookie_reports_none(
        self,
        session: FetchSession,
    ) -> None:
        seen: list[dict[str, str]] = []
        with _serve(_response(headers=[("X-Test", "yes")])):
            fetch_curl(
                "https://example.com/",
                on_response=lambda _s, headers, _u: seen.append(headers),
                session=session,
            )
        assert seen == [{"x-test": "yes"}]

    def test_without_a_session_it_keeps_the_connection_on_the_default(self) -> None:
        with _serve(_response(), _response()):
            fetch_curl("https://example.com/")
            fetch_curl("https://example.com/")
        assert list(default_session().curl) == [
            (
                "",
                "example.com",
                ValidatedHost(host="example.com", ip="93.184.216.34"),
                443,
            ),
        ]


class TestConnections:
    """One pinned curl_cffi Session per egress, host, pin, and port."""

    def test_a_public_only_request_pins_the_ip_it_validated(
        self,
        session: FetchSession,
    ) -> None:
        with _serve(_response()):
            fetch_curl("https://example.com/x", session=session)
        assert _resolves(session) == {
            (
                _EGRESS,
                "example.com",
                ValidatedHost(host="example.com", ip="93.184.216.34"),
                443,
            ): ["example.com:443:93.184.216.34"],
        }

    @pytest.mark.parametrize(
        ("url", "resolve"),
        [
            ("https://example.com:8443/x", "example.com:8443:93.184.216.34"),
            ("http://example.com/x", "example.com:80:93.184.216.34"),
        ],
    )
    def test_the_pin_names_the_real_port(
        self,
        session: FetchSession,
        url: str,
        resolve: str,
    ) -> None:
        """``RESOLVE`` is per host:port, so the pin must name the real port.

        An entry naming 443 does not apply to a request on 8443 -- curl falls
        back to its own resolution and the connection is unpinned, with nothing
        reporting that the guard lapsed.
        """
        with _serve(_response()):
            fetch_curl(url, session=session)
        assert list(_resolves(session).values()) == [[resolve]]

    def test_an_ipv6_pin_is_bracketed(self, session: FetchSession) -> None:
        # `curl` mis-parses an unbracketed IPv6: its colons collide with host:port.
        with (
            patch.object(
                curl,
                "pinned_host",
                return_value=ValidatedHost(host="v6.example", ip="2606:4700:20::1"),
            ),
            _serve(_response()),
        ):
            fetch_curl("https://v6.example/", session=session)
        assert list(_resolves(session).values()) == [
            ["v6.example:443:[2606:4700:20::1]"],
        ]

    def test_allowing_private_hosts_skips_the_pin(self, session: FetchSession) -> None:
        with _serve(_response()):
            fetch_curl(
                "http://127.0.0.1:8080/",
                allow_private_hosts=True,
                session=session,
            )
        assert _resolves(session) == {(_EGRESS, "127.0.0.1", None, 8080): None}

    def test_private_hosts_are_refused_by_default(self, session: FetchSession) -> None:
        with _serve() as request, pytest.raises(ValueError, match="non-public"):
            fetch_curl("http://127.0.0.1/", session=session)
        request.assert_not_called()
        assert session.curl == {}

    def test_a_cross_host_redirect_opens_a_second_pinned_connection(
        self,
        session: FetchSession,
    ) -> None:
        with _serve(
            _response(302, b"", [("Location", "https://other.com/final")]),
            _response(content=b"done"),
        ):
            assert fetch_curl("https://example.com/start", session=session) == b"done"
        assert list(_resolves(session).values()) == [
            ["example.com:443:93.184.216.34"],
            ["other.com:443:93.184.216.34"],
        ]

    def test_a_same_origin_redirect_reuses_the_connection(
        self,
        session: FetchSession,
    ) -> None:
        with _serve(
            _response(302, b"", [("Location", "/next")]),
            _response(),
        ):
            fetch_curl("https://example.com/start", session=session)
        assert len(session.curl) == 1

    def test_every_key_component_separates_connections(
        self,
        session: FetchSession,
    ) -> None:
        def connect(url: str) -> object:
            return curl._connection(session, url, allow_private_hosts=False)

        first = connect("https://a.example.com:8443/")
        assert connect("https://a.example.com:8443/other") is first
        assert connect("https://b.example.com:8443/") is not first
        assert connect("https://a.example.com/") is not first
        session.egress = "198.51.100.2"
        assert connect("https://a.example.com:8443/") is not first
        assert len(session.curl) == 4

    def test_opening_a_connection_holds_the_session_lock(
        self,
        session: FetchSession,
    ) -> None:
        opened: list[object] = []
        worker = Thread(
            target=lambda: opened.append(
                curl._connection(
                    session,
                    "https://a.example/",
                    allow_private_hosts=False,
                ),
            ),
        )
        with session.lock:
            worker.start()
            worker.join(timeout=0.05)
            assert worker.is_alive()
            assert opened == []
        worker.join()
        assert opened == list(session.curl.values())

    def test_close_closes_every_connection(self, session: FetchSession) -> None:
        with _serve(
            _response(302, b"", [("Location", "https://other.com/")]),
            _response(),
        ):
            fetch_curl("https://example.com/", session=session)
        opened = list(session.curl.values())
        with patch.object(cc_requests.Session, "close", autospec=True) as close:
            session.close()
        assert [call.args[0] for call in close.call_args_list] == opened
        assert session.curl == {}


class TestThroughFetch:
    """``fetch`` on the curl transport: cookies, redirects, and retries."""

    def test_a_get_impersonates_chrome_and_returns_the_body(
        self,
        session: FetchSession,
    ) -> None:
        with _serve(_response(content=b"hello")) as request:
            assert fetch("https://example.com", session=session) == b"hello"
        kwargs = request.call_args.kwargs
        assert kwargs["impersonate"] == "chrome"
        assert kwargs["allow_redirects"] is False
        assert kwargs["discard_cookies"] is True

    def test_caller_cookies_ride_one_cookie_header(self, session: FetchSession) -> None:
        with _serve(_response()) as request:
            fetch(
                "https://example.com",
                request=_curl(
                    content=ContentParams(
                        headers={"cookie": "a=1"},
                        cookies={"b": "2", "CONSENT": "YES+"},
                    ),
                ),
                session=session,
            )
        kwargs = request.call_args.kwargs
        sent = cast(dict[str, str], kwargs["headers"])
        assert [v for k, v in sent.items() if k.lower() == "cookie"] == [
            "a=1; b=2; CONSENT=YES+",
        ]
        assert "cookies" not in kwargs
        assert kwargs["discard_cookies"] is True

    def test_a_post_redirected_cross_origin_names_the_new_origin(
        self,
        session: FetchSession,
    ) -> None:
        with _serve(
            _response(307, b"", [("Location", "https://b.com/land")]),
            _response(content=b"done"),
        ) as request:
            fetch(
                "https://a.com/submit",
                request=_curl(content=ContentParams(method="POST", data={"x": "1"})),
                session=session,
            )
        assert lower_headers(request.call_args.kwargs)["origin"] == "https://b.com"

    def test_the_redirect_cap_returns_the_last_3xx_body(
        self,
        session: FetchSession,
    ) -> None:
        seen: list[str] = []
        with _serve(
            _response(302, b"", [("Location", "https://a.com/1")]),
            _response(302, b"", [("Location", "https://a.com/2")]),
            _response(302, b"final 3xx", [("Location", "/3")]),
        ):
            body = fetch(
                "https://a.com/start",
                request=_curl(
                    retry=RetryParams(max_redirects=2),
                    observe=ObserveParams(on_redirect=seen.append),
                ),
                session=session,
            )
        assert body == b"final 3xx"
        assert seen == ["https://a.com/1", "https://a.com/2"]

    def test_max_redirects_zero_returns_the_3xx_body(
        self,
        session: FetchSession,
    ) -> None:
        with _serve(
            _response(302, b"redirect body", [("Location", "https://example.com/o")]),
        ) as request:
            body = fetch(
                "https://example.com",
                request=_curl(retry=RetryParams(max_redirects=0)),
                session=session,
            )
        assert body == b"redirect body"
        assert request.call_count == 1

    def test_a_challenge_status_raises_with_its_body(
        self,
        session: FetchSession,
    ) -> None:
        # ``classify_challenge`` matches on markup, so the error body must be the
        # page curl_cffi already decoded.
        html = b"<!DOCTYPE html><html>Just a moment...</html>"
        with (
            _serve(_response(403, html, [("server", "cloudflare")])),
            pytest.raises(FetchError) as error,
        ):
            fetch("https://example.com", request=_curl(), session=session)
        assert (error.value.status, error.value.body) == (403, html)

    def test_raw_headers_keep_a_caller_user_agent(self, session: FetchSession) -> None:
        with _serve(_response()) as request:
            fetch(
                "https://example.com",
                request=_curl(
                    content=ContentParams(
                        headers={"User-Agent": "custom"},
                        raw_headers=True,
                    ),
                ),
                session=session,
            )
        sent = lower_headers(request.call_args.kwargs)
        assert sent["user-agent"] == "custom"
        assert "content-type" not in sent

    @pytest.mark.parametrize(
        "content",
        [
            ContentParams(method="POST", data={"x": "1"}),
            ContentParams(method="POST", json={"x": 1}),
        ],
    )
    def test_a_303_turns_the_post_into_a_bodyless_get(
        self,
        session: FetchSession,
        content: ContentParams,
    ) -> None:
        with _serve(
            _response(303, b"", [("Location", "https://example.com/result")]),
            _response(content=b"got it"),
        ) as request:
            body = fetch(
                "https://example.com/submit",
                request=_curl(content=content),
                session=session,
            )
        assert body == b"got it"
        first, second = request.call_args_list
        assert "content-type" in lower_headers(first.kwargs)
        assert second.args[0] == "GET"
        assert second.kwargs["data"] is None
        assert "content-type" not in lower_headers(second.kwargs)

    def test_a_connection_error_is_retried(self, session: FetchSession) -> None:
        with (
            _serve(CurlError("connection refused"), _response(content=b"ok")),
            patch("wesearch.fetch.fetch.time.sleep"),
        ):
            assert (
                fetch(
                    "https://example.com",
                    request=_curl(retry=RetryParams(retries=1)),
                    session=session,
                )
                == b"ok"
            )

    def test_a_persistent_connection_error_raises_after_the_retries(
        self,
        session: FetchSession,
    ) -> None:
        with (
            _serve(*[CurlError("refused")] * 3),
            patch("wesearch.fetch.fetch.time.sleep"),
            pytest.raises(FetchError) as error,
        ):
            fetch(
                "https://example.com",
                request=_curl(retry=RetryParams(retries=2)),
                session=session,
            )
        assert error.value.status == 0
        assert b"refused" in error.value.body


class TestRealWire:
    """Only a REAL curl_cffi call reveals the headers its impersonation injects.

    A request without a User-Agent is refused by UA-gated APIs (GitHub's REST
    API 403s it). Hermetic: a loopback server echoes the request headers.
    """

    def test_curl_sends_a_whole_chrome_identity(self, session: FetchSession) -> None:
        seen: dict[str, str] = {}

        class _Echo(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                seen.update((k.lower(), v) for k, v in self.headers.items())
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            @override
            def log_message(self, format: str, *args: object) -> None:
                del format, args

        server = HTTPServer(("127.0.0.1", 0), _Echo)
        thread = Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            body = fetch(
                f"http://127.0.0.1:{server.server_address[1]}/",
                # The oracle IS loopback, so this test writes its own URL.
                request=RequestParams(
                    policy=PolicyParams(transport="curl", allow_private_hosts=True),
                ),
                session=session,
            )
        finally:
            server.server_close()
            thread.join(timeout=5)
        assert body == b"ok"
        assert "chrome" in seen["user-agent"].lower()
        assert seen["accept"]
        assert seen["sec-fetch-mode"] == "navigate"


def test_curl_is_available_where_it_imports() -> None:
    curl.curl_available.cache_clear()
    try:
        assert curl.curl_available()
    finally:
        curl.curl_available.cache_clear()


def test_curl_is_unavailable_where_its_native_library_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", broken)
    curl.curl_available.cache_clear()
    try:
        assert not curl.curl_available()
        assert not curl.curl_available()
    finally:
        curl.curl_available.cache_clear()
    assert caplog.text.count("curl_cffi is unavailable") == 1


def test_fetch_imports_and_resolves_stdlib_without_curl_cffi() -> None:
    program = (
        "import sys\n"
        "sys.modules['curl_cffi'] = None\n"
        "import typing\n"
        "from wesearch.fetch import ContentParams, resolve_transport\n"
        "typing.get_type_hints(ContentParams)\n"
        "print(resolve_transport('curl'))\n"
    )
    done = subprocess.run(  # noqa: S603 -- fixed interpreter and program.
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "stdlib"


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
