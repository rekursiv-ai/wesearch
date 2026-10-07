"""Tests for wesearch.fetch."""

from __future__ import annotations

from http.cookiejar import CookieJar
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import cast, override
from unittest.mock import Mock, patch

import importlib
import io
import warnings

from curl_cffi import (
    CurlError,
    CurlInfo,
    CurlOpt,
    requests as cc_requests,
)
from curl_cffi.requests import Response

import pytest

from wesearch.fetch import (
    ContentParams,
    ObserveParams,
    PolicyParams,
    RequestParams,
    RetryParams,
    ValidatedHost,
    fetch,
)
from wesearch.fetch.testing import (
    StubCookies,
    StubSession,
    const_curl_session,
)
from wesearch.fetch.transport import curl
from wesearch.fetch.transport.curl import (
    _CurlLoop,
    _jar_set,
    _registrable_domain,
    _SessionKey,
    close_curl_session,
    close_curl_sessions_except,
    curl_session,
    fetch_curl,
    seed_session_jar,
    set_session_cookies,
)
from wesearch.lib.codec import from_plain
from wesearch.types.errors import (
    FetchError,
)


fetch_mod = importlib.import_module("wesearch.fetch.fetch")


# Captured before any fixture stubs it, so the pool-locking tests can invoke the
# real curl_session.
_REAL_CURL_SESSION = curl.curl_session


class TestRegistrableDomain:
    """The session pool keys on eTLD+1 so sibling subdomains coalesce onto one.

    Connection (a browser's HTTP/2 coalescing); ``www.google.com`` and
    ``scholar.google.com`` must map to the same key.
    """

    def test_subdomains_share_registrable_domain(self) -> None:
        assert _registrable_domain("www.google.com") == "google.com"
        assert _registrable_domain("scholar.google.com") == "google.com"

    def test_bare_domain_unchanged(self) -> None:
        assert _registrable_domain("google.com") == "google.com"

    def test_single_label_unchanged(self) -> None:
        assert _registrable_domain("localhost") == "localhost"
        # Two letters, like a ccTLD, with no second-level label to look at.
        assert _registrable_domain("db") == "db"

    def test_cc_second_level_tld_keeps_three_labels(self) -> None:
        assert _registrable_domain("a.example.co.uk") == "example.co.uk"
        assert _registrable_domain("example.co.uk") == "example.co.uk"
        assert _registrable_domain("x.y.example.com.au") == "example.com.au"

    def test_plain_gtld_keeps_two_labels(self) -> None:
        assert _registrable_domain("deep.sub.example.org") == "example.org"


class TestFetchCurlBackend:
    """The curl_cffi backend: SSRF pinning, redirects, decompression, errors.

    All tests mock at the curl boundary -- either ``curl_cffi.requests.request``
    (high-level path, no ``validated_hosts``) or the ``curl_cffi.Curl`` class
    (low-level path, ``validated_hosts`` set) -- so nothing hits the network.
    ``_HAVE_CURL`` is forced True so the dispatch routes through the backend
    regardless of install state.
    """

    def _mock_response(
        self,
        *,
        status: int = 200,
        content: bytes = b"hello",
        headers: dict[str, str] | None = None,
        url: str = "https://example.com/",
    ) -> Mock:
        resp = Mock()
        resp.status_code = status
        resp.content = content
        resp.headers = headers or {}
        resp.url = url
        return resp

    # Each hop is a Mock carrying ``.status`` (int), ``.body`` (bytes), and
    # ``.raw_headers`` (bytes: the CRLF header block). ``perform`` advances through hops
    # in order, writing into the WRITEDATA / HEADERDATA buffers. The returned list
    # captures every ``(option, value)`` passed to setopt.
    def _fake_curl_class(
        self,
        hops: list[Mock],
    ) -> tuple[type, list[tuple[int, object]]]:
        """Build a fake ``Curl`` class replaying *hops* and recording setopts."""
        setopts: list[tuple[int, object]] = []
        state = {"i": 0}

        class _FakeCurl:
            def __init__(self) -> None:
                self._write: io.BytesIO | None = None
                self._header: io.BytesIO | None = None

            def setopt(self, option: int, value: object) -> int:
                setopts.append((int(option), value))
                if int(option) == int(CurlOpt.WRITEDATA):
                    assert isinstance(value, io.BytesIO)
                    self._write = value
                elif int(option) == int(CurlOpt.HEADERDATA):
                    assert isinstance(value, io.BytesIO)
                    self._header = value
                return 0

            def impersonate(self, target: str, default_headers: bool = True) -> int:
                del target, default_headers
                return 0

            def perform(
                self,
                clear_headers: bool = True,
                clear_resolve: bool = True,
            ) -> None:
                del clear_headers, clear_resolve
                hop = hops[state["i"]]
                state["i"] += 1
                assert self._write is not None
                assert self._header is not None
                body = hop.body
                raw_headers = hop.raw_headers
                assert isinstance(body, bytes)
                assert isinstance(raw_headers, bytes)
                _ = self._write.write(body)
                _ = self._header.write(raw_headers)

            def getinfo(self, option: int) -> bytes | int:
                hop = hops[state["i"] - 1]
                if int(option) == int(CurlInfo.RESPONSE_CODE):
                    status = hop.status
                    assert isinstance(status, int)
                    return status
                return b""

            def close(self) -> None:
                pass

            def reset(self) -> None:
                self._write = None
                self._header = None

        return _FakeCurl, setopts

    def _hop(
        self,
        *,
        status: int,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> Mock:
        raw = b"".join(f"{k}: {v}\r\n".encode() for k, v in (headers or {}).items())
        m = Mock()
        m.status = status
        m.body = body
        m.raw_headers = b"HTTP/2 %d\r\n" % status + raw + b"\r\n"
        return m

    def test_high_level_get_impersonates_and_returns_body(self) -> None:
        # The simple curl path uses high-level requests with chrome
        # impersonation and no manual conn; returns the decoded body.
        resp = self._mock_response(content=b"hello")
        with (
            patch("curl_cffi.requests.request", return_value=resp) as mock_req,
        ):
            body, _ = fetch("https://example.com")
        assert body == b"hello"
        assert mock_req.call_args.kwargs["impersonate"] == "chrome"
        assert mock_req.call_args.kwargs["allow_redirects"] is False

    def test_ssrf_pin_and_repin_on_cross_host_redirect(self) -> None:
        # The pin is an option on the POOLED session, so each host gets its own
        # pool entry: "host:port:ip" for the origin, and a re-pin to the
        # redirect target's validated IP. Formerly this drove a raw Curl handle,
        # which cost the request its connection and cookie jar.
        pins: list[tuple[str, tuple[str, str] | None, int]] = []

        def spy(
            egress: str,
            domain: str,
            impersonate: str,
            *,
            pin: ValidatedHost | None = None,
            port: int = 443,
        ) -> StubSession:
            del egress, impersonate
            pins.append((domain, None if pin is None else (pin.host, pin.ip), port))
            return StubSession()

        redirect = self._mock_response(
            status=302,
            headers={"location": "https://other.com/final"},
        )
        final = self._mock_response(status=200, content=b"done")
        with (
            patch("curl_cffi.requests.request", side_effect=[redirect, final]),
            patch.object(fetch_mod, "curl_session", spy),
        ):
            body, _ = fetch(
                "https://example.com/start",
                request=RequestParams(
                    observe=ObserveParams(on_redirect=lambda _u: None),
                    policy=PolicyParams(transport="curl"),
                ),
            )
        assert body == b"done"
        assert [domain for domain, _pin, _port in pins] == ["example.com", "other.com"]
        assert all(pin is not None for _d, pin, _p in pins)

    def test_one_shot_request_pins_the_ip_it_validated(self) -> None:
        """A keyless request must connect to the address it validated.

        Discarding the resolved IP leaves curl to look the host up a second
        time, which is the DNS-rebinding window ``pinned_host`` exists to close.
        """
        resp = self._mock_response(content=b"hello")
        with (
            patch.object(
                curl,
                "pinned_host",
                return_value=ValidatedHost(host="example.com", ip="93.184.216.34"),
            ),
            patch("curl_cffi.requests.request", return_value=resp) as mock_req,
        ):
            fetch(
                "https://example.com/x",
                request=RequestParams(
                    content=ContentParams(raw_headers=True, headers={}),
                    policy=PolicyParams(transport="curl"),
                ),
            )

        options = _recorded_curl_options(mock_req)
        assert options.get(CurlOpt.RESOLVE) == ["example.com:443:93.184.216.34"]

    def test_one_shot_request_pins_a_non_default_port(self) -> None:
        """``RESOLVE`` is per host:port, so the pin must name the real port.

        An entry naming 443 simply does not apply to a request on 8443 -- curl
        falls back to its own resolution and the connection is unpinned, with
        nothing in the request reporting that the guard lapsed.
        """
        resp = self._mock_response(content=b"hello")
        with (
            patch.object(
                curl,
                "pinned_host",
                return_value=ValidatedHost(host="example.com", ip="93.184.216.34"),
            ),
            patch("curl_cffi.requests.request", return_value=resp) as mock_req,
        ):
            fetch(
                "https://example.com:8443/x",
                request=RequestParams(
                    content=ContentParams(raw_headers=True, headers={}),
                    policy=PolicyParams(transport="curl"),
                ),
            )

        options = _recorded_curl_options(mock_req)
        assert options.get(CurlOpt.RESOLVE) == ["example.com:8443:93.184.216.34"]

    def test_pin_brackets_ipv6_resolve_entry(self) -> None:
        # REV2-002: a v6 pin must be "host:port:[v6]" -- curl mis-parses an
        # unbracketed IPv6 (its colons collide with the host:port delimiters).
        built: list[dict[object, object]] = []

        class _Session:
            def __init__(self, **kwargs: object) -> None:
                options = kwargs.get("curl_options")
                assert isinstance(options, dict) or options is None
                built.append(cast(dict[object, object], options or {}))
                self.cookies = StubCookies()

            def close(self) -> None:
                pass

        pool: dict[_SessionKey, cc_requests.Session[Response]] = {}
        with (
            patch.object(curl, "_curl_sessions", pool),
            patch("curl_cffi.requests.Session", _Session),
        ):
            curl.curl_session(
                "203.0.113.1",
                "v6.example",
                "chrome",
                pin=ValidatedHost(host="v6.example", ip="2606:4700:20::1"),
            )
        resolves = [
            value
            for options in built
            for key, value in options.items()
            if key == CurlOpt.RESOLVE
        ]
        assert ["v6.example:443:[2606:4700:20::1]"] in resolves

    def test_pinned_curl_rewrites_origin_on_cross_host_redirect(self) -> None:
        # REV2-001: a POST that redirects cross-origin must NOT leak the source
        # Origin. Header must be rewritten to the new origin on each hop.
        redirect = self._mock_response(
            status=307,
            headers={"location": "https://b.com/land"},
        )
        final = self._mock_response(status=200, content=b"done")
        with patch(
            "curl_cffi.requests.request",
            side_effect=[redirect, final],
        ) as mock_req:
            fetch(
                "https://a.com/submit",
                request=RequestParams(
                    content=ContentParams(method="POST", data={"x": "1"}),
                    policy=PolicyParams(transport="curl"),
                ),
            )
        second = _recorded_headers(mock_req)
        assert second["Origin"] == "https://b.com"

    def test_simple_curl_rewrites_origin_on_cross_host_redirect(self) -> None:
        # REV2-001 (high-level path): same Origin-leak guard without pinning.
        redir = self._mock_response(
            status=307,
            content=b"",
            headers={"location": "https://b.com/land"},
        )
        ok = self._mock_response(status=200, content=b"done")
        with (
            patch("curl_cffi.requests.request", side_effect=[redir, ok]) as mock_req,
        ):
            fetch(
                "https://a.com/submit",
                request=RequestParams(
                    content=ContentParams(method="POST", data={"x": "1"}),
                ),
            )
        second_headers = _recorded_headers(mock_req)
        assert second_headers.get("Origin") == "https://b.com"

    def test_pooled_curl_loads_caller_cookies_into_jar_not_header(self) -> None:
        # F3 / S1: on the pooled-curl path a caller cookie is loaded INTO the
        # session jar (the single cookie source), never ALSO sent via a Cookie
        # header -- curl would then emit both, duplicating a name the jar holds.
        stub = StubSession()
        resp = self._mock_response(content=b"ok")
        with (
            patch("curl_cffi.requests.request", return_value=resp) as mock_req,
            patch.object(fetch_mod, "curl_session", const_curl_session(stub)),
        ):
            fetch(
                "https://example.com",
                request=RequestParams(
                    content=ContentParams(cookies={"CONSENT": "YES+"}),
                ),
            )
        kwargs = mock_req.call_args.kwargs
        # Cookie is in the jar, not the header, and cookies= kwarg is unset.
        assert {(c.name, c.value) for c in stub.cookies.jar} == {("CONSENT", "YES+")}
        assert "Cookie" not in _recorded_headers(mock_req)
        assert not kwargs.get("cookies")

    def test_case_variant_cookie_header_not_duplicated(self) -> None:
        # REV2A-008: a caller lowercase headers={"cookie":...} plus a cookies=
        # param must collapse to ONE cookie header key (HTTP header names are
        # case-insensitive; two dict keys -> two Cookie lines on the wire).
        resp = self._mock_response(content=b"ok")
        with patch("curl_cffi.requests.request", return_value=resp) as mock_req:
            fetch(
                "https://example.com",
                request=RequestParams(
                    content=ContentParams(
                        headers={"cookie": "a=1"},
                        cookies={"b": "2"},
                    ),
                ),
            )
        sent = _recorded_headers(mock_req)
        cookie_keys = [k for k in sent if k.lower() == "cookie"]
        assert len(cookie_keys) == 1, f"duplicate cookie header keys: {cookie_keys}"

    def test_redirect_cap_follows_up_to_limit_then_returns_body(self) -> None:
        # on_redirect fires once per FOLLOWED hop; when the cap is reached the
        # curl path returns the final 3xx body (matching fetch_stdlib's
        # "return the 3xx body at the cap" contract), it does NOT raise.
        seen: list[str] = []
        responses = [
            self._mock_response(status=302, headers={"location": "https://a.com/1"}),
            self._mock_response(status=302, headers={"location": "https://a.com/2"}),
            self._mock_response(
                status=302,
                content=b"final 3xx",
                headers={"location": "/3"},
            ),
        ]
        with patch("curl_cffi.requests.request", side_effect=responses):
            body, _ = fetch(
                "https://a.com/start",
                request=RequestParams(
                    retry=RetryParams(max_redirects=2),
                    observe=ObserveParams(on_redirect=seen.append),
                    policy=PolicyParams(transport="curl"),
                ),
            )
        assert body == b"final 3xx"
        assert seen == ["https://a.com/1", "https://a.com/2"]

    def test_error_status_raises_withdecompressed_body(self) -> None:
        # A challenge 403 must surface a READABLE body in FetchError.body --
        # ``classify_challenge`` matches on markup, so an undecoded body would
        # silently downgrade every Cloudflare wall to a generic HTTP error.
        html = b"<!DOCTYPE html><html>Just a moment...</html>"
        response = self._mock_response(
            status=403,
            content=html,
            headers={"server": "cloudflare"},
        )
        with (
            patch("curl_cffi.requests.request", return_value=response),
            pytest.raises(FetchError) as exc,
        ):
            fetch(
                "https://example.com",
                request=RequestParams(policy=PolicyParams(transport="curl")),
            )
        assert exc.value.status == 403
        assert exc.value.body == html

    def test_raw_headers_sends_only_provided_headers(self) -> None:
        # (d) raw_headers=True: the high-level curl request receives exactly the
        # caller's header (plus nothing derived from the Chrome default set).
        resp = self._mock_response(content=b"ok")
        with (
            patch("curl_cffi.requests.request", return_value=resp) as mock_req,
        ):
            fetch(
                "https://example.com",
                request=RequestParams(
                    content=ContentParams(
                        headers={"User-Agent": "custom"},
                        raw_headers=True,
                    ),
                ),
            )
        assert mock_req.call_args.kwargs["headers"] == {"User-Agent": "custom"}

    def test_curl_exception_maps_to_fetch_error_status_zero(self) -> None:
        # (e) any curl_cffi exception (connection/timeout) becomes
        # FetchError(status=0) rather than leaking the raw curl error.
        with (
            patch(
                "curl_cffi.requests.request",
                side_effect=CurlError("connection refused"),
            ),
            pytest.raises(FetchError) as exc,
        ):
            fetch("https://example.com")
        assert exc.value.status == 0
        assert b"connection refused" in exc.value.body

    def test_303_converts_post_to_get_and_drops_body(self) -> None:
        # A 303 on the curl path switches the follow-up to GET with no body.
        resp_303 = self._mock_response(
            status=303,
            headers={"location": "https://example.com/result"},
        )
        resp_ok = self._mock_response(content=b"got it")
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def _record(*args: object, **kwargs: object) -> Mock:
            calls.append((args, kwargs))
            return resp_303 if len(calls) == 1 else resp_ok

        with (
            patch("curl_cffi.requests.request", side_effect=_record),
        ):
            body, _ = fetch(
                "https://example.com/submit",
                request=RequestParams(
                    content=ContentParams(method="POST", data={"x": "1"}),
                    observe=ObserveParams(on_redirect=lambda _u: None),
                ),
            )
        assert body == b"got it"
        # Method is the first positional arg to cc_requests.request.
        assert calls[1][0][0] == "GET"
        assert calls[1][1]["data"] is None

    def test_303_get_drops_content_type_header(self) -> None:
        # REV2061-002: a 303 switches POST->GET; the POST-only Content-Type must
        # NOT survive onto the bodyless GET (a real browser drops it).
        resp_303 = self._mock_response(
            status=303,
            headers={"location": "https://example.com/result"},
        )
        resp_ok = self._mock_response(content=b"ok")
        calls: list[dict[str, object]] = []

        def _record(*_a: object, **kw: object) -> Mock:
            calls.append(kw)
            return resp_303 if len(calls) == 1 else resp_ok

        with (
            patch("curl_cffi.requests.request", side_effect=_record),
        ):
            fetch(
                "https://example.com/submit",
                request=RequestParams(
                    content=ContentParams(method="POST", json={"x": 1}),
                ),
            )
        headers = calls[1]["headers"]
        assert isinstance(headers, dict)
        assert "Content-Type" not in (headers)

    def test_max_redirects_zero_returns_3xx_body_on_curl(self) -> None:
        # REV2061-001: max_redirects=0 means "do not follow, return the 3xx
        # body" (the documented contract, matched by the stdlib path) -- the
        # curl path must NOT raise on the first redirect.
        resp = self._mock_response(
            status=302,
            content=b"redirect body",
            headers={"location": "https://example.com/other"},
        )
        with (
            patch("curl_cffi.requests.request", return_value=resp) as mock_req,
        ):
            body, _ = fetch(
                "https://example.com",
                request=RequestParams(retry=RetryParams(max_redirects=0)),
            )
        assert body == b"redirect body"
        assert mock_req.call_count == 1  # Never followed.

    def test_curl_connection_error_is_retried(self) -> None:
        # A1: a curl transport error (connection refused/timeout) becomes
        # FetchError(status=0). retries= must retry it -- the stdlib path retries
        # a raw OSError, so the curl path must retry its status-0 equivalent, or
        # the two transports disagree on what `retries=` means.
        ok = self._mock_response(content=b"ok")
        with (
            patch(
                "curl_cffi.requests.request",
                side_effect=[CurlError("connection refused"), ok],
            ),
            patch("wesearch.fetch.fetch.time.sleep"),
        ):
            assert (
                fetch(
                    "https://example.com",
                    request=RequestParams(retry=RetryParams(retries=1)),
                )[0]
                == b"ok"
            )

    def test_curl_connection_error_exhausts_retries_then_raises(self) -> None:
        # The retry must still terminate: a persistent curl error raises after
        # the budget, not loop forever.
        with (
            patch("curl_cffi.requests.request", side_effect=CurlError("refused")),
            patch("wesearch.fetch.fetch.time.sleep"),
            pytest.raises(FetchError) as exc,
        ):
            fetch(
                "https://example.com",
                request=RequestParams(retry=RetryParams(retries=2)),
            )
        assert exc.value.status == 0

    def test_same_origin_redirect_keeps_one_pooled_session(self) -> None:
        # A3: a same-origin hop must NOT re-seat the session -- the pin and the
        # cookie jar are per-Session, so churning one per hop would discard the
        # connection continuity the pool exists to provide.
        seats: list[str] = []

        def spy(
            egress: str,
            domain: str,
            impersonate: str,
            *,
            pin: ValidatedHost | None = None,
            port: int = 443,
        ) -> StubSession:
            del egress, impersonate, pin, port
            seats.append(domain)
            return StubSession()

        redirect = self._mock_response(
            status=302,
            headers={"location": "https://example.com/next"},
        )
        final = self._mock_response(status=200, content=b"ok")
        with (
            patch("curl_cffi.requests.request", side_effect=[redirect, final]),
            patch.object(fetch_mod, "curl_session", spy),
        ):
            body, _ = fetch(
                "https://example.com/start",
                request=RequestParams(
                    observe=ObserveParams(on_redirect=lambda _u: None),
                    policy=PolicyParams(transport="curl"),
                ),
            )
        assert body == b"ok"
        assert seats == ["example.com"]  # Seated once, reused on the same-origin hop.


class TestCurlSessionPoolLocking:
    """Every mutation of the curl session pool holds its pool lock."""

    @pytest.fixture(autouse=True)
    def _real_curl_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The module isolate_profiles fixture stubs curl_session; restore the
        # real function so these tests exercise its actual locking.
        monkeypatch.setattr(fetch_mod, "curl_session", _REAL_CURL_SESSION)

    def test_curl_session_holds_pool_lock(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        acquired: list[str] = []
        real_lock = curl._curl_lock

        class _Instrumented:
            def __enter__(self) -> None:
                acquired.append("enter")
                real_lock.acquire()

            def __exit__(self, *_a: object) -> None:
                real_lock.release()

        monkeypatch.setattr(curl, "_curl_lock", _Instrumented())
        monkeypatch.setattr(curl, "_curl_sessions", {})
        with patch("curl_cffi.requests.Session", return_value=Mock()):
            curl.curl_session("1.2.3.4", "x.com", "chrome")
        assert acquired, "curl_session mutated the pool without _curl_lock"

    def test_close_curl_session_holds_pool_lock(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        acquired: list[str] = []
        real_lock = curl._curl_lock

        class _Instrumented:
            def __enter__(self) -> None:
                acquired.append("enter")
                real_lock.acquire()

            def __exit__(self, *_a: object) -> None:
                real_lock.release()

        monkeypatch.setattr(curl, "_curl_lock", _Instrumented())
        monkeypatch.setattr(curl, "_curl_sessions", {})
        curl.close_curl_session("1.2.3.4", "x.com", "chrome")  # Absent: no-op.
        assert acquired, "close_curl_session mutated the pool without _curl_lock"

    def test_close_sessions_except_preserves_current_egress(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        current = Mock()
        stale = Mock()
        monkeypatch.setattr(
            curl,
            "_curl_sessions",
            {
                ("1.2.3.4", "x.com", "chrome"): current,
                ("5.6.7.8", "x.com", "chrome"): stale,
            },
        )

        curl.close_curl_sessions_except("1.2.3.4")

        assert list(curl._curl_sessions) == [("1.2.3.4", "x.com", "chrome")]
        current.close.assert_not_called()
        stale.close.assert_called_once_with()


class TestCurlPathSendsUserAgent:
    """The curl path presents a coherent browser identity on the real wire.

    A request that omits the User-Agent is rejected by UA-gated APIs (GitHub's
    REST API 403s UA-less requests), which once surfaced as spurious
    ``Fetch failed: HTTP 403`` on every WebFetch -- the SSRF-pinned fork drove a
    raw handle whose ``impersonate()`` injected no request headers at all. That
    fork is gone; this test is what proves the surviving path still speaks
    Chrome, since only a REAL curl_cffi call reveals its header injection.

    Hermetic: a loopback HTTP server echoes the request headers back.
    """

    def test_curl_get_sends_user_agent_header(self) -> None:
        seen: dict[str, str] = {}

        class _Echo(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                for k, v in self.headers.items():
                    seen[k.lower()] = v
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            @override
            def log_message(self, format: str, *args: object) -> None:
                del format, args

        server = HTTPServer(("127.0.0.1", 0), _Echo)
        port = server.server_address[1]
        thread = Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            body, _ = fetch(
                f"http://127.0.0.1:{port}/",
                # The oracle IS loopback, so this test authors its own URL --
                # exactly what "internal" declares. Leaving it untrusted would
                # (correctly) refuse the fetch.
                request=RequestParams(
                    policy=PolicyParams(transport="curl", trust="internal"),
                ),
            )
        finally:
            server.server_close()
            thread.join(timeout=5)

        assert body == b"ok"
        assert seen.get("user-agent"), (
            "curl path sent no User-Agent; UA-gated APIs (e.g. GitHub) "
            f"403 such requests. headers seen: {sorted(seen)}"
        )
        assert "chrome" in seen["user-agent"].lower()
        # Full coherent Chrome identity, not just a bare UA (a partial set is
        # itself a bot tell): every rung must send what the others send.
        assert seen.get("accept"), f"curl path missing Accept: {sorted(seen)}"
        assert seen.get("sec-fetch-mode") == "navigate", (
            f"curl path missing Sec-Fetch navigation headers: {sorted(seen)}"
        )


class TestSeedSessionJar:
    def test_secure_prefixed_cookie_seeded_without_warning(self) -> None:
        # RFC 6265bis: a __Secure-/__Host- prefixed cookie is only valid Secure;
        # seeding it without secure=True made curl_cffi emit a CurlCffiWarning
        # (which the live Google-search integration path surfaced as a failure).
        session = cast(cc_requests.Session[Response], cc_requests.Session())
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                seed_session_jar(
                    session,
                    "www.google.com",
                    {"__Secure-STRP": "abc", "__Host-GSP": "def", "NID": "ghi"},
                )
        finally:
            session.close()
        jar = {c.name: c for c in session.cookies.jar}
        assert jar["__Secure-STRP"].secure is True
        assert jar["__Secure-STRP"].domain == "www.google.com"
        # __Host- is host-only per spec: Secure, no Domain, Path=/.
        assert jar["__Host-GSP"].secure is True
        assert jar["__Host-GSP"].path == "/"
        # A plain cookie is seeded non-Secure (Chrome sends it over either).
        assert jar["NID"].secure is False


def _recorded_curl_options(mock_req: Mock) -> dict[object, object]:
    """Return the typed curl options mapping recorded by a request mock."""
    options = mock_req.call_args.kwargs.get("curl_options")
    assert isinstance(options, dict)
    return cast(dict[object, object], options)


def _recorded_headers(mock_req: Mock) -> dict[str, str]:
    """Return the typed headers mapping recorded by a request mock."""
    return from_plain(mock_req.call_args.kwargs["headers"], dict[str, str])


def test_curl_set_cookies_uses_all_set_cookie_headers() -> None:
    class Headers:
        def items(self) -> list[tuple[str, str]]:
            return []

        def get_list(self, name: str) -> list[str]:
            assert name == "set-cookie"
            return ["A=1", "B=2"]

    response = Mock(status_code=200, content=b"ok", headers=Headers())
    with patch("curl_cffi.requests.request", return_value=response):
        _, session = fetch("https://example.com")
    assert session.cookies_for("https://example.com/") == {"A": "1", "B": "2"}


def test_curl_set_cookies_falls_back_to_one_string() -> None:
    class Headers:
        def items(self) -> list[tuple[str, str]]:
            return []

        def get(self, name: str) -> str:
            assert name == "set-cookie"
            return "A=1"

    response = Mock(status_code=200, content=b"ok", headers=Headers())
    with patch("curl_cffi.requests.request", return_value=response):
        _, session = fetch("https://example.com")
    assert session.cookies_for("https://example.com/") == {"A": "1"}


def test_curl_set_cookies_ignores_non_string_fallback() -> None:
    class Headers:
        def items(self) -> list[tuple[str, str]]:
            return []

        def get(self, name: str) -> None:
            assert name == "set-cookie"

    response = Mock(status_code=200, content=b"ok", headers=Headers())
    with patch("curl_cffi.requests.request", return_value=response):
        _, session = fetch("https://example.com")
    assert session.cookies_for("https://example.com/") == {}


def test_jar_set_preserves_cookie_prefix_contract() -> None:
    session: cc_requests.Session[Response] = cc_requests.Session()
    try:
        with patch.object(session.cookies, "set") as set_cookie:
            _jar_set(session, "example.com", "__Host-ID", "host")
            _jar_set(session, "example.com", "__Secure-ID", "secure")
            _jar_set(session, "example.com", "ID", "plain")
        assert set_cookie.call_args_list == [
            (("__Host-ID", "host"), {"path": "/", "secure": True}),
            (("__Secure-ID", "secure"), {"domain": "example.com", "secure": True}),
            (("ID", "plain"), {"domain": "example.com"}),
        ]
    finally:
        session.close()


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("example.co.uk", "example.co.uk"),
        ("x.example.co.uk", "example.co.uk"),
        ("example.com", "example.com"),
        ("x.example.com", "example.com"),
        ("a.abcd.co", "abcd.co"),
        ("a.b", "a.b"),
    ],
)
def test_registrable_domain_boundary(host: str, expected: str) -> None:
    assert _registrable_domain(host) == expected


def test_curl_session_key_includes_every_identity_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions: dict[object, object] = {}
    created: list[dict[str, object]] = []

    class Session:
        def __init__(self, **kwargs: object) -> None:
            created.append(kwargs)

    monkeypatch.setattr(curl, "_curl_sessions", sessions)
    monkeypatch.setattr("curl_cffi.requests.Session", Session)
    pin = ValidatedHost(host="example.com", ip="192.0.2.1")
    first = curl_session("egress", "a.example.com", "chrome", pin=pin, port=8443)
    assert (
        curl_session("egress", "b.example.com", "chrome", pin=pin, port=8443) is first
    )
    assert (
        curl_session("egress", "a.example.com", "firefox", pin=pin, port=8443)
        is not first
    )
    assert (
        curl_session("other", "a.example.com", "chrome", pin=pin, port=8443)
        is not first
    )
    assert (
        curl_session("egress", "a.example.com", "chrome", pin=pin, port=443)
        is not first
    )
    assert len(created) == 4
    assert created[0]["impersonate"] == "chrome"
    assert created[0]["curl_options"]


def test_close_curl_session_drops_all_pins_and_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matching_a = Mock()
    matching_b = Mock()
    other = Mock()
    monkeypatch.setattr(
        curl,
        "_curl_sessions",
        {
            ("e", "example.com", "chrome", None, 443): matching_a,
            ("e", "example.com", "chrome", None, 8443): matching_b,
            ("other", "example.com", "chrome", None, 443): other,
        },
    )
    close_curl_session("e", "www.example.com", "chrome")
    assert matching_a.close.call_count == 1
    assert matching_b.close.call_count == 1
    assert other.close.call_count == 0


def test_close_curl_sessions_except_keeps_none_egress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained = Mock()
    removed = Mock()
    monkeypatch.setattr(
        curl,
        "_curl_sessions",
        {
            (None, "example.com", "chrome", None, 443): retained,
            ("e", "example.com", "chrome", None, 443): removed,
        },
    )
    close_curl_sessions_except(None)
    assert list(curl._curl_sessions) == [(None, "example.com", "chrome", None, 443)]
    retained.close.assert_not_called()
    removed.close.assert_called_once_with()


def test_curl_loop_follow_reports_response_and_transforms_redirect() -> None:
    loop = _CurlLoop(
        url="https://a.example/submit",
        method="POST",
        headers={"Origin": "https://a.example", "Content-Type": "x"},
        body=b"body",
        remaining=1,
    )
    responses: list[tuple[int, dict[str, str], str]] = []
    redirects: list[str] = []

    def record_response(
        status: int,
        headers: dict[str, str],
        response_url: str,
    ) -> None:
        responses.append((status, headers, response_url))

    assert loop.follow(
        303,
        {"location": "https://b.example/result"},
        on_response=record_response,
        on_redirect=redirects.append,
    )
    assert responses == [
        (303, {"location": "https://b.example/result"}, "https://a.example/submit"),
    ]
    assert redirects == ["https://b.example/result"]
    assert loop.url == "https://b.example/result"
    assert loop.method == "GET"
    assert loop.body is None
    assert loop.headers["Origin"] == "https://b.example"
    assert loop.remaining == 0


def test_curl_loop_follow_does_not_follow_at_zero() -> None:
    loop = _CurlLoop(
        url="https://a.example/",
        method="GET",
        headers={},
        body=None,
        remaining=0,
    )
    assert not loop.follow(
        302,
        {"location": "https://b.example/"},
        on_response=None,
        on_redirect=None,
    )
    assert loop.url == "https://a.example/"


def test_fetch_curl_records_one_shot_request_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Mock(status_code=200, content=b"body", headers={})
    calls: list[tuple[object, ...]] = []

    def request(*args: object, **kwargs: object) -> Mock:
        calls.append((args, kwargs))
        return response

    monkeypatch.setattr("curl_cffi.requests.request", request)

    def pin(*_args: object, **_kwargs: object) -> ValidatedHost:
        return ValidatedHost(host="example.com", ip="192.0.2.1")

    monkeypatch.setattr(curl, "pinned_host", pin)
    assert (
        fetch_curl(
            "https://example.com/path",
            method="POST",
            headers={"X-Test": "yes"},
            body=b"payload",
            timeout_sec=9.0,
            connect_timeout_sec=3.0,
            max_redirects=0,
            impersonate="chrome",
            on_redirect=None,
            on_response=None,
        )
        == b"body"
    )
    args, kwargs = calls[0]
    assert args == ("POST", "https://example.com/path")
    assert kwargs == {
        "headers": {"X-Test": "yes"},
        "data": b"payload",
        "impersonate": "chrome",
        "timeout": (3.0, 9.0),
        "allow_redirects": False,
        "curl_options": {CurlOpt.RESOLVE: ["example.com:443:192.0.2.1"]},
    }


def test_fetch_curl_classifies_status_four_hundred() -> None:
    response = Mock(status_code=400, content=b"bad", headers={})
    with (
        patch("curl_cffi.requests.request", return_value=response),
        pytest.raises(FetchError) as error,
    ):
        fetch_curl(
            "https://example.com/path",
            method="GET",
            headers={},
            body=None,
            timeout_sec=1.0,
            max_redirects=0,
            impersonate="chrome",
            on_redirect=None,
            on_response=None,
        )
    assert error.value.status == 400


def test_fetch_curl_passes_trust_to_each_one_shot_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[object] = []
    response = Mock(status_code=200, content=b"", headers={})

    def pin(url: str, trust: object) -> ValidatedHost:
        del url
        seen.append(trust)
        return ValidatedHost(host="example.com", ip="192.0.2.1")

    monkeypatch.setattr(curl, "pinned_host", pin)

    def request(*_args: object, **_kwargs: object) -> Mock:
        return response

    monkeypatch.setattr("curl_cffi.requests.request", request)
    assert (
        fetch_curl(
            "https://example.com/",
            method="GET",
            headers={},
            body=None,
            timeout_sec=1.0,
            max_redirects=0,
            impersonate="chrome",
            on_redirect=None,
            on_response=None,
            trust="internal",
        )
        == b""
    )
    assert seen == ["internal"]


def test_fetch_curl_error_keeps_current_url_and_empty_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Mock(status_code=400, content=None, headers={})

    def request(*_args: object, **_kwargs: object) -> Mock:
        return response

    monkeypatch.setattr("curl_cffi.requests.request", request)
    with pytest.raises(FetchError) as error:
        fetch_curl(
            "https://example.com/path",
            method="GET",
            headers={},
            body=None,
            timeout_sec=1.0,
            max_redirects=0,
            impersonate="chrome",
            on_redirect=None,
            on_response=None,
        )
    assert error.value.url == "https://example.com/path"
    assert error.value.body == b""


def test_fetch_curl_pooled_request_uses_session_arguments() -> None:
    response = Mock(status_code=200, content=b"body", headers={})
    session: cc_requests.Session[Response] = cc_requests.Session()
    with patch.object(session, "request", return_value=response) as request:
        assert (
            fetch_curl(
                "https://example.com/path",
                method="POST",
                headers={"X-Test": "yes"},
                body=b"payload",
                timeout_sec=9.0,
                connect_timeout_sec=3.0,
                max_redirects=0,
                impersonate="chrome",
                on_redirect=None,
                on_response=None,
                session=session,
            )
            == b"body"
        )
    request.assert_called_once_with(
        "POST",
        "https://example.com/path",
        headers={"X-Test": "yes"},
        data=b"payload",
        impersonate="chrome",
        timeout=(3.0, 9.0),
        allow_redirects=False,
    )
    session.close()


def test_seed_and_set_cookies_preserve_domain_and_value() -> None:
    session: cc_requests.Session[Response] = cc_requests.Session()
    try:
        seed_session_jar(session, "example.com", {"A": "one"})
        set_session_cookies(session, "other.example", {"B": "two"})
        cookies: set[tuple[str, str, str | None]] = {
            (cookie.domain, cookie.name, cookie.value)
            for cookie in cast(CookieJar, session.cookies.jar)
        }
        assert ("example.com", "A", "one") in cookies
        assert ("other.example", "B", "two") in cookies
    finally:
        session.close()


def test_curl_loop_passes_current_url_to_redirect_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def transform(
        current_url: str,
        headers: dict[str, str],
        method: str,
        *,
        body: bytes | None,
        status: int,
        redirect_url: str,
    ) -> tuple[dict[str, str], str, bytes | None]:
        del headers, method, body, status, redirect_url
        seen.append(current_url)
        return {}, "GET", None

    monkeypatch.setattr(curl, "apply_redirect", transform)
    loop = _CurlLoop(
        url="https://a.example/start",
        method="GET",
        headers={},
        body=None,
        remaining=1,
    )
    assert loop.follow(
        302,
        {"location": "https://a.example/next"},
        on_response=None,
        on_redirect=None,
    )
    assert seen == ["https://a.example/start"]


def test_curl_loop_same_origin_preserves_authorization() -> None:
    loop = _CurlLoop(
        url="https://a.example/start",
        method="GET",
        headers={"Authorization": "Bearer secret"},
        body=None,
        remaining=1,
    )
    assert loop.follow(
        302,
        {"location": "https://a.example/next"},
        on_response=None,
        on_redirect=None,
    )
    assert loop.headers == {"Authorization": "Bearer secret"}


def test_curl_loop_follow_resolves_relative_location_and_preserves_post() -> None:
    loop = _CurlLoop(
        url="https://a.example/base/start",
        method="POST",
        headers={"Content-Type": "x"},
        body=b"body",
        remaining=1,
    )
    assert loop.follow(
        307,
        {"location": "next"},
        on_response=None,
        on_redirect=None,
    )
    assert loop.url == "https://a.example/base/next"
    assert loop.method == "POST"
    assert loop.body == b"body"


def test_default_trust_is_used_for_omitted_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[object] = []

    def pin(url: str, trust: object) -> None:
        del url
        seen.append(trust)

    def request(*args: object, **kwargs: object) -> Mock:
        del args, kwargs
        return Mock(status_code=200, content=b"ok", headers={})

    monkeypatch.setattr(curl, "pinned_host", pin)
    monkeypatch.setattr("curl_cffi.requests.request", request)
    assert (
        fetch_curl(
            "https://example.com/",
            method="GET",
            headers={},
            body=None,
            timeout_sec=1.0,
            max_redirects=0,
            impersonate="chrome",
            on_redirect=None,
            on_response=None,
        )
        == b"ok"
    )
    assert seen == ["untrusted"]


def test_unpinned_one_shot_request_passes_empty_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def request(*_args: object, **kwargs: object) -> Mock:
        calls.append(kwargs)
        return Mock(status_code=200, content=b"ok", headers={})

    def pin(url: str, trust: object) -> None:
        del url, trust

    monkeypatch.setattr(curl, "pinned_host", pin)
    monkeypatch.setattr("curl_cffi.requests.request", request)
    fetch_curl(
        "http://example.com/",
        method="GET",
        headers={},
        body=None,
        timeout_sec=1.0,
        max_redirects=0,
        impersonate="chrome",
        on_redirect=None,
        on_response=None,
    )
    assert calls[0]["curl_options"] == {}


def test_http_one_shot_pin_uses_port_eighty() -> None:
    response = Mock(status_code=200, content=b"ok", headers={})
    with (
        patch.object(
            curl,
            "pinned_host",
            return_value=ValidatedHost(host="example.com", ip="192.0.2.1"),
        ),
        patch("curl_cffi.requests.request", return_value=response) as request,
    ):
        fetch_curl(
            "http://example.com/",
            method="GET",
            headers={},
            body=None,
            timeout_sec=1.0,
            max_redirects=0,
            impersonate="chrome",
            on_redirect=None,
            on_response=None,
        )
    assert request.call_args.kwargs["curl_options"] == {
        CurlOpt.RESOLVE: ["example.com:80:192.0.2.1"],
    }


def test_curl_error_preserves_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_request(*_args: object, **_kwargs: object) -> Mock:
        raise CurlError("broken")

    monkeypatch.setattr("curl_cffi.requests.request", fail_request)
    with pytest.raises(FetchError) as error:
        fetch_curl(
            "https://example.com/fail",
            method="GET",
            headers={},
            body=None,
            timeout_sec=1.0,
            max_redirects=0,
            impersonate="chrome",
            on_redirect=None,
            on_response=None,
        )
    assert error.value.url == "https://example.com/fail"


def test_redirect_target_uses_status_and_source_url() -> None:
    loop = _CurlLoop(
        url="https://a.example/base/start",
        method="POST",
        headers={},
        body=b"body",
        remaining=1,
    )
    with pytest.raises(FetchError, match="302"):
        loop.follow(302, {}, on_response=None, on_redirect=None)


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
