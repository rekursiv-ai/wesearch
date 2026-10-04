"""Tests for wesearch.fetch."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from http import client
from typing import TYPE_CHECKING, Protocol, cast
from unittest.mock import Mock, call, patch

import base64
import importlib
import math

from curl_cffi import requests as cc_requests

import pytest

from wesearch import fetch
from wesearch.fetch import (
    ContentParams,
    FetchSession,
    ObserveParams,
    PolicyParams,
    RequestParams,
    RetryParams,
)
from wesearch.fetch.fetch import (
    _accept_ch_hints,
    _build_headers,
    _curl_structural_headers,
    _fetch_once,
    _fetch_with_identity,
    _google_headers,
    _is_valid_ip_address,
    _Request,
    _reseat,
    _ResponseLearner,
    _send_as,
    _send_via_zendriver,
    _split_userinfo,
    _url_with_params,
    _validated_body,
    egress_ip,
    last_known_egress_ip,
    on_egress_rotation,
    resolve_transport,
    set_last_egress_ip,
)
from wesearch.fetch.testing import (
    StubSession,
    const_curl_session,
    lower_headers,
    zstd_compress,
)
from wesearch.fetch.transport import transport_routing, zendriver
from wesearch.fetch.transport.zendriver import BrowserResult
from wesearch.profile import Profile, ProfileStore
from wesearch.types.errors import (
    BotDetectionError,
    CloudflareChallengeError,
    FetchError,
    GoogleJavascriptRequiredError,
    PuzzleChallengeError,
)


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import logging
    import types


fetch_mod = cast("_FetchModule", importlib.import_module("wesearch.fetch.fetch"))


def test_fetch_uses_transport_package_layout() -> None:
    assert fetch.__file__ is not None
    assert fetch.__file__.endswith("/fetch/__init__.py")
    assert callable(fetch.fetch)
    for module in ("common", "fetch"):
        importlib.import_module(f"wesearch.fetch.{module}")
    for module in ("curl", "stdlib", "zendriver", "transport_routing"):
        importlib.import_module(f"wesearch.fetch.transport.{module}")


def test_accept_ch_hints_keeps_only_known_hints() -> None:
    assert _accept_ch_hints({"accept-ch": "Sec-CH-UA-Arch, unknown"}) == frozenset(
        {"sec-ch-ua-arch"},
    )
    assert _accept_ch_hints({}) == frozenset()


def test_split_userinfo_unquotes_credentials_and_strips_userinfo() -> None:
    assert _split_userinfo("https://u%40ser:p%40ss@example.com/path") == (
        "https://example.com/path",
        "Basic dUBzZXI6cEBzcw==",
    )
    assert _split_userinfo("https://:pass@example.com/path")[1] == "Basic OnBhc3M="
    assert _split_userinfo("https://u:p@ss@example.com/path")[0] == (
        "https://example.com/path"
    )
    assert _split_userinfo("https://example.com/path") == (
        "https://example.com/path",
        None,
    )


def test_ip_family_validation_is_exact() -> None:
    assert _is_valid_ip_address("192.0.2.1", ipv6=False)
    assert not _is_valid_ip_address("2001:db8::1", ipv6=False)
    assert _is_valid_ip_address("2001:db8::1", ipv6=True)
    assert not _is_valid_ip_address("invalid", ipv6=True)


def test_build_headers_raw_returns_only_explicit_headers() -> None:
    assert _build_headers(
        method="GET",
        url="https://example.com",
        content_type=None,
        extra={"X-Test": "yes"},
        raw_headers=True,
        impersonate="chrome",
        use_curl=False,
        accept_ch={},
    ) == {"X-Test": "yes"}


def test_curl_structural_headers_adds_post_fields_and_extras() -> None:
    with patch.object(fetch_mod, "_google_headers", return_value={"X-Google": "yes"}):
        assert _curl_structural_headers(
            method="POST",
            url="https://example.com/path",
            content_type="application/json",
            extra={"X-Test": "yes"},
            impersonate="chrome",
            accept_ch={},
        ) == {
            "Content-Type": "application/json",
            "Origin": "https://example.com",
            "X-Google": "yes",
            "X-Test": "yes",
        }


def test_google_headers_are_empty_off_google() -> None:
    assert _google_headers("https://example.com", "chrome") == {}


def test_request_fetch_passes_raw_header_mode_and_seeded_cookies() -> None:
    request = _Request(
        url="https://example.com/path",
        session=FetchSession(cookies={"https://example.com": {"old": "1"}}),
        params=RequestParams(
            content=ContentParams(
                headers={"X-Test": "yes"},
                cookies={"new": "2"},
                raw_headers=True,
            ),
        ),
    )
    with patch.object(_Request, "send", return_value=b"ok") as send:
        assert request.fetch() == (b"ok", request.session)
    send.assert_called_once_with(
        headers={"X-Test": "yes"},
        cookies={"old": "1", "new": "2"},
        raw_headers=True,
    )


def test_request_send_forwards_every_transport_argument() -> None:
    request = _Request(
        url="https://example.com/path",
        session=FetchSession(impersonate="chrome146"),
        params=RequestParams(),
    )
    observer = Mock()
    Mock()
    reseat = Mock()
    with patch.object(fetch_mod, "_fetch_once", return_value=b"ok") as send:
        assert (
            request.send(
                headers={"X": "1"},
                cookies={"c": "2"},
                raw_headers=True,
                on_response=observer,
                curl=None,
                reseat=reseat,
            )
            == b"ok"
        )
    send.assert_called_once_with(
        "https://example.com/path",
        request.params,
        headers={"X": "1"},
        cookies={"c": "2"},
        raw_headers=True,
        impersonate="chrome146",
        accept_ch=request.session.accept_ch,
        on_response=observer,
        session=None,
        reseat=reseat,
    )


def test_request_send_uses_stored_observer_when_argument_is_none() -> None:
    observer = Mock()
    request = _Request(
        url="https://example.com/path",
        session=FetchSession(),
        params=RequestParams(),
        observer=observer,
    )
    with patch.object(fetch_mod, "_fetch_once", return_value=b"ok") as send:
        request.send(headers=None, cookies=None, raw_headers=False)
    assert send.call_args.kwargs["on_response"] is observer


def test_request_fetch_resolves_auto_using_the_http_method() -> None:
    request = _Request(
        url="https://example.com/path",
        session=FetchSession(),
        params=RequestParams(content=ContentParams(method="POST")),
    )
    with (
        patch.object(_Request, "send", return_value=b"ok"),
        patch.object(
            fetch_mod,
            "resolve_transport",
            wraps=resolve_transport,
        ) as resolve,
    ):
        request.fetch()
    resolve.assert_called_once_with(
        "auto",
        method="POST",
        raw_headers=False,
        has_body=False,
    )


def test_reseat_skips_when_no_egress_is_known() -> None:
    request = _Request(
        url="https://example.com",
        session=FetchSession(),
        params=RequestParams(),
    )
    assert _reseat(request, None, "chrome", "https://other.example") is None


def test_reseat_builds_a_pinned_session_for_redirect_target() -> None:
    request = _Request(
        url="https://example.com",
        session=FetchSession(),
        params=RequestParams(),
    )
    pin = Mock(host="other.example", ip="203.0.113.9")
    session = Mock()
    with (
        patch.object(fetch_mod, "pinned_host", return_value=pin) as validate,
        patch.object(fetch_mod, "curl_session", return_value=session) as build,
    ):
        assert (
            _reseat(
                request,
                "198.51.100.4",
                "chrome",
                "https://other.example:8443/path",
            )
            is session
        )
    validate.assert_called_once_with("https://other.example:8443/path", "untrusted")
    build.assert_called_once_with(
        "198.51.100.4",
        "other.example",
        "chrome",
        pin=pin,
        port=8443,
    )


def test_reseat_preserves_empty_host_and_https_default_port() -> None:
    request = _Request(
        url="https://example.com",
        session=FetchSession(),
        params=RequestParams(),
    )
    with (
        patch.object(fetch_mod, "pinned_host", return_value=None),
        patch.object(fetch_mod, "curl_session", return_value=Mock()) as build,
    ):
        _reseat(request, "198.51.100.4", "chrome", "https:///path")
    assert build.call_args.args[1] == ""
    assert build.call_args.kwargs["port"] == 443


def test_egress_rotation_callbacks_receive_the_new_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetch_mod, "_on_egress_rotation", [])
    monkeypatch.setattr(fetch_mod, "_last_egress_ip", None)
    callback = Mock()
    on_egress_rotation(callback)
    with patch.object(fetch_mod, "close_curl_sessions_except"):
        set_last_egress_ip("203.0.113.10")
    callback.assert_called_once_with("203.0.113.10")


class TestUrlWithParams:
    def test_params_precede_a_fragment(self) -> None:
        # A fragment ends the URL and is never sent to the server, so appending
        # the query after it silently dropped every parameter from the wire.
        assert (
            _url_with_params("https://e/p#section", {"q": "x"})
            == "https://e/p?q=x#section"
        )

    def test_params_merge_into_an_existing_query(self) -> None:
        assert (
            _url_with_params("https://e/p?a=1#s", {"q": "x"}) == "https://e/p?a=1&q=x#s"
        )


class TestBackoffDelay:
    def test_exponential_growth(self) -> None:
        d0 = RetryParams().backoff_delay(0, {})
        d2 = RetryParams().backoff_delay(2, {})
        assert d0 < d2

    def test_capped_at_30(self) -> None:
        assert RetryParams().backoff_delay(100, {}) <= 45  # 30 + 0.5*30.

    def test_retry_after_header(self) -> None:
        assert RetryParams().backoff_delay(0, {"retry-after": "5"}) == 5.0

    def test_retry_after_capped(self) -> None:
        assert RetryParams().backoff_delay(0, {"retry-after": "999"}) == 30.0

    def test_retry_after_http_date_honored(self) -> None:
        # REV2A-007: Retry-After may be an HTTP-date, not just delta-seconds.
        # A near-future date must produce a positive delay (honored), not fall
        # through to exponential backoff.
        future = datetime.now(tz=UTC) + timedelta(seconds=10)
        delay = RetryParams().backoff_delay(0, {"retry-after": format_datetime(future)})
        assert 5 <= delay <= 30  # ~10s, capped at 30; not the ~1s exp backoff.

    def test_retry_after_past_date_is_zero(self) -> None:
        # A past HTTP-date means "retry now": non-negative, small.
        assert (
            RetryParams().backoff_delay(
                0,
                {"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"},
            )
            == 0.0
        )

    @pytest.mark.parametrize("value", ["-1", "nan", "-inf"])
    def test_retry_after_hostile_numeric_stays_sleepable(self, value: str) -> None:
        # The delay goes straight to time.sleep, which raises ValueError on a
        # negative and never wakes on a NaN -- either turns a retryable response
        # into a crash that masks the underlying FetchError.
        delay = RetryParams().backoff_delay(0, {"retry-after": value})
        assert math.isfinite(delay)
        assert delay >= 0.0


class TestSplitUserinfo:
    def test_no_userinfo(self) -> None:
        assert _split_userinfo("https://example.com/p?q=1") == (
            "https://example.com/p?q=1",
            None,
        )

    def test_user_pass_stripped_and_encoded(self) -> None:
        url, auth = _split_userinfo("https://u:p@example.com:8443/x")
        assert url == "https://example.com:8443/x"
        assert auth == "Basic " + base64.b64encode(b"u:p").decode()

    def test_pct_decoded_credentials(self) -> None:
        url, auth = _split_userinfo("https://u%40x:p%3Aw@example.com/")
        assert url == "https://example.com/"
        assert auth == "Basic " + base64.b64encode(b"u@x:p:w").decode()

    def test_user_only(self) -> None:
        url, auth = _split_userinfo("https://u@example.com/")
        assert url == "https://example.com/"
        assert auth == "Basic " + base64.b64encode(b"u:").decode()


class TestFetchError:
    def test_attributes(self) -> None:
        err = FetchError(
            "https://x.com",
            404,
            {"content-type": "text/html"},
            b"nope",
        )
        assert err.url == "https://x.com"
        assert err.status == 404
        assert err.headers == {"content-type": "text/html"}
        assert err.body == b"nope"
        assert "404" in str(err)

    def test_status_zero_renders_as_connection_failure_not_http_0(self) -> None:
        # RED: status 0 is the internal "no HTTP response" sentinel (timeout,
        # TLS/connect failure). Rendering it as "HTTP 0" leaks the sentinel and
        # misleads -- there is no HTTP status 0. It must read as a connection
        # failure and surface the reason (the body carries it).
        err = FetchError("https://x.com", 0, {}, b"Failed to connect to x.com port 443")
        msg = str(err)
        assert "HTTP 0" not in msg
        # Renders "connection failed: <url>: <reason>" -- assert the URL lands in
        # its slot via the exact prefix (not a bare substring membership check).
        assert msg.startswith("connection failed: https://x.com")
        assert "connect" in msg.lower() or "connection" in msg.lower()


class TestFetchInputValidation:
    """Invalid numeric args are rejected at the boundary with a ValueError.

    Leaked as an internal AssertionError or silent transport-specific behavior.
    """

    def test_negative_retries_rejected(self) -> None:
        # O-WEB-001: retries=-1 -> range(1+-1)=range(0), the loop never runs and
        # the internal "unreachable" AssertionError leaks. Reject up front.
        with pytest.raises(ValueError, match="retries"):
            fetch.fetch(
                "https://example.com",
                request=RequestParams(retry=RetryParams(retries=-1)),
            )

    def test_negative_max_redirects_rejected(self) -> None:
        # O-WEB-007: max_redirects=-1 silently behaves like 0 (never follow),
        # but the contract documents only 0 as "disable". Reject the ambiguous -1.
        with pytest.raises(ValueError, match="max_redirects"):
            fetch.fetch(
                "https://example.com",
                request=RequestParams(retry=RetryParams(max_redirects=-1)),
            )

    def test_nonpositive_timeout_rejected(self) -> None:
        # O-WEB-008: timeout_sec=0 means opposite things per transport (curl 0 =
        # no timeout, stdlib 0 = non-blocking). Reject non-positive timeouts.
        with pytest.raises(ValueError, match="timeout_sec"):
            fetch.fetch(
                "https://example.com",
                request=RequestParams(retry=RetryParams(timeout_sec=0)),
            )


class TestFetchClassifiesBlockAtBoundary:
    """``fetch()`` classifies a 4xx/5xx block ONCE at the boundary and raises the.

    SPECIFIC :class:`BotDetectionError` subclass, so every ``except FetchError``
    consumer gets ``.guidance`` for free instead of re-deriving the kind (some paths
    forgot to, yielding a generic "HTTP 403").

        Mocks at the curl high-level boundary (``curl_cffi.requests.request``), the
        same seam the rest of ``TestFetchCurlBackend`` uses.
    """

    def _mock_403(self, body: bytes, headers: dict[str, str]) -> Mock:
        resp = Mock()
        resp.status_code = 403
        resp.content = body
        resp.headers = headers
        resp.url = "https://x.com/"
        return resp

    def test_cloudflare_403_raises_cloudflare_challenge_error(self) -> None:
        # A CF-fronted 403 with a challenge body: fetch() must raise the specific
        # CloudflareChallengeError -- which is-a BotDetectionError, is-a FetchError
        # -- carrying status/headers/body plus the CF .guidance.
        resp = self._mock_403(
            b"<!DOCTYPE html><html><head><title>Just a moment...</title>"
            b'<div class="challenge-platform"></div></head></html>',
            {"server": "cloudflare", "cf-ray": "a1-LAX"},
        )
        with (
            patch("curl_cffi.requests.request", return_value=resp),
            pytest.raises(CloudflareChallengeError) as exc,
        ):
            fetch.fetch(
                "https://x.com",
                request=RequestParams(policy=PolicyParams(transport="curl")),
            )
        assert isinstance(exc.value, FetchError)
        assert isinstance(exc.value, BotDetectionError)
        assert exc.value.status == 403
        assert exc.value.headers == {"server": "cloudflare", "cf-ray": "a1-LAX"}
        assert b"challenge-platform" in exc.value.body
        assert "cloudflare" in exc.value.guidance.lower()

    def test_recaptcha_403_raises_puzzle_challenge_error(self) -> None:
        # A reCAPTCHA body pins a solve-a-puzzle wall regardless of the CF front.
        resp = self._mock_403(
            b'<div class="g-recaptcha" data-sitekey="x"></div>',
            {"content-type": "text/html"},
        )
        with (
            patch("curl_cffi.requests.request", return_value=resp),
            pytest.raises(PuzzleChallengeError) as exc,
        ):
            fetch.fetch(
                "https://x.com",
                request=RequestParams(policy=PolicyParams(transport="curl")),
            )
        assert exc.value.status == 403
        assert "captcha" in exc.value.guidance.lower()

    def test_genuine_404_raises_plain_fetch_error_not_bot_flag(self) -> None:
        # No markers, non-CF origin: a real 404 must stay a plain FetchError,
        # never a BotDetectionError (else a dead URL looks recoverable).
        resp = Mock()
        resp.status_code = 404
        resp.content = b"<html><body><h1>404 Not Found</h1></body></html>"
        resp.headers = {"server": "nginx"}
        resp.url = "https://x.com/"
        with (
            patch("curl_cffi.requests.request", return_value=resp),
            pytest.raises(FetchError) as exc,
        ):
            fetch.fetch(
                "https://x.com",
                request=RequestParams(policy=PolicyParams(transport="curl")),
            )
        assert not isinstance(exc.value, BotDetectionError)
        assert exc.value.status == 404

    def test_except_fetch_error_catches_the_specific_subclass(self) -> None:
        # The whole point: an existing ``except FetchError`` still catches the
        # newly-specific CloudflareChallengeError (subclass), no call-site change.
        resp = self._mock_403(
            b"<html><head><title>Just a moment...</title></head></html>",
            {"server": "cloudflare", "cf-ray": "b2-LAX"},
        )
        caught: FetchError | None = None
        with (
            patch("curl_cffi.requests.request", return_value=resp),
        ):
            try:
                fetch.fetch(
                    "https://x.com",
                    request=RequestParams(policy=PolicyParams(transport="curl")),
                )
            except FetchError as e:
                caught = e
        assert isinstance(caught, CloudflareChallengeError)


class TestFetchRetry:
    @pytest.fixture(autouse=True)
    def _force_stdlib(self) -> object:
        # Stdlib path is selected per-call via transport="stdlib", not a global.
        return

    def _mock_http_response(
        self,
        status: int = 200,
        body: bytes = b"hello",
        headers: list[tuple[str, str]] | None = None,
    ) -> Mock:
        resp = Mock(spec=client.HTTPResponse)
        resp.status = status
        resp.read.return_value = body
        resp.getheaders.return_value = headers or [
            ("content-encoding", "identity"),
        ]
        return resp

    def test_retries_on_500(self) -> None:
        resp_500 = self._mock_http_response(status=500, body=b"ISE")
        resp_ok = self._mock_http_response(body=b"ok")
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.side_effect = [resp_500, resp_ok]

        with (
            patch(
                "wesearch.fetch.transport.stdlib._open_connection",
                return_value=mock_conn,
            ),
            patch("wesearch.fetch.fetch.time.sleep"),
        ):
            assert (
                fetch.fetch(
                    "https://example.com",
                    request=RequestParams(
                        retry=RetryParams(retries=1),
                        policy=PolicyParams(transport="stdlib"),
                    ),
                )[0]
                == b"ok"
            )

    def test_no_retry_on_404(self) -> None:
        resp = self._mock_http_response(status=404, body=b"NF")
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp
        with (
            patch(
                "wesearch.fetch.transport.stdlib._open_connection",
                return_value=mock_conn,
            ),
            pytest.raises(FetchError, match="404"),
        ):
            fetch.fetch(
                "https://example.com",
                request=RequestParams(
                    retry=RetryParams(retries=3),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )

    def test_error_body_isdecompressed(self) -> None:
        # RED: an error response (e.g. a Cloudflare 403 challenge page) is
        # compressed like any other; the success path decompresses but the error
        # path stored the body RAW, so FetchError.body was undecodable garbage --
        # which is exactly why a challenge page can't be told from a plain 404.
        html = b"<!DOCTYPE html><html>Just a moment...</html>"
        resp = self._mock_http_response(
            status=403,
            body=zstd_compress(html),
            headers=[("content-encoding", "zstd"), ("server", "cloudflare")],
        )
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp
        with (
            patch(
                "wesearch.fetch.transport.stdlib._open_connection",
                return_value=mock_conn,
            ),
            pytest.raises(FetchError) as exc,
        ):
            fetch.fetch(
                "https://x.com",
                request=RequestParams(policy=PolicyParams(transport="stdlib")),
            )
        # The caller must receive readable HTML, not the raw zstd frame.
        assert exc.value.body == html

    def test_retries_on_network_error(self) -> None:
        resp_ok = self._mock_http_response(body=b"ok")
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.side_effect = [OSError("refused"), resp_ok]

        with (
            patch(
                "wesearch.fetch.transport.stdlib._open_connection",
                return_value=mock_conn,
            ),
            patch("wesearch.fetch.fetch.time.sleep"),
        ):
            assert (
                fetch.fetch(
                    "https://example.com",
                    request=RequestParams(
                        retry=RetryParams(retries=1),
                        policy=PolicyParams(transport="stdlib"),
                    ),
                )[0]
                == b"ok"
            )


class TestHeaderOrder:
    """Lock the canonical Chrome header order on the wire.

    http.client emits user-supplied headers in dict insertion order, so
    asserting the dict's key order asserts the wire order. ``Host`` and
    ``Content-Length`` are added by http.client itself (right after the
    request line) and are not part of the user-headers dict here.
    """

    @pytest.fixture(autouse=True)
    def _force_stdlib(self) -> object:
        # Stdlib path is selected per-call via transport="stdlib", not a global.
        return

    def _capture_headers(
        self,
        content: ContentParams = ContentParams(),  # noqa: B008 -- Frozen dataclass; a shared default is safe.
    ) -> dict[str, str]:
        resp = Mock(spec=client.HTTPResponse)
        resp.status = 200
        resp.read.return_value = b"ok"
        resp.getheaders.return_value = [("content-encoding", "identity")]
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ):
            fetch.fetch(
                "https://example.com/",
                request=RequestParams(
                    content=content,
                    policy=PolicyParams(transport="stdlib"),
                ),
            )
        return _recorded_headers(mock_conn.request)

    def test_get_navigation_order(self) -> None:
        # The exact order a real Chrome 146 navigation sends over HTTP/1.1 (the
        # stdlib transport): no Connection header, sec-ch-ua first, and NO
        # Priority -- Priority is an HTTP/2 construct a real Chrome omits on
        # HTTP/1.1 (verified against live Chrome via the parity oracle).
        headers = self._capture_headers()
        # Host leads: trust is honored by default, so the stdlib path pins the
        # connection and must state Host itself rather than let http.client
        # auto-generate it. Real Chrome sends Host first too, and servers that
        # observe header order 403 a trailing one.
        assert list(headers) == [
            "Host",
            "sec-ch-ua",
            "sec-ch-ua-mobile",
            "sec-ch-ua-platform",
            "Upgrade-Insecure-Requests",
            "User-Agent",
            "Accept",
            "Sec-Fetch-Site",
            "Sec-Fetch-Mode",
            "Sec-Fetch-User",
            "Sec-Fetch-Dest",
            "Accept-Encoding",
            "Accept-Language",
        ]
        assert headers["Sec-Fetch-Mode"] == "navigate"
        assert "Chrome/" in headers["User-Agent"]

    def test_user_agent_agrees_with_the_client_hints_beside_it(self) -> None:
        # The client hints are built from the impersonate target while the UA
        # was drawn from the pool, so a Windows Chrome/48 UA rode beside
        # sec-ch-ua-platform "macOS" and sec-ch-ua v="146". Both headers are
        # present and correctly ordered, so no order check can see it -- but a
        # browser that contradicts itself is provably not a browser.
        headers = self._capture_headers()
        user_agent = headers["User-Agent"]
        platform = headers["sec-ch-ua-platform"].strip('"')
        token = {"Windows": "Windows NT", "Linux": "X11; Linux", "macOS": "Macintosh"}
        assert token[platform] in user_agent, (
            f"sec-ch-ua-platform {platform!r} contradicts UA {user_agent!r}"
        )
        major = user_agent.split("Chrome/", 1)[1].split(".", 1)[0]
        assert f'v="{major}"' in headers["sec-ch-ua"], (
            f"sec-ch-ua {headers['sec-ch-ua']!r} contradicts UA Chrome/{major}"
        )

    def test_post_xhr_order_with_json(self) -> None:
        headers = self._capture_headers(ContentParams(method="POST", json={"q": "x"}))
        assert list(headers) == [
            "Host",
            "sec-ch-ua",
            "sec-ch-ua-mobile",
            "sec-ch-ua-platform",
            "User-Agent",
            "Accept",
            "Content-Type",
            "Origin",
            "Sec-Fetch-Site",
            "Sec-Fetch-Mode",
            "Sec-Fetch-Dest",
            "Accept-Encoding",
            "Accept-Language",
        ]
        assert headers["Accept"] == "*/*"
        assert headers["Content-Type"] == "application/json"
        assert headers["Sec-Fetch-Mode"] == "cors"
        assert headers["Origin"] == "https://example.com"
        assert "Upgrade-Insecure-Requests" not in headers

    def test_post_xhr_order_with_form(self) -> None:
        headers = self._capture_headers(ContentParams(method="POST", data={"q": "x"}))
        assert headers["Content-Type"] == "application/x-www-form-urlencoded"
        # Content-Type lives between Accept and Origin.
        keys = list(headers)
        assert keys.index("Content-Type") == keys.index("Accept") + 1
        assert keys.index("Origin") == keys.index("Content-Type") + 1

    def test_post_without_body_omits_content_type(self) -> None:
        headers = self._capture_headers(ContentParams(method="POST"))
        assert "Content-Type" not in headers

    def test_caller_override_preserves_slot(self) -> None:
        headers = self._capture_headers(
            ContentParams(headers={"User-Agent": "Custom/1.0"}),
        )
        keys = list(headers)
        assert headers["User-Agent"] == "Custom/1.0"
        # Slot is the same as the default User-Agent slot (after
        # Upgrade-Insecure-Requests, before Accept).
        assert (
            keys.index("Upgrade-Insecure-Requests")
            < keys.index("User-Agent")
            < keys.index("Accept")
        )

    def test_caller_new_header_appended(self) -> None:
        headers = self._capture_headers(ContentParams(headers={"X-Trace": "abc"}))
        assert list(headers)[-1] == "X-Trace"

    def test_validated_hosts_puts_host_first(self) -> None:
        resp = Mock(spec=client.HTTPResponse)
        resp.status = 200
        resp.read.return_value = b"ok"
        resp.getheaders.return_value = [("content-encoding", "identity")]
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        with patch(
            "wesearch.fetch.transport.stdlib._open_connection",
            return_value=mock_conn,
        ):
            fetch.fetch(
                "https://example.com/",
                request=RequestParams(policy=PolicyParams(transport="stdlib")),
            )

        captured = _recorded_headers(mock_conn.request)
        assert next(iter(captured)) == "Host"
        assert captured["Host"] == "example.com"


class TestOnResponse:
    """``on_response(status, headers)`` fires once per received response.

    On success, on an HTTP error before it raises, and on every redirect hop -- for both
    transports. It is the seam a cookie jar uses to observe Set-Cookie.
    """

    def _stdlib_resp(
        self,
        status: int,
        headers: list[tuple[str, str]],
        body: bytes = b"ok",
    ) -> Mock:
        r = Mock(spec=client.HTTPResponse)
        r.status = status
        r.read.return_value = body
        r.getheaders.return_value = [("content-encoding", "identity"), *headers]
        return r

    def test_stdlib_success_reports_status_and_headers(self) -> None:
        conn = Mock(request=Mock())
        conn.getresponse.return_value = self._stdlib_resp(
            200,
            [("set-cookie", "GSP=abc")],
        )
        seen: list[tuple[int, dict[str, str]]] = []
        with (
            patch(
                "wesearch.fetch.transport.stdlib._open_connection",
                return_value=conn,
            ),
        ):
            fetch.fetch(
                "https://x.com",
                request=RequestParams(
                    observe=ObserveParams(on_response=lambda s, h: seen.append((s, h))),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )
        assert len(seen) == 1
        status, headers = seen[0]
        assert status == 200
        assert headers.get("set-cookie") == "GSP=abc"

    def test_stdlib_fires_per_redirect_hop_then_final(self) -> None:
        redir = self._stdlib_resp(
            302,
            [("location", "https://x.com/2"), ("set-cookie", "a=1")],
            b"",
        )
        final = self._stdlib_resp(200, [("set-cookie", "b=2")])
        conn = Mock(request=Mock())
        conn.getresponse.side_effect = [redir, final]
        seen: list[int] = []
        with (
            patch(
                "wesearch.fetch.transport.stdlib._open_connection",
                return_value=conn,
            ),
        ):
            fetch.fetch(
                "https://x.com/1",
                request=RequestParams(
                    observe=ObserveParams(on_response=lambda s, _h: seen.append(s)),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )
        assert seen == [302, 200]

    def test_stdlib_error_reports_before_raising(self) -> None:
        conn = Mock(request=Mock())
        conn.getresponse.return_value = self._stdlib_resp(
            404,
            [("set-cookie", "x=1")],
            b"nope",
        )
        seen: list[int] = []
        with (
            patch(
                "wesearch.fetch.transport.stdlib._open_connection",
                return_value=conn,
            ),
            pytest.raises(FetchError),
        ):
            fetch.fetch(
                "https://x.com",
                request=RequestParams(
                    observe=ObserveParams(on_response=lambda s, _h: seen.append(s)),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )
        assert seen == [404]

    def test_curl_success_reports_status_and_headers(self) -> None:
        resp = Mock()
        resp.status_code = 200
        resp.content = b"ok"
        resp.headers = {"set-cookie": "GSP=xyz"}
        resp.url = "https://x.com/"
        seen: list[tuple[int, dict[str, str]]] = []
        with (
            patch("curl_cffi.requests.request", return_value=resp),
        ):
            fetch.fetch(
                "https://x.com",
                request=RequestParams(
                    observe=ObserveParams(on_response=lambda s, h: seen.append((s, h))),
                ),
            )
        assert len(seen) == 1
        assert seen[0][0] == 200
        assert seen[0][1].get("set-cookie") == "GSP=xyz"


class TestTransportConsistency:
    """The curl and stdlib transports must behave IDENTICALLY on the redirect.

    Contract (cap -> return 3xx body; cross-origin -> Origin rewritten). These tests run
    the SAME scenario through both and assert equality, so the two remaining redirect
    loops cannot silently drift (the class of bug that recurred across several review
    rounds).
    """

    def _stdlib_result(
        self,
        hops: list[tuple[int, bytes, dict[str, str]]],
        *,
        max_redirects: int = 10,
    ) -> bytes:
        resps: list[Mock] = []
        for status, body, hdrs in hops:
            r = Mock(spec=client.HTTPResponse)
            r.status = status
            r.read.return_value = body
            r.getheaders.return_value = [
                ("content-encoding", "identity"),
                *hdrs.items(),
            ]
            resps.append(r)
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.side_effect = resps
        with (
            patch(
                "wesearch.fetch.transport.stdlib._open_connection",
                return_value=mock_conn,
            ),
        ):
            return fetch.fetch(
                "https://a.com/start",
                request=RequestParams(
                    retry=RetryParams(max_redirects=max_redirects),
                    policy=PolicyParams(transport="stdlib"),
                ),
            )[0]

    def _curl_result(
        self,
        hops: list[tuple[int, bytes, dict[str, str]]],
        *,
        max_redirects: int = 10,
    ) -> bytes:
        resps: list[Mock] = []
        for status, body, hdrs in hops:
            r = Mock()
            r.status_code = status
            r.content = body
            r.headers = hdrs
            resps.append(r)
        with (
            patch("curl_cffi.requests.request", side_effect=resps),
        ):
            return fetch.fetch(
                "https://a.com/start",
                request=RequestParams(retry=RetryParams(max_redirects=max_redirects)),
            )[0]

    def test_cap_returns_3xx_body_identically(self) -> None:
        # max_redirects=0: both transports return the 3xx body, neither raises.
        hops: list[tuple[int, bytes, dict[str, str]]] = [
            (302, b"the 3xx body", {"location": "https://a.com/next"}),
        ]
        assert self._stdlib_result(hops, max_redirects=0) == b"the 3xx body"
        assert self._curl_result(hops, max_redirects=0) == b"the 3xx body"

    def test_followed_redirect_returns_final_body_identically(self) -> None:
        hops: list[tuple[int, bytes, dict[str, str]]] = [
            (302, b"", {"location": "https://a.com/2"}),
            (200, b"final", {}),
        ]
        assert self._stdlib_result(hops) == b"final"
        assert self._curl_result(hops) == b"final"


class TestFetchSession:
    """``FetchSession`` is a frozen browsing identity a caller threads across.

    Requests: ``fetch`` returns the session updated with what each response taught it
    (cookies set, ``Accept-CH`` opt-ins), so the next request is more browser-like --
    the value-typed, functional API for reuse.
    """

    def _curl_response(
        self,
        *,
        headers: dict[str, str],
        content: bytes = b"ok",
    ) -> Mock:
        resp = Mock()
        resp.status_code = 200
        resp.content = content
        resp.headers = headers
        resp.url = "https://x.com/"
        return resp

    def test_defaults_are_empty_and_frozen(self) -> None:
        session = FetchSession()
        assert session.impersonate == "chrome"
        assert dict(session.cookies) == {}
        assert dict(session.accept_ch) == {}
        with pytest.raises(AttributeError):
            session.impersonate = "firefox"  # ty: ignore[invalid-assignment] -- The negative test writes a deliberately invalid frozen field.  # pyright: ignore[reportAttributeAccessIssue] -- The negative test writes a deliberately invalid frozen field.

    def test_with_cookies_returns_a_merged_copy(self) -> None:
        base = FetchSession(cookies={"https://x.com": {"a": "1"}})
        updated = base.with_cookies("https://x.com/p", {"b": "2"})
        assert updated.cookies_for("https://x.com/q") == {"a": "1", "b": "2"}
        assert base.cookies_for("https://x.com/q") == {"a": "1"}  # Original unchanged.

    def test_cookies_are_scoped_to_the_setting_origin(self) -> None:
        # A flat name->value jar sent a cookie one host set to the NEXT host
        # fetched with the same session, leaking a session id across origins.
        session = FetchSession().with_cookies("https://a.example/", {"SID": "secret"})
        assert session.cookies_for("https://a.example/other") == {"SID": "secret"}
        assert session.cookies_for("https://b.example/") == {}

    def test_scoped_jar_survives_serialization(self) -> None:
        session = FetchSession().with_cookies("https://a.example/", {"SID": "s"})
        restored = FetchSession.deserialize(session.serialize())
        assert restored.cookies_for("https://a.example/") == {"SID": "s"}
        assert restored.cookies_for("https://b.example/") == {}

    def test_serialize_emits_exact_session_shape(self) -> None:
        session = FetchSession(
            impersonate="chrome146",
            cookies={"https://a.example": {"SID": "s"}},
            accept_ch={
                "https://a.example": frozenset({"sec-ch-ua-bitness", "sec-ch-ua-arch"}),
            },
        )
        assert session.serialize() == {
            "impersonate": "chrome146",
            "cookies": {"https://a.example": {"SID": "s"}},
            "accept_ch": {
                "https://a.example": ["sec-ch-ua-arch", "sec-ch-ua-bitness"],
            },
        }

    def test_deserialize_rebuilds_every_session_field_and_defaults(self) -> None:
        assert FetchSession.deserialize(
            {
                "impersonate": "chrome146",
                "cookies": {"https://a.example": {"SID": "s"}},
                "accept_ch": {"https://a.example": ["sec-ch-ua-arch"]},
            },
        ) == FetchSession(
            impersonate="chrome146",
            cookies={"https://a.example": {"SID": "s"}},
            accept_ch={"https://a.example": frozenset({"sec-ch-ua-arch"})},
        )
        assert FetchSession.deserialize({}) == FetchSession()

    def test_with_accept_ch_ignores_empty_and_duplicate_hints(self) -> None:
        session = FetchSession()
        assert session.with_accept_ch("https://x.com", frozenset()) is session
        warmed = session.with_accept_ch("https://x.com", frozenset({"x"}))
        assert warmed.with_accept_ch("https://x.com", frozenset({"x"})) is warmed

    def test_with_accept_ch_records_origin_opt_in(self) -> None:
        session = FetchSession().with_accept_ch(
            "https://x.com",
            frozenset({"sec-ch-ua-arch"}),
        )
        assert session.accept_ch["https://x.com"] == frozenset({"sec-ch-ua-arch"})

    def test_fetch_session_returns_body_and_session(self) -> None:
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={}),
        ):
            body, _ = fetch.fetch("https://x.com/p")
        assert body == b"ok"

    def test_session_learns_set_cookie(self) -> None:
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={"set-cookie": "GSP=z; Path=/"}),
        ):
            _body, session = fetch.fetch("https://x.com/p")
        assert session.cookies_for("https://x.com/p")["GSP"] == "z"

    def test_session_learns_accept_ch(self) -> None:
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(
                headers={"accept-ch": "Sec-CH-UA-Arch, Sec-CH-UA-Bitness"},
            ),
        ):
            _body, session = fetch.fetch("https://x.com/p")
        assert session.accept_ch["https://x.com"] == frozenset(
            {"sec-ch-ua-arch", "sec-ch-ua-bitness"},
        )

    def test_threaded_accept_ch_emits_extended_hints(self) -> None:
        # A session that opted into Accept-CH must, on the NEXT request to that
        # origin, send exactly those extended client hints -- the behavior once
        # backed by a module global, now threaded through the session.
        prior = FetchSession().with_accept_ch(
            "https://x.com",
            frozenset({"sec-ch-ua-arch", "sec-ch-ua-bitness"}),
        )
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={}),
        ) as req:
            fetch.fetch("https://x.com/p", session=prior)
        sent = _recorded_headers(req)
        assert "sec-ch-ua-arch" in sent
        assert "sec-ch-ua-bitness" in sent
        assert "sec-ch-ua-model" not in sent  # Never opted in.

    def test_cold_origin_sends_no_extended_hints(self) -> None:
        # A fresh session (no Accept-CH opt-in) sends none of the extended hints,
        # exactly as Chrome's first request to an origin does.
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={}),
        ) as req:
            fetch.fetch(
                "https://x.com/p",
                request=RequestParams(policy=PolicyParams(transport="curl")),
            )
        sent = _recorded_headers(req)
        assert "sec-ch-ua-arch" not in sent

    def test_threaded_session_seeds_prior_cookies(self) -> None:
        # Prior session cookies are loaded into the pooled jar (the single cookie
        # source on the curl path), not the Cookie header.
        prior = FetchSession(cookies={"https://x.com": {"SID": "abc"}})
        stub = StubSession()
        with (
            patch(
                "curl_cffi.requests.request",
                return_value=self._curl_response(headers={}),
            ),
            patch.object(fetch_mod, "curl_session", const_curl_session(stub)),
        ):
            fetch.fetch("https://x.com/p", session=prior)
        assert ("SID", "abc") in {(c.name, c.value) for c in stub.cookies.jar}

    def test_caller_on_response_still_fires(self) -> None:
        seen: list[int] = []
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={"set-cookie": "a=1"}),
        ):
            fetch.fetch(
                "https://x.com/p",
                request=RequestParams(
                    observe=ObserveParams(on_response=lambda s, _h: seen.append(s)),
                ),
            )
        assert seen == [200]


class TestRedirectIdentityScoping:
    """Cross-origin redirects must re-scope every origin-bound identity element.

    A real browser, following a redirect to a NEW origin, does not carry the
    source origin's Cookie header or extended client hints to the target, does
    not attribute the target's Set-Cookie to the source, and downgrades a
    301/302 POST to a bodyless GET. These tests drive the curl backend through a
    two-hop redirect and assert each of those rules on the second hop.
    """

    # 200.
    def _two_hop(
        self,
        *,
        first_status: int,
        target_set_cookie: str | None = None,
    ) -> Callable[..., Mock]:
        """Return a curl ``request`` mock: a.com/start -> (status) -> b.com/next ->."""

        def fake_request(verb: str, url: str, **_kw: object) -> Mock:
            del verb
            resp = Mock()
            if url == "https://a.com/start":
                resp.status_code = first_status
                resp.headers = {"location": "https://b.com/next"}
                resp.content = b""
            else:
                resp.status_code = 200
                resp.headers = (
                    {"set-cookie": target_set_cookie} if target_set_cookie else {}
                )
                resp.content = b"done"
            resp.url = url
            return resp

        return fake_request

    def test_302_post_downgrades_to_bodyless_get(self) -> None:
        # A 301/302 POST must convert to a bodyless GET on the next hop (browser
        # behavior; only 307/308 preserve the method). Currently only 303 does.
        calls: list[tuple[str, str, object]] = []

        def fake_request(verb: str, url: str, **kw: object) -> Mock:
            calls.append((verb, url, kw.get("data")))
            resp = Mock()
            if url == "https://a.com/start":
                resp.status_code = 302
                resp.headers = {"location": "https://a.com/land"}
                resp.content = b""
            else:
                resp.status_code = 200
                resp.headers = {}
                resp.content = b"done"
            resp.url = url
            return resp

        with (
            patch("curl_cffi.requests.request", side_effect=fake_request),
            patch.object(fetch_mod, "egress_ip", return_value=None),
        ):
            fetch.fetch(
                "https://a.com/start",
                request=RequestParams(
                    content=ContentParams(method="POST", data={"x": "1"}),
                ),
            )
        # Second hop must be a GET with no body.
        _verb, _url, second_body = calls[1]
        assert calls[1][0] == "GET"
        assert second_body is None

    def test_cross_origin_redirect_drops_cookie_header(self) -> None:
        # a.com's session cookie must NOT be sent to b.com after a cross-origin
        # redirect (a real browser scopes cookies to their origin).
        sent: list[tuple[str, dict[str, str]]] = []

        def fake_request(verb: str, url: str, **kw: object) -> Mock:
            del verb
            sent.append((url, lower_headers(kw)))
            resp = Mock()
            if url == "https://a.com/start":
                resp.status_code = 302
                resp.headers = {"location": "https://b.com/next"}
                resp.content = b""
            else:
                resp.status_code = 200
                resp.headers = {}
                resp.content = b"done"
            resp.url = url
            return resp

        with (
            patch("curl_cffi.requests.request", side_effect=fake_request),
            patch.object(fetch_mod, "egress_ip", return_value=None),
        ):
            fetch.fetch(
                "https://a.com/start",
                session=FetchSession(cookies={"https://a.com": {"SID": "secret"}}),
            )
        b_headers = next(h for url, h in sent if url == "https://b.com/next")
        assert "cookie" not in b_headers

    def test_same_origin_redirect_keeps_cookie_header(self) -> None:
        # A same-origin redirect must PRESERVE the cookie (the scoping rule only
        # drops on origin change).
        sent: list[tuple[str, dict[str, str]]] = []

        def fake_request(verb: str, url: str, **kw: object) -> Mock:
            del verb
            sent.append((url, lower_headers(kw)))
            resp = Mock()
            if url == "https://a.com/start":
                resp.status_code = 302
                resp.headers = {"location": "https://a.com/next"}
                resp.content = b""
            else:
                resp.status_code = 200
                resp.headers = {}
                resp.content = b"done"
            resp.url = url
            return resp

        with (
            patch("curl_cffi.requests.request", side_effect=fake_request),
            patch.object(fetch_mod, "egress_ip", return_value=None),
        ):
            fetch.fetch(
                "https://a.com/start",
                session=FetchSession(cookies={"https://a.com": {"SID": "secret"}}),
            )
        next_headers = next(h for url, h in sent if url == "https://a.com/next")
        assert next_headers.get("cookie") == "SID=secret"

    def test_cross_origin_redirect_cookies_land_on_their_own_origin(self) -> None:
        """A redirect target's Set-Cookie warms THAT origin, not the requester's.

        They were discarded outright while the jar was flat, because the only
        alternative then was mis-attributing them to the requesting origin. The
        per-origin jar can express the browser behaviour, so it should.
        """

        def fake_request(verb: str, url: str, **kw: object) -> Mock:
            del verb
            del kw
            resp = Mock()
            if url == "https://a.com/start":
                resp.status_code = 302
                resp.headers = {"location": "https://b.com/next"}
                resp.content = b""
            else:
                resp.status_code = 200
                resp.headers = {"set-cookie": "B=1; Path=/"}
                resp.content = b"done"
            resp.url = url
            return resp

        with (
            patch("curl_cffi.requests.request", side_effect=fake_request),
            patch.object(fetch_mod, "egress_ip", return_value=None),
        ):
            _body, session = fetch.fetch(
                "https://a.com/start",
                request=RequestParams(policy=PolicyParams(transport="curl")),
            )
        assert session.cookies_for("https://b.com/x") == {"B": "1"}
        assert session.cookies_for("https://a.com/x") == {}

    def test_cross_origin_redirect_drops_extended_hints(self) -> None:
        # a.com's opted-in extended client hints must NOT leak to b.com.
        sent: list[tuple[str, dict[str, str]]] = []

        def fake_request(verb: str, url: str, **kw: object) -> Mock:
            del verb
            sent.append((url, lower_headers(kw)))
            resp = Mock()
            if url == "https://a.com/start":
                resp.status_code = 302
                resp.headers = {"location": "https://b.com/next"}
                resp.content = b""
            else:
                resp.status_code = 200
                resp.headers = {}
                resp.content = b"done"
            resp.url = url
            return resp

        session = FetchSession().with_accept_ch(
            "https://a.com",
            frozenset({"sec-ch-ua-arch", "sec-ch-ua-bitness"}),
        )
        with (
            patch("curl_cffi.requests.request", side_effect=fake_request),
            patch.object(fetch_mod, "egress_ip", return_value=None),
        ):
            fetch.fetch("https://a.com/start", session=session)
        b_headers = next(h for url, h in sent if url == "https://b.com/next")
        assert "sec-ch-ua-arch" not in b_headers

    def test_cross_origin_target_cookie_not_persisted_to_source_profile(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # b.com's Set-Cookie must NOT be stored in a.com's (egress,domain) profile.
        store = ProfileStore(base_dir=tmp_path)

        def _fixed_egress(**_kw: object) -> str:
            return "9.9.9.9"

        def _no_pool(*_a: object, **_kw: object) -> None:
            return None

        monkeypatch.setattr(ProfileStore, "shared", classmethod(_shared(store)))
        monkeypatch.setattr(fetch_mod, "egress_ip", _fixed_egress)
        monkeypatch.setattr(fetch_mod, "curl_session", _no_pool)
        with patch(
            "curl_cffi.requests.request",
            side_effect=self._two_hop(
                first_status=302,
                target_set_cookie="FOREIGN=1; Path=/",
            ),
        ):
            fetch.fetch("https://a.com/start", request=RequestParams())
        profile = store.load("9.9.9.9", "a.com")
        assert profile is not None
        assert "FOREIGN" not in profile.cookies

    def test_cross_origin_target_cookie_not_attributed_to_source_session(
        self,
    ) -> None:
        # The returned session must not record b.com's cookie under a.com.
        with (
            patch(
                "curl_cffi.requests.request",
                side_effect=self._two_hop(
                    first_status=302,
                    target_set_cookie="FOREIGN=1; Path=/",
                ),
            ),
            patch.object(fetch_mod, "egress_ip", return_value=None),
        ):
            _body, session = fetch.fetch("https://a.com/start")
        # a.com is the request origin; FOREIGN belongs to b.com, not a.com's jar.
        assert "FOREIGN" not in session.cookies

    def test_browser_target_cookie_is_not_attributed_to_the_source(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The browser leg must scope a redirect target's cookies like curl does.

        Chrome follows its own redirects, so the transport harvests cookies for
        where it LANDED. Filing them under the requested domain -- which the
        caller has no way to know is wrong -- stored ``b.example``'s session
        cookie in ``a.example``'s profile and jar, and the next fetch to
        ``a.example`` sent it there.

        Args:
          tmp_path: Temporary profile-store directory.
          monkeypatch: Test monkeypatch fixture.

        """
        store = ProfileStore(base_dir=tmp_path)

        def fixed_egress(**_kw: object) -> str:
            return "9.9.9.9"

        def landed_elsewhere(*_a: object, **_kw: object) -> BrowserResult:
            return BrowserResult(
                body=b"<html>ok</html>",
                cookies={"B_SESSION": "secret"},
                final_url="https://b.example/landing",
            )

        monkeypatch.setattr(ProfileStore, "shared", classmethod(_shared(store)))
        monkeypatch.setattr(fetch_mod, "egress_ip", fixed_egress)
        monkeypatch.setattr(zendriver, "fetch_zendriver", landed_elsewhere)
        _body, session = fetch.fetch(
            "https://a.example/start",
            request=RequestParams(policy=PolicyParams(transport="zendriver")),
        )
        assert "B_SESSION" not in session.cookies_for("https://a.example/start")
        assert session.cookies_for("https://b.example/landing") == {
            "B_SESSION": "secret",
        }
        assert store.load("9.9.9.9", "a.example") is None


class TestIdentityLayer:
    """``fetch`` transparently backs each call with a persistent per-(egress.

    Domain) identity: it seeds the stored UA + cookies (caller values win), saves ``Set-
    Cookie`` back, and on a bot-block of a KNOWN identity discards it and retries once
    fresh. The ``isolate_profiles`` fixture pins egress to ``203.0.113.1`` and points
    the store at a tmp dir.
    """

    _EGRESS = "203.0.113.1"

    def _curl_response(
        self,
        *,
        status: int = 200,
        content: bytes = b"ok",
        headers: dict[str, str],
    ) -> Mock:
        resp = Mock()
        resp.status_code = status
        resp.content = content
        resp.headers = headers
        resp.url = "https://x.com/"
        return resp

    def _store(self) -> ProfileStore:
        return ProfileStore.shared()

    def test_delegates_ua_and_cookie_jar_to_curl_session(self) -> None:
        # On the curl path curl_cffi's impersonate emits a coherent User-Agent
        # (matching its TLS fingerprint), so fetch does NOT send a User-Agent
        # header. The stored jar is NOT seeded into the Cookie header either --
        # the pooled curl session's own jar persists + resends cookies, so
        # header-seeding them too would duplicate the Cookie header (a bot tell).
        self._store().save(
            self._EGRESS,
            "x.com",
            Profile(ua="StoredUA/9", cookies={"GSP": "s"}),
        )
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={}),
        ) as req:
            fetch.fetch(
                "https://x.com/p",
                request=RequestParams(policy=PolicyParams(transport="curl")),
            )
        sent = _recorded_headers(req)
        assert "User-Agent" not in sent
        assert "Cookie" not in sent  # `jar` carries the stored cookie, not the header.

    def test_caller_ua_and_cookie_override_profile(self) -> None:
        self._store().save(
            self._EGRESS,
            "x.com",
            Profile(ua="StoredUA/9", cookies={"GSP": "s"}),
        )
        stub = StubSession()
        with (
            patch(
                "curl_cffi.requests.request",
                return_value=self._curl_response(headers={}),
            ) as req,
            patch.object(fetch_mod, "curl_session", const_curl_session(stub)),
        ):
            fetch.fetch(
                "https://x.com/p",
                request=RequestParams(
                    content=ContentParams(
                        headers={"User-Agent": "Mine/1"},
                        cookies={"GSP": "caller"},
                    ),
                ),
            )
        sent = _recorded_headers(req)
        assert sent["User-Agent"] == "Mine/1"
        # The caller cookie overrides the profile's GSP in the jar (single source).
        assert ("GSP", "caller") in {(c.name, c.value) for c in stub.cookies.jar}
        assert ("GSP", "s") not in {(c.name, c.value) for c in stub.cookies.jar}

    def test_no_profile_delegates_ua_to_impersonate(self) -> None:
        # First contact, no profile: still no seeded User-Agent header on the
        # curl path -- impersonate supplies a coherent one at the transport.
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={}),
        ) as req:
            fetch.fetch(
                "https://x.com/p",
                request=RequestParams(policy=PolicyParams(transport="curl")),
            )
        assert "User-Agent" not in _recorded_headers(req)

    def test_set_cookie_is_persisted(self) -> None:
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(
                headers={"set-cookie": "GSP=minted; Path=/"},
            ),
        ):
            fetch.fetch(
                "https://x.com/p",
                request=RequestParams(policy=PolicyParams(transport="curl")),
            )
        got = self._store().load(self._EGRESS, "x.com")
        assert got is not None
        assert got.cookies == {"GSP": "minted"}

    def test_caller_on_response_still_fires(self) -> None:
        seen: list[int] = []
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={"set-cookie": "a=1"}),
        ):
            fetch.fetch(
                "https://x.com/p",
                request=RequestParams(
                    observe=ObserveParams(on_response=lambda s, _h: seen.append(s)),
                ),
            )
        assert seen == [200]

    def test_burn_on_known_identity_discards_and_retries_fresh(self) -> None:
        self._store().save(
            self._EGRESS,
            "x.com",
            Profile(ua="PoisonUA", cookies={"GSP": "old"}),
        )
        blocked = self._curl_response(
            status=403,
            content=(b'<div class="g-recaptcha" data-sitekey="x"></div>'),
            headers={"content-type": "text/html"},
        )
        ok = self._curl_response(content=b"ok", headers={})
        with patch("curl_cffi.requests.request", side_effect=[blocked, ok]) as req:
            body, _ = fetch.fetch("https://x.com/p")
        assert body == b"ok"
        assert req.call_count == 2
        # The retry used a fresh identity: no poisoned cookies ride along (the UA
        # is curl's coherent impersonate UA, never seeded, so it cannot leak).
        retry_headers = _recorded_headers(req, index=1)
        assert "GSP=old" not in retry_headers.get("Cookie", "")
        # The poisoned identity was discarded and a fresh one saved.
        got = self._store().load(self._EGRESS, "x.com")
        assert got is not None
        assert "GSP" not in got.cookies

    def test_second_burn_raises(self) -> None:
        self._store().save(self._EGRESS, "x.com", Profile(ua="U", cookies={"GSP": "x"}))
        blocked = self._curl_response(
            status=403,
            content=b'<div class="g-recaptcha" data-sitekey="x"></div>',
            headers={"content-type": "text/html"},
        )
        with (
            patch("curl_cffi.requests.request", return_value=blocked),
            pytest.raises(PuzzleChallengeError),
        ):
            fetch.fetch(
                "https://x.com/p",
                request=RequestParams(policy=PolicyParams(transport="curl")),
            )

    def test_first_contact_burn_does_not_retry(self) -> None:
        blocked = self._curl_response(
            status=403,
            content=b'<div class="g-recaptcha" data-sitekey="x"></div>',
            headers={"content-type": "text/html"},
        )
        with (
            patch("curl_cffi.requests.request", return_value=blocked) as req,
            pytest.raises(PuzzleChallengeError),
        ):
            fetch.fetch(
                "https://x.com/p",
                request=RequestParams(policy=PolicyParams(transport="curl")),
            )
        assert req.call_count == 1  # No retry with no known identity.

    def test_raw_headers_bypasses_identity(self) -> None:
        self._store().save(
            self._EGRESS,
            "x.com",
            Profile(ua="StoredUA", cookies={"GSP": "s"}),
        )
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={}),
        ) as req:
            fetch.fetch(
                "https://x.com/p",
                request=RequestParams(
                    content=ContentParams(
                        headers={"User-Agent": "raw"},
                        raw_headers=True,
                    ),
                ),
            )
        sent = _recorded_headers(req)
        assert sent == {"User-Agent": "raw"}  # No profile UA, no stored cookie.

    def test_send_as_keyless_when_egress_none(self, tmp_path: Path) -> None:
        # _send_as with egress=None draws a UA, sends, persists nothing.
        request = _Request(
            url="https://x.com/p",
            session=FetchSession(impersonate="chrome"),
            params=RequestParams(policy=PolicyParams(transport="curl")),
        )
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={"set-cookie": "GSP=z"}),
        ):
            body = _send_as(request, None, None, None, None)
        assert body == b"ok"
        assert not list(tmp_path.glob("*.json"))


class TestEgressIp:
    """``egress_ip`` probes an echo cascade for the host's public IP.

    Memoizing into the last-known global; ``cache=True`` reads it without a network
    call, ``cache=False`` refreshes it, ``last_known_egress_ip`` is a pure read.
    """

    @pytest.fixture(autouse=True)
    def _real_egress(self, monkeypatch: pytest.MonkeyPatch) -> object:
        # The module isolate_profiles fixture stubs egress_ip to a fixed value;
        # restore the REAL function here and just reset the last-known global.
        monkeypatch.setattr(fetch_mod, "egress_ip", egress_ip)
        monkeypatch.setattr(fetch_mod, "_last_egress_ip", None)
        return

    def _probe(self, fetch_mock: Mock, *, ipv6: bool = False) -> str | None:
        # egress_ip unpacks fetch's (body, session) tuple; adapt the byte-valued
        # mock so a bytes return becomes (bytes, session) and an exception still
        # raises (the echo-cascade paths this test exercises).
        def adapt(*args: object, **kwargs: object) -> tuple[bytes, FetchSession]:
            return cast(bytes, fetch_mock(*args, **kwargs)), FetchSession()

        with patch.object(fetch_mod, "fetch", side_effect=adapt):
            return egress_ip(cache=False, ipv6=ipv6)

    def test_first_echo_returned(self) -> None:
        assert self._probe(Mock(return_value=b" 203.0.113.7\n")) == "203.0.113.7"

    def test_probe_builds_exact_raw_echo_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        echo = Mock(return_value=(b"203.0.113.7", FetchSession()))
        monkeypatch.setattr(fetch_mod, "fetch", echo)
        monkeypatch.setattr(fetch_mod, "set_last_egress_ip", Mock())
        assert (
            egress_ip(
                cache=False,
                v4_echoes=("https://echo.example",),
                timeout_sec=7.0,
            )
            == "203.0.113.7"
        )
        request = echo.call_args.kwargs["request"]
        assert isinstance(request, RequestParams)
        assert request.content.headers == {}
        assert request.content.raw_headers is True
        assert request.retry.timeout_sec == 7.0

    def test_probe_default_timeout_is_five_seconds(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        echo = Mock(return_value=(b"203.0.113.7", FetchSession()))
        monkeypatch.setattr(fetch_mod, "fetch", echo)
        monkeypatch.setattr(fetch_mod, "set_last_egress_ip", Mock())
        assert (
            egress_ip(cache=False, v4_echoes=("https://echo.example",)) == "203.0.113.7"
        )
        request = echo.call_args.kwargs["request"]
        assert isinstance(request, RequestParams)
        assert request.retry.timeout_sec == 5.0

    def test_probe_selects_v4_echoes_when_ipv6_is_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        echo = Mock(return_value=(b"203.0.113.7", FetchSession()))
        monkeypatch.setattr(fetch_mod, "fetch", echo)
        monkeypatch.setattr(fetch_mod, "set_last_egress_ip", Mock())
        assert (
            egress_ip(
                cache=False,
                ipv6=False,
                v4_echoes=("https://v4.example",),
                v6_echoes=("https://v6.example",),
            )
            == "203.0.113.7"
        )
        assert echo.call_args.args[0] == "https://v4.example"

    def test_non_v4_reply_falls_through(self) -> None:
        assert self._probe(Mock(side_effect=[b"2001:db8::1", b"198.51.100.9"])) == (
            "198.51.100.9"
        )

    def test_fetch_error_falls_through(self) -> None:
        err = FetchError(url="u", status=500, headers={}, body=b"")
        assert self._probe(Mock(side_effect=[err, b"192.0.2.5"])) == "192.0.2.5"

    def test_all_fail_resolves_none(self) -> None:
        assert self._probe(Mock(side_effect=OSError("offline"))) is None

    def test_v6_echo_returned(self) -> None:
        assert (
            self._probe(Mock(return_value=b"2606:4700:4700::1111\n"), ipv6=True)
            == "2606:4700:4700::1111"
        )

    def test_v4_reply_rejected_for_v6_request(self) -> None:
        assert self._probe(Mock(return_value=b"203.0.113.7"), ipv6=True) is None

    def test_uses_v6_endpoints(self) -> None:
        mock = Mock(return_value=b"2001:db8::5")
        self._probe(mock, ipv6=True)
        url = _recorded_url(mock)
        assert "ipv6" in url or "api64" in url

    def test_malformed_v6_reply_rejected(self) -> None:
        assert self._probe(Mock(return_value=b"::::"), ipv6=True) is None
        assert self._probe(Mock(return_value=b"ff:"), ipv6=True) is None

    def test_probe_records_last_known(self) -> None:
        assert last_known_egress_ip() is None
        self._probe(Mock(return_value=b"203.0.113.7"))
        assert last_known_egress_ip() == "203.0.113.7"

    def test_cache_true_returns_last_known_without_probing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(fetch_mod, "_last_egress_ip", "9.9.9.9")
        echo = Mock()
        with patch.object(fetch_mod, "fetch", echo):
            assert egress_ip() == "9.9.9.9"
        echo.assert_not_called()

    def test_cache_true_probes_to_fill_empty(self) -> None:
        echo = Mock(return_value=(b"1.2.3.4", FetchSession()))
        with patch.object(fetch_mod, "fetch", echo):
            assert egress_ip() == "1.2.3.4"
        assert echo.call_count == 1
        assert last_known_egress_ip() == "1.2.3.4"

    def test_cache_false_always_probes_and_refreshes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(fetch_mod, "_last_egress_ip", "1.1.1.1")
        with patch.object(
            fetch_mod,
            "fetch",
            Mock(return_value=(b"2.2.2.2", FetchSession())),
        ):
            assert egress_ip(cache=False) == "2.2.2.2"
        assert last_known_egress_ip() == "2.2.2.2"

    def test_failed_probe_leaves_last_known_untouched(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(fetch_mod, "_last_egress_ip", "keepme")
        with patch.object(fetch_mod, "fetch", Mock(side_effect=OSError("x"))):
            assert egress_ip(cache=False) is None
        assert last_known_egress_ip() == "keepme"

    def test_set_last_egress_ip_injects_without_probing(self) -> None:
        # A caller who knows the egress (e.g. just rolled the VPN) can set it;
        # a cached read then returns it with no network.
        close = Mock()
        with patch.object(fetch_mod, "close_curl_sessions_except", close):
            set_last_egress_ip("5.5.5.5")
        close.assert_called_once_with("5.5.5.5")
        echo = Mock()
        with patch.object(fetch_mod, "fetch", echo):
            assert egress_ip() == "5.5.5.5"
        echo.assert_not_called()
        assert last_known_egress_ip() == "5.5.5.5"


class TestBrowserBackend:
    """The opt-in ``transport="zendriver"`` path and its parameter guards."""

    def test_rejects_non_get_method(self) -> None:
        with pytest.raises(ValueError, match="zendriver backend supports only GET"):
            RequestParams(
                content=ContentParams(method="POST"),
                policy=PolicyParams(transport="zendriver"),
            )

    def test_rejects_request_body(self) -> None:
        with pytest.raises(ValueError, match="cannot send a request body"):
            RequestParams(
                content=ContentParams(data={"a": "1"}),
                policy=PolicyParams(transport="zendriver"),
            )

    def test_accepts_untrusted_trust(self) -> None:
        # Regression: an SSRF-validated browser request used to be rejected
        # outright, which cost the only caller that asked for safety its entire
        # browser path. Chrome owns its DNS, so "untrusted" validates the host
        # and declines to pin -- it does not refuse the request.
        params = RequestParams(policy=PolicyParams(transport="zendriver"))
        assert params.policy.trust == "untrusted"

    def test_default_transport_is_auto(self) -> None:
        assert RequestParams().policy.transport == "auto"

    def test_auto_uses_general_curl_then_browser_fallback(self) -> None:
        assert resolve_transport("auto") == "curl-then-zendriver"

    def test_auto_uses_curl_for_post(self) -> None:
        assert resolve_transport("auto", method="POST") == "curl"

    def test_auto_uses_curl_for_get_body(self) -> None:
        with patch.object(
            fetch_mod,
            "_fetch_with_identity",
            return_value=b"ok",
        ) as direct:
            body, _ = fetch.fetch(
                "https://google.com/api",
                request=RequestParams(content=ContentParams(json={"query": "value"})),
            )
        assert body == b"ok"
        assert _recorded_request(direct).policy.transport == "curl"

    def test_auto_post_to_learned_domain_uses_curl(self) -> None:
        # A domain learned to require the browser must not override method/body
        # eligibility: an automatic POST to it resolves to curl, not the GET-only
        # zendriver leg (whose construction raises "supports only GET").
        with (
            patch.object(
                transport_routing,
                "zendriver_domains",
                return_value=frozenset({"walled.example"}),
            ),
            patch.object(
                fetch_mod,
                "_fetch_with_identity",
                return_value=b"ok",
            ) as direct,
        ):
            body, _ = fetch.fetch(
                "https://walled.example/api",
                request=RequestParams(content=ContentParams(json={"q": "v"})),
            )
        assert body == b"ok"
        assert _recorded_request(direct).policy.transport == "curl"

    def test_browser_fetch_observes_a_cookie_free_response(self) -> None:
        # ObserveParams.on_response promises a callback for EVERY response; gating it
        # on a non-empty jar meant a successful cookie-free browser fetch told
        # the caller nothing at all.
        seen: list[tuple[int, dict[str, str]]] = []
        with (
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch(
                "wesearch.fetch.fetch.zendriver.fetch_zendriver",
                return_value=BrowserResult(body=b"ok", cookies={}, final_url=""),
            ),
        ):
            fetch.fetch(
                "https://walled.example/x",
                request=RequestParams(
                    observe=ObserveParams(on_response=lambda s, h: seen.append((s, h))),
                    policy=PolicyParams(transport="zendriver"),
                ),
            )
        assert len(seen) == 1

    def test_browser_fetch_forwards_url_params_headers_and_cookies(self) -> None:
        result = BrowserResult(body=b"ok", cookies={}, final_url="")
        with (
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch(
                "wesearch.fetch.fetch.zendriver.fetch_zendriver",
                return_value=result,
            ) as via,
        ):
            fetch.fetch(
                "https://google.com/search?hl=en",
                session=FetchSession(
                    cookies={"https://google.com": {"SID": "session"}},
                ),
                request=RequestParams(
                    content=ContentParams(
                        params={"q": "test query"},
                        headers={"X-Test": "yes"},
                        cookies={"CONSENT": "YES+"},
                    ),
                    policy=PolicyParams(transport="zendriver"),
                ),
            )
        assert via.call_args.args[0] == ("https://google.com/search?hl=en&q=test+query")
        assert via.call_args.kwargs["headers"] == {"X-Test": "yes"}
        assert via.call_args.kwargs["cookies"] == {
            "SID": "session",
            "CONSENT": "YES+",
        }
        assert "resolve_host" not in via.call_args.kwargs

    def test_browser_fetch_forwards_the_redirect_budget(self) -> None:
        # The header transports receive RetryParams.max_redirects; the browser leg
        # did not, so Chrome followed its own default of 10 hops whatever the caller
        # set -- including 0, which RetryParams documents as "no redirects".
        with (
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch(
                "wesearch.fetch.fetch.zendriver.fetch_zendriver",
                return_value=BrowserResult(body=b"ok", cookies={}, final_url=""),
            ) as via,
        ):
            fetch.fetch(
                "https://walled.example/x",
                request=RequestParams(
                    retry=RetryParams(max_redirects=0),
                    policy=PolicyParams(transport="zendriver"),
                ),
            )
        assert via.call_args.kwargs.get("max_redirects") == 0

    def test_browser_fetch_returns_body_and_warms_session(self) -> None:
        # A browser fetch must return the rendered bytes AND fold the browser's
        # harvested cookies into the returned FetchSession, so a following curl
        # fetch on the same session is warm (the review's key requirement).
        result = BrowserResult(
            body=b"<html>rendered</html>",
            cookies={"SID": "xyz"},
            final_url="https://walled.example/x",
        )
        store = Mock()
        with (
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch(
                "wesearch.fetch.fetch.zendriver.fetch_zendriver",
                return_value=result,
            ) as via,
            patch("wesearch.profile.ProfileStore.shared", return_value=store),
        ):
            body, session = fetch.fetch(
                "https://walled.example/x",
                request=RequestParams(policy=PolicyParams(transport="zendriver")),
            )
        assert body == b"<html>rendered</html>"
        # Session warmed, and scoped to the origin that set the cookie.
        assert session.cookies_for("https://walled.example/x") == {"SID": "xyz"}
        assert via.call_count == 1
        store.save.assert_not_called()

    def test_browser_fetch_persists_cookies_to_profile_store(self) -> None:
        result = BrowserResult(
            body=b"ok",
            cookies={"cf_clearance": "tok"},
            final_url="https://walled.example/x",
        )
        store = Mock()
        store.load.return_value = None
        with (
            patch.object(fetch_mod, "egress_ip", return_value="5.5.5.5"),
            patch(
                "wesearch.fetch.fetch.zendriver.fetch_zendriver",
                return_value=result,
            ),
            patch("wesearch.profile.ProfileStore.shared", return_value=store),
        ):
            fetch.fetch(
                "https://walled.example/x",
                request=RequestParams(policy=PolicyParams(transport="zendriver")),
            )
        # A fresh (egress, domain) key is saved with the harvested cookies.
        store.save.assert_called_once()
        saved_profile = _recorded_profile(store.save)
        assert saved_profile.cookies == {"cf_clearance": "tok"}
        assert saved_profile.ua


class TestCurlThenZendriverBackend:
    """``transport="curl-then-zendriver"``: curl first, zendriver only on a bot block."""

    def test_curl_then_zendriver_inherits_zendriver_restrictions(self) -> None:
        # curl-then-zendriver may fall back to the browser, so it remains GET-only
        # and body-free.
        with pytest.raises(
            ValueError,
            match="curl-then-zendriver backend supports only GET",
        ):
            RequestParams(
                content=ContentParams(method="POST"),
                policy=PolicyParams(transport="curl-then-zendriver"),
            )
        with pytest.raises(
            ValueError,
            match="curl-then-zendriver backend cannot send a request",
        ):
            RequestParams(
                content=ContentParams(data={"a": "1"}),
                policy=PolicyParams(transport="curl-then-zendriver"),
            )
        with pytest.raises(
            ValueError,
            match="curl-then-zendriver transport cannot honor 'raw_headers'",
        ):
            RequestParams(
                content=ContentParams(raw_headers=True),
                policy=PolicyParams(transport="curl-then-zendriver"),
            )

    def test_curl_then_zendriver_returns_curl_body_without_touching_browser(
        self,
    ) -> None:
        # When curl succeeds, the browser backend is never invoked.
        with (
            patch.object(fetch_mod, "_send_as", return_value=b"curl body"),
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch("wesearch.fetch.fetch.zendriver.fetch_zendriver") as via,
        ):
            body, _ = fetch.fetch(
                "https://ok.example/",
                request=RequestParams(
                    policy=PolicyParams(transport="curl-then-zendriver"),
                ),
            )
        assert body == b"curl body"
        via.assert_not_called()

    def test_curl_then_zendriver_falls_back_to_zendriver_on_bot_block(self) -> None:
        # A curl BotDetectionError triggers the zendriver leg; its body is returned.
        result = BrowserResult(
            body=b"rendered",
            cookies={"cf_clearance": "t"},
            final_url="https://walled.example/",
        )
        with (
            patch.object(fetch_mod, "_send_as", side_effect=CloudflareChallengeError()),
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch.object(
                transport_routing,
                "remember_zendriver_domain",
            ) as remember,
            patch(
                "wesearch.fetch.fetch.zendriver.fetch_zendriver",
                return_value=result,
            ) as via,
        ):
            body, _ = fetch.fetch(
                "https://walled.example/",
                request=RequestParams(
                    policy=PolicyParams(transport="curl-then-zendriver"),
                ),
            )
        assert body == b"rendered"
        assert via.call_count == 1
        remember.assert_called_once_with("walled.example")

    def test_raw_headers_request_still_runs_the_body_validator(self) -> None:
        """A raw-header caller's ``body_validator`` must run.

        This branch bypasses ``_fetch_with_identity``, where every other return
        is validated. Unwrapped, the validator never ran for any raw-header
        caller, so a challenge page served with HTTP 200 reached the caller as
        an ordinary body -- a silent wrong answer rather than a typed error.
        """
        seen: list[bytes] = []

        def validate_body(body: bytes) -> None:
            seen.append(body)
            raise PuzzleChallengeError("challenge served")

        with (
            patch.object(fetch_mod, "_fetch_once", return_value=b"<form id=x>"),
            pytest.raises(PuzzleChallengeError, match="challenge served"),
        ):
            fetch.fetch(
                "https://raw.example/",
                request=RequestParams(
                    content=ContentParams(
                        headers={"User-Agent": "x"},
                        raw_headers=True,
                    ),
                    observe=ObserveParams(body_validator=validate_body),
                ),
            )

        assert seen == [b"<form id=x>"], "the raw-header body was never validated"

    def test_success_body_challenge_falls_back_and_remembers_domain(self) -> None:
        result = BrowserResult(body=b"rendered", cookies={}, final_url="")

        def validate_body(body: bytes) -> None:
            if b"enablejs" in body:
                raise GoogleJavascriptRequiredError("JavaScript required")

        with (
            patch.object(fetch_mod, "_send_as", return_value=b"enablejs"),
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch.object(
                transport_routing,
                "remember_zendriver_domain",
            ) as remember,
            patch(
                "wesearch.fetch.fetch.zendriver.fetch_zendriver",
                return_value=result,
            ) as via,
        ):
            body, _ = fetch.fetch(
                "https://walled.example/",
                request=RequestParams(
                    observe=ObserveParams(body_validator=validate_body),
                    policy=PolicyParams(transport="curl-then-zendriver"),
                ),
            )

        assert body == b"rendered"
        via.assert_called_once()
        remember.assert_called_once_with("walled.example")

    def test_bot_block_remembers_domain_even_when_browser_still_fails(self) -> None:
        with (
            patch.object(fetch_mod, "_send_as", side_effect=PuzzleChallengeError()),
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch.object(
                transport_routing,
                "remember_zendriver_domain",
            ) as remember,
            patch(
                "wesearch.fetch.fetch.zendriver.fetch_zendriver",
                side_effect=PuzzleChallengeError("human required"),
            ),
            pytest.raises(PuzzleChallengeError, match="human required"),
        ):
            fetch.fetch(
                "https://walled.example/",
                request=RequestParams(
                    policy=PolicyParams(transport="curl-then-zendriver"),
                ),
            )

        remember.assert_called_once_with("walled.example")

    def test_auto_reuses_persisted_zendriver_fallback(self) -> None:
        domains: set[str] = set()
        result = BrowserResult(body=b"rendered", cookies={}, final_url="")
        with (
            patch.object(
                fetch_mod,
                "_send_as",
                side_effect=CloudflareChallengeError(),
            ) as via_curl,
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch.object(
                transport_routing,
                "zendriver_domains",
                side_effect=lambda: frozenset(domains),
            ),
            patch.object(
                transport_routing,
                "remember_zendriver_domain",
                side_effect=domains.add,
            ),
            patch(
                "wesearch.fetch.fetch.zendriver.fetch_zendriver",
                return_value=result,
            ) as via_browser,
        ):
            first, _ = fetch.fetch("https://walled.example/")
            second, _ = fetch.fetch("https://walled.example/")

        assert first == second == b"rendered"
        assert via_curl.call_count == 1
        assert via_browser.call_count == 2

    def test_curl_then_zendriver_does_not_fall_back_on_non_block_error(self) -> None:
        # A plain 404 (not a bot block) propagates -- the browser would not help
        # and must not silently pay Chrome's launch cost.
        with (
            patch.object(
                fetch_mod,
                "_send_as",
                side_effect=FetchError("https://x/", 404, {}, b""),
            ),
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch("wesearch.fetch.fetch.zendriver.fetch_zendriver") as via,
            pytest.raises(FetchError),
        ):
            fetch.fetch(
                "https://x/",
                request=RequestParams(
                    policy=PolicyParams(transport="curl-then-zendriver"),
                ),
            )
        via.assert_not_called()


class _ZendriverModule(Protocol):
    fetch_zendriver: object


class _FetchModule(Protocol):
    """Patchable attributes imported from the fetch implementation module."""

    _Request: type[_Request]
    _fetch_once: object
    _send_as: object
    curl_session: object
    egress_ip: object
    fetch: object
    logger: logging.Logger
    time: types.ModuleType
    zendriver: _ZendriverModule


def _recorded_headers(mock: Mock, *, index: int = 0) -> dict[str, str]:
    """Return typed request headers recorded by a mock."""
    call = mock.call_args_list[index]
    assert isinstance(call.kwargs["headers"], dict)
    return cast(dict[str, str], call.kwargs["headers"])


def _recorded_url(mock: Mock) -> str:
    """Return the typed URL recorded by a mock."""
    call = mock.call_args
    assert call is not None
    assert isinstance(call.args[0], str)
    return call.args[0]


def _recorded_request(mock: Mock) -> RequestParams:
    """Return typed request parameters recorded by a mock."""
    call = mock.call_args
    assert call is not None
    assert isinstance(call.args[0], _Request)
    request = call.args[0]
    return request.params


def _recorded_profile(mock: Mock) -> Profile:
    """Return the typed profile recorded by a mock."""
    call = mock.call_args
    assert call is not None
    assert isinstance(call.args[2], Profile)
    return call.args[2]


def _shared(store: ProfileStore) -> Callable[[type[ProfileStore]], ProfileStore]:
    """Return a ``ProfileStore.shared`` replacement that always yields ``store``."""

    def shared(cls: type[ProfileStore]) -> ProfileStore:
        del cls
        return store

    return shared


def _identity_body(request: _Request, body: bytes) -> bytes:
    del request
    return body


def _hint_value(*, major: int, **_kwargs: object) -> dict[str, str]:
    return {f"hint-{major}": "v"}


def _captured_response(**kwargs: object) -> bytes:
    callback = kwargs["on_response"]
    assert callable(callback)
    callback(200, {"set-cookie": "new=3"}, "https://x.example:8443/path")
    return b"ok"


def _egress_value(*, cache: bool) -> str:
    return "old" if cache else "new"


class TestFetchCoreMutationCoverage:
    def test_fetch_once_builds_exact_request_and_forwards_all_options(self) -> None:
        params = RequestParams(
            content=ContentParams(
                method="POST",
                params={"q": "a b"},
                data={"x": "y"},
                headers={"Cookie": "from-header"},
            ),
            policy=PolicyParams(transport="curl", trust="internal"),
            retry=RetryParams(
                retries=0,
                timeout_sec=7.0,
                connect_timeout_sec=2.0,
                max_redirects=3,
            ),
        )
        response = Mock(return_value=b"body")
        on_response = Mock()
        StubSession()
        reseat = Mock()
        with patch.object(fetch_mod, "fetch_curl", response):
            got = _fetch_once(
                "https://u:p@example.com/path#frag",
                params,
                headers={"X-Test": "yes", "Cookie": "from-header"},
                cookies={"sid": "cookie"},
                raw_headers=False,
                impersonate="chrome133",
                accept_ch={},
                on_response=on_response,
                session=None,
                reseat=reseat,
            )
        assert got == b"body"
        response.assert_called_once_with(
            "https://example.com/path?q=a+b#frag",
            method="POST",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Origin": "https://example.com",
                "X-Test": "yes",
                "Authorization": "Basic dTpw",
                "Cookie": "from-header; sid=cookie",
            },
            body=b"x=y",
            timeout_sec=7.0,
            connect_timeout_sec=2.0,
            max_redirects=3,
            impersonate="chrome133",
            on_redirect=None,
            on_response=on_response,
            trust="internal",
            session=None,
            reseat=reseat,
        )

    def test_fetch_once_retries_status_zero_and_logs_exactly(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        params = RequestParams(
            policy=PolicyParams(transport="curl"),
            retry=RetryParams(retries=1),
        )
        err = FetchError("https://x/", 0, {}, b"down")
        backend = Mock(side_effect=[err, b"ok"])
        sleeps = Mock()
        monkeypatch.setattr(fetch_mod.time, "sleep", sleeps)
        caplog.set_level("DEBUG", logger=fetch_mod.logger.name)
        with (
            patch("wesearch.types.params.random.uniform", return_value=0.0),
            patch.object(fetch_mod, "fetch_curl", backend),
        ):
            assert (
                _fetch_once(
                    "https://x/",
                    params,
                    headers=None,
                    cookies=None,
                    raw_headers=False,
                    impersonate="chrome",
                    accept_ch={},
                    on_response=None,
                    session=None,
                )
                == b"ok"
            )
        assert backend.call_count == 2
        assert sleeps.call_args_list == [((1.0,), {})]
        assert [r.getMessage() for r in caplog.records] == [
            "fetch https://x/ → 0, retry in 1.0s",
        ]

    def test_fetch_once_json_body_and_case_insensitive_cookies(self) -> None:
        params = RequestParams(
            content=ContentParams(
                method="POST",
                json={"a": 1},
                headers={"cookie": "a=header", "COOKIE": "b=header"},
            ),
            policy=PolicyParams(transport="curl"),
        )
        backend = Mock(return_value=b"ok")
        with patch.object(fetch_mod, "fetch_curl", backend):
            _fetch_once(
                "https://x/",
                params,
                headers={"cookie": "a=header", "COOKIE": "b=header"},
                cookies={"c": "param"},
                raw_headers=False,
                impersonate="chrome",
                accept_ch={},
                on_response=None,
                session=None,
            )
        assert backend.call_args.kwargs["body"] == b'{"a": 1}'
        headers = _recorded_headers(backend)
        assert headers["Cookie"] == "a=header; b=header; c=param"
        assert set(headers) == {
            "Content-Type",
            "Cookie",
            "Origin",
        }


class TestResponseLearnerMutationCoverage:
    def test_observe_and_merge_preserve_exact_origin_state(self) -> None:
        caller = Mock()
        learner = _ResponseLearner(caller=caller)
        learner.observe(
            302,
            {"set-cookie": "a=1; Path=/\nb=2", "accept-ch": "Sec-CH-UA-Arch"},
            "https://a.example/redirect",
        )
        session = learner.merge_into(FetchSession())
        assert session.cookies == {"https://a.example": {"a": "1", "b": "2"}}
        assert session.accept_ch == {"https://a.example": frozenset({"sec-ch-ua-arch"})}
        caller.assert_called_once_with(
            302,
            {"set-cookie": "a=1; Path=/\nb=2", "accept-ch": "Sec-CH-UA-Arch"},
        )

    def test_validated_body_returns_body_after_callback(self) -> None:
        seen = Mock()
        request = _Request(
            url="https://x/",
            session=FetchSession(),
            params=RequestParams(observe=ObserveParams(body_validator=seen)),
        )
        assert _validated_body(request, b"exact") == b"exact"
        seen.assert_called_once_with(b"exact")


class TestIdentityMutationCoverage:
    def test_domainless_identity_sends_and_validates(self) -> None:
        request = _Request(
            url="/relative",
            session=FetchSession(),
            params=RequestParams(),
        )
        with (
            patch.object(fetch_mod, "_send_as", return_value=b"body") as send,
            patch.object(
                fetch_mod,
                "_validated_body",
                side_effect=_identity_body,
            ) as validate,
        ):
            assert (
                _fetch_with_identity(
                    request,
                    caller_headers={"X": "1"},
                    caller_cookies={"c": "2"},
                )
                == b"body"
            )
        send.assert_called_once_with(request, None, None, {"X": "1"}, {"c": "2"})
        validate.assert_called_once_with(request, b"body")

    def test_send_via_zendriver_forwards_exact_arguments_and_observes_landing(
        self,
        tmp_path: Path,
    ) -> None:
        request = _Request(
            url="https://a.example/path",
            session=FetchSession(impersonate="chrome133"),
            params=RequestParams(
                content=ContentParams(params={"q": "x y"}),
                retry=RetryParams(timeout_sec=11.0),
                observe=ObserveParams(on_redirect=Mock()),
            ),
        )
        result = BrowserResult(
            body=b"rendered",
            cookies={"sid": "1", "token": "2"},
            final_url="https://b.example/landed",
        )
        observer = Mock()
        request = request.__class__(
            url=request.url,
            session=request.session,
            params=request.params,
            observer=observer,
        )
        store = Mock()
        store.load.return_value = None
        with (
            patch.object(fetch_mod, "pinned_host", return_value=None) as pin,
            patch.object(
                fetch_mod,
                "egress_ip",
                side_effect=["198.51.100.8"],
            ) as egress,
            patch.object(
                fetch_mod.zendriver,
                "fetch_zendriver",
                return_value=result,
            ) as browser,
            patch.object(fetch_mod, "ProfileStore", shared=Mock(return_value=store)),
            patch.object(
                fetch_mod,
                "data_dir",
                return_value=tmp_path,
            ),
        ):
            assert (
                _send_via_zendriver(
                    request,
                    headers={"X": "h"},
                    cookies={"c": "v"},
                )
                == b"rendered"
            )
        pin.assert_called_once_with("https://a.example/path", "untrusted")
        egress.assert_called_once_with(cache=True)
        browser.assert_called_once()
        assert browser.call_args.args[0] == "https://a.example/path?q=x+y"
        assert browser.call_args.kwargs["egress"] == "198.51.100.8"
        assert browser.call_args.kwargs["timeout_sec"] == 11.0
        assert browser.call_args.kwargs["headers"] == {"X": "h"}
        assert browser.call_args.kwargs["cookies"] == {"c": "v"}
        assert observer.call_args.args == (
            200,
            {"set-cookie": "sid=1\ntoken=2"},
            "https://b.example/landed",
        )
        store.save.assert_called_once()


class TestHeaderMutationCoverage:
    def test_accept_ch_uses_version_one_hint_catalog(self) -> None:
        with patch.object(
            fetch_mod,
            "chrome_client_hints",
            side_effect=_hint_value,
        ) as hints:
            assert _accept_ch_hints({"accept-ch": "hint-1, hint-2"}) == {"hint-1"}
        hints.assert_called_once_with(major=1)

    def test_google_headers_forward_exact_identity(self) -> None:
        with (
            patch.object(
                fetch_mod,
                "impersonate_version_platform",
                return_value=(133, "linux"),
            ) as identity,
            patch.object(
                fetch_mod,
                "chrome_headers_for_google",
                return_value={"X": "google"},
            ) as google,
        ):
            assert _google_headers("https://www.google.com/search", "chrome133") == {
                "X": "google",
            }
        identity.assert_called_once_with("chrome133")
        google.assert_called_once_with(major=133, platform="linux")

    def test_build_headers_forwards_stdlib_identity_and_http_version(self) -> None:
        nav = Mock(return_value={"A": "b"})
        with (
            patch.object(fetch_mod, "chrome_navigation_headers", nav),
            patch.object(fetch_mod, "_google_headers", return_value={"G": "h"}),
            patch.object(
                fetch_mod,
                "impersonate_version_platform",
                return_value=(133, "linux"),
            ),
        ):
            assert _build_headers(
                method="GET",
                url="https://x.example/path",
                content_type=None,
                extra={"E": "f"},
                raw_headers=False,
                impersonate="chrome133",
                use_curl=False,
                accept_ch={},
            ) == {"A": "b", "G": "h", "E": "f"}
        nav.assert_called_once_with(
            major=133,
            platform="linux",
            method="GET",
            content_type="",
            origin="https://x.example",
            http2=False,
        )

    def test_curl_structural_headers_distinguish_get_head_and_post(self) -> None:
        with patch.object(fetch_mod, "_google_headers", return_value={}):
            assert (
                _curl_structural_headers(
                    method="GET",
                    url="https://x.example/p",
                    content_type="text/plain",
                    extra=None,
                    impersonate="chrome",
                    accept_ch={},
                )
                == {}
            )
            assert (
                _curl_structural_headers(
                    method="HEAD",
                    url="https://x.example/p",
                    content_type="text/plain",
                    extra=None,
                    impersonate="chrome",
                    accept_ch={},
                )
                == {}
            )
            assert _curl_structural_headers(
                method="POST",
                url="https://x.example/p",
                content_type="text/plain",
                extra=None,
                impersonate="chrome",
                accept_ch={},
            ) == {"Content-Type": "text/plain", "Origin": "https://x.example"}


class TestSendAsMutationCoverage:
    def test_stdlib_send_saves_captured_cookie_with_exact_arguments(self) -> None:
        request = _Request(
            url="https://x.example:8443/path",
            session=FetchSession(impersonate="chrome133"),
            params=RequestParams(policy=PolicyParams(transport="stdlib")),
        )
        store = Mock()
        request_send = Mock(side_effect=_captured_response)
        request = request.__class__(
            url=request.url,
            session=request.session,
            params=request.params,
            observer=Mock(),
        )
        profile = Profile(ua="stored", cookies={"old": "1"})
        with (
            patch.object(fetch_mod, "ProfileStore", shared=Mock(return_value=store)),
            patch.object(fetch_mod, "pinned_host", return_value=None),
            patch.object(fetch_mod, "egress_ip", return_value="198.51.100.7"),
            patch.object(fetch_mod, "draw_user_agent", return_value="drawn"),
            patch.object(fetch_mod._Request, "send", request_send),
        ):
            assert (
                _send_as(request, profile, "198.51.100.7", {"H": "v"}, {"new": "2"})
                == b"ok"
            )
        request_send.assert_called_once()
        assert request_send.call_args.kwargs["headers"] == {"H": "v"}
        assert request_send.call_args.kwargs["cookies"] == {"old": "1", "new": "2"}
        assert request_send.call_args.kwargs["raw_headers"] is False
        store.update_cookies.assert_called_once_with(
            "198.51.100.7",
            "x.example",
            {"new": "3"},
        )

    def test_curl_send_seeds_jar_and_supplies_reseat(self) -> None:
        request = _Request(
            url="https://x.example/path",
            session=FetchSession(impersonate="chrome133"),
            params=RequestParams(policy=PolicyParams(transport="curl")),
        )
        curl = Mock()
        store = Mock()
        with (
            patch.object(fetch_mod, "curl_session", return_value=curl) as make,
            patch.object(fetch_mod, "seed_session_jar") as seed,
            patch.object(fetch_mod, "set_session_cookies") as set_cookies,
            patch.object(fetch_mod, "ProfileStore", shared=Mock(return_value=store)),
            patch.object(fetch_mod, "pinned_host", return_value=None),
            patch.object(
                fetch_mod._Request,
                "send",
                return_value=b"ok",
            ) as send,
        ):
            assert (
                _send_as(
                    request,
                    Profile(ua="u", cookies={"old": "1"}),
                    "198.51.100.7",
                    None,
                    {"new": "2"},
                )
                == b"ok"
            )
        make.assert_called_once_with(
            "198.51.100.7",
            "x.example",
            "chrome133",
            pin=None,
            port=443,
        )
        seed.assert_called_once_with(curl, "x.example", {"old": "1"})
        set_cookies.assert_called_once_with(curl, "x.example", {"new": "2"})
        assert send.call_args.kwargs["curl"] is curl
        assert send.call_args.kwargs["cookies"] is None
        assert send.call_args.kwargs["raw_headers"] is False
        assert send.call_args.kwargs["reseat"] is not None


class TestFetchWithIdentityMutationCoverage:
    def test_missing_profile_uses_cached_then_live_egress(self) -> None:
        request = _Request(
            url="https://x.example/path",
            session=FetchSession(),
            params=RequestParams(),
        )
        store = Mock()
        store.load.return_value = None
        send = Mock(return_value=b"ok")
        with (
            patch.object(fetch_mod, "ProfileStore", shared=Mock(return_value=store)),
            patch.object(fetch_mod, "egress_ip", side_effect=["old", "new"]) as egress,
            patch.object(fetch_mod, "_send_as", send),
        ):
            assert (
                _fetch_with_identity(request, caller_headers=None, caller_cookies=None)
                == b"ok"
            )
        assert egress.call_args_list == [call(cache=True), call(cache=False)]
        send.assert_called_once_with(request, None, "new", None, None)

    def test_browser_identity_forwards_headers_and_cookies(self) -> None:
        request = _Request(
            url="https://x.example/path",
            session=FetchSession(),
            params=RequestParams(policy=PolicyParams(transport="zendriver")),
        )
        with (
            patch.object(
                fetch_mod,
                "_send_via_zendriver",
                return_value=b"ok",
            ) as browser,
            patch.object(
                fetch_mod,
                "_validated_body",
                side_effect=_identity_body,
            ),
        ):
            assert (
                _fetch_with_identity(
                    request,
                    caller_headers={"H": "v"},
                    caller_cookies={"c": "1"},
                )
                == b"ok"
            )
        browser.assert_called_once_with(request, headers={"H": "v"}, cookies={"c": "1"})


class TestFetchRetryAndTransportMutationCoverage:
    def test_fetch_once_retries_oserror_and_logs_exact_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        params = RequestParams(
            policy=PolicyParams(transport="curl"),
            retry=RetryParams(retries=1),
        )
        backend = Mock(side_effect=[OSError("offline"), b"ok"])
        sleeps = Mock()
        monkeypatch.setattr(fetch_mod.time, "sleep", sleeps)
        caplog.set_level("DEBUG", logger=fetch_mod.logger.name)
        with (
            patch("wesearch.types.params.random.uniform", return_value=0.0),
            patch.object(fetch_mod, "fetch_curl", backend),
        ):
            assert (
                _fetch_once(
                    "https://x/",
                    params,
                    headers=None,
                    cookies=None,
                    raw_headers=False,
                    impersonate="chrome",
                    accept_ch={},
                    on_response=None,
                    session=None,
                )
                == b"ok"
            )
        assert sleeps.call_args_list == [((1.0,), {})]
        assert [record.getMessage() for record in caplog.records] == [
            "fetch https://x/ failed: offline, retry in 1.0s",
        ]

    def test_fetch_once_does_not_retry_nonretryable_status(self) -> None:
        params = RequestParams(
            policy=PolicyParams(transport="curl"),
            retry=RetryParams(retries=1),
        )
        error = FetchError("https://x/", 404, {"h": "v"}, b"missing")
        backend = Mock(side_effect=error)
        with (
            patch.object(fetch_mod, "fetch_curl", backend),
            pytest.raises(FetchError) as raised,
        ):
            _fetch_once(
                "https://x/",
                params,
                headers=None,
                cookies=None,
                raw_headers=False,
                impersonate="chrome",
                accept_ch={},
                on_response=None,
                session=None,
            )
        assert raised.value is error
        backend.assert_called_once()

    def test_fetch_once_get_without_body_forwards_none(self) -> None:
        params = RequestParams(policy=PolicyParams(transport="curl"))
        backend = Mock(return_value=b"ok")
        with patch.object(fetch_mod, "fetch_curl", backend):
            _fetch_once(
                "https://x/",
                params,
                headers=None,
                cookies=None,
                raw_headers=False,
                impersonate="chrome",
                accept_ch={},
                on_response=None,
                session=None,
            )
        assert backend.call_args.kwargs["body"] is None


class TestRemainingHeaderAndBrowserMutationCoverage:
    def test_curl_headers_forward_hints_and_google_identity(self) -> None:
        with (
            patch.object(
                fetch_mod,
                "impersonate_version_platform",
                return_value=(133, "linux"),
            ) as identity,
            patch.object(
                fetch_mod,
                "chrome_client_hints",
                return_value={"Hint": "v"},
            ) as hints,
            patch.object(
                fetch_mod,
                "_google_headers",
                return_value={"Google": "v"},
            ) as google,
        ):
            assert _curl_structural_headers(
                method="GET",
                url="https://www.google.com/path",
                content_type=None,
                extra={"Extra": "v"},
                impersonate="chrome133",
                accept_ch={"https://www.google.com": frozenset({"Hint"})},
            ) == {"Hint": "v", "Google": "v", "Extra": "v"}
        identity.assert_called_once_with("chrome133")
        hints.assert_called_once_with(major=133, platform="linux")
        google.assert_called_once_with("https://www.google.com/path", "chrome133")

    def test_browser_call_forwards_trust_redirect_and_profile_path(
        self,
        tmp_path: Path,
    ) -> None:
        request = _Request(
            url="https://x.example/path",
            session=FetchSession(impersonate="chrome133"),
            params=RequestParams(
                policy=PolicyParams(transport="zendriver", trust="internal"),
                retry=RetryParams(timeout_sec=9.0),
                observe=ObserveParams(on_redirect=Mock()),
            ),
        )
        result = BrowserResult(body=b"ok", cookies={}, final_url="")
        with (
            patch.object(fetch_mod, "pinned_host", return_value=None),
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch.object(
                fetch_mod.zendriver,
                "fetch_zendriver",
                return_value=result,
            ) as browser,
            patch.object(
                fetch_mod,
                "data_dir",
                return_value=tmp_path,
            ),
        ):
            assert _send_via_zendriver(request, headers=None, cookies=None) == b"ok"
        assert browser.call_args.kwargs == {
            "profile_dir": tmp_path / "rekursiv-ai" / "wesearch" / "fetch-zendriver",
            "egress": "",
            "timeout_sec": 9.0,
            "headers": None,
            "cookies": None,
            "trust": "internal",
            "max_redirects": 10,
            "on_redirect": request.params.observe.on_redirect,
        }


class TestFinalMutationCoverage:
    def test_curl_reseat_callback_preserves_all_identity_arguments(self) -> None:
        request = _Request(
            url="https://x.example/path",
            session=FetchSession(impersonate="chrome133"),
            params=RequestParams(policy=PolicyParams(transport="curl")),
        )
        with (
            patch.object(fetch_mod, "curl_session", return_value=Mock()),
            patch.object(fetch_mod, "pinned_host", return_value=None),
            patch.object(fetch_mod, "_reseat", return_value=None) as reseat,
            patch.object(fetch_mod._Request, "send", return_value=b"ok") as send,
        ):
            _send_as(request, Profile(ua="u"), "198.51.100.7", None, None)
            callback = send.call_args.kwargs["reseat"]
            assert callable(callback)
            callback("https://other.example/path")
        reseat.assert_called_once_with(
            request,
            "198.51.100.7",
            "chrome133",
            "https://other.example/path",
        )

    def test_browser_fallback_and_existing_profile_update_use_exact_egress(
        self,
    ) -> None:
        request = _Request(
            url="https://x.example/path",
            session=FetchSession(impersonate="chrome133"),
            params=RequestParams(),
        )
        result = BrowserResult(
            body=b"ok",
            cookies={"sid": "new"},
            final_url="https://x.example/path",
        )
        store = Mock()
        store.load.return_value = Profile(ua="u", cookies={"old": "1"})
        with (
            patch.object(fetch_mod, "pinned_host", return_value=None),
            patch.object(fetch_mod, "egress_ip", side_effect=[None, "198.51.100.8"]),
            patch.object(fetch_mod.zendriver, "fetch_zendriver", return_value=result),
            patch.object(fetch_mod, "ProfileStore", shared=Mock(return_value=store)),
        ):
            assert _send_via_zendriver(request, headers=None, cookies=None) == b"ok"
        store.update_cookies.assert_called_once_with(
            "198.51.100.8",
            "x.example",
            {"sid": "new"},
        )

    def test_known_profile_burn_discards_exact_identity_before_retry(self) -> None:
        request = _Request(
            url="https://x.example/path",
            session=FetchSession(impersonate="chrome133"),
            params=RequestParams(),
        )
        profile = Profile(ua="u", cookies={"old": "1"})
        store = Mock()
        store.load.return_value = profile
        send = Mock(side_effect=[BotDetectionError("blocked"), b"ok"])
        with (
            patch.object(fetch_mod, "ProfileStore", shared=Mock(return_value=store)),
            patch.object(
                fetch_mod,
                "egress_ip",
                side_effect=["198.51.100.8", "198.51.100.9"],
            ),
            patch.object(fetch_mod, "_send_as", send),
            patch.object(fetch_mod, "close_curl_session") as close,
        ):
            assert (
                _fetch_with_identity(
                    request,
                    caller_headers={"H": "v"},
                    caller_cookies={"c": "1"},
                )
                == b"ok"
            )
        store.discard.assert_called_once_with("198.51.100.8", "x.example")
        close.assert_called_once_with("198.51.100.8", "x.example", "chrome133")
        assert send.call_args_list == [
            call(request, profile, "198.51.100.8", {"H": "v"}, {"c": "1"}),
            call(request, None, "198.51.100.9", {"H": "v"}, {"c": "1"}),
        ]


class TestLastSurvivorCoverage:
    def test_build_headers_forwards_each_curl_argument(self) -> None:
        structural = Mock(return_value={"ok": "yes"})
        with patch.object(fetch_mod, "_curl_structural_headers", structural):
            assert _build_headers(
                method="PATCH",
                url="https://x.example/path",
                content_type="application/json",
                extra={"X": "y"},
                raw_headers=False,
                impersonate="chrome133",
                use_curl=True,
                accept_ch={"https://x.example": frozenset({"Hint"})},
            ) == {"ok": "yes"}
        structural.assert_called_once_with(
            method="PATCH",
            url="https://x.example/path",
            content_type="application/json",
            extra={"X": "y"},
            impersonate="chrome133",
            accept_ch={"https://x.example": frozenset({"Hint"})},
        )

    def test_build_headers_raw_without_extras_is_empty(self) -> None:
        assert (
            _build_headers(
                method="GET",
                url="https://x.example/path",
                content_type=None,
                extra=None,
                raw_headers=True,
                impersonate="chrome133",
                use_curl=True,
                accept_ch={},
            )
            == {}
        )

    def test_fetch_once_uses_all_cookie_values_and_retry_budget(self) -> None:
        params = RequestParams(
            policy=PolicyParams(transport="curl"),
            retry=RetryParams(retries=1),
        )
        error = FetchError("https://x/", 500, {}, b"retry")
        backend = Mock(side_effect=[error, error, b"ok"])
        with patch.object(fetch_mod, "fetch_curl", backend), pytest.raises(FetchError):
            _fetch_once(
                "https://x/",
                params,
                headers=None,
                cookies={"a": "1", "b": "2"},
                raw_headers=False,
                impersonate="chrome",
                accept_ch={},
                on_response=None,
                session=None,
            )
        assert backend.call_count == 2

    def test_identity_burn_refreshes_live_egress_without_cache(self) -> None:
        request = _Request(
            url="https://x.example/path",
            session=FetchSession(),
            params=RequestParams(),
        )
        store = Mock()
        store.load.return_value = Profile(ua="u", cookies={})
        egress = Mock(side_effect=_egress_value)
        send = Mock(side_effect=[BotDetectionError("blocked"), b"ok"])
        with (
            patch.object(fetch_mod, "ProfileStore", shared=Mock(return_value=store)),
            patch.object(fetch_mod, "egress_ip", egress),
            patch.object(fetch_mod, "_send_as", send),
            patch.object(fetch_mod, "close_curl_session"),
        ):
            assert (
                _fetch_with_identity(request, caller_headers=None, caller_cookies=None)
                == b"ok"
            )
        assert egress.call_args_list == [call(cache=True), call(cache=False)]

    def test_google_hostname_fallback_is_empty(self) -> None:
        with patch.object(fetch_mod, "is_google_property", return_value=True) as google:
            _google_headers("https:///path", "chrome")
        google.assert_called_once_with("")

    def test_browser_fallback_calls_cache_false_and_not_true_twice(self) -> None:
        request = _Request(
            url="https://x.example/path",
            session=FetchSession(),
            params=RequestParams(),
        )
        result = BrowserResult(body=b"ok", cookies={}, final_url="")
        with (
            patch.object(fetch_mod, "pinned_host", return_value=None),
            patch.object(fetch_mod, "egress_ip", side_effect=[None, "ip"]) as egress,
            patch.object(fetch_mod.zendriver, "fetch_zendriver", return_value=result),
        ):
            _send_via_zendriver(request, headers=None, cookies=None)
        assert egress.call_args_list == [call(cache=True), call(cache=False)]


class TestExactForwardedMutationCoverage:
    def test_build_headers_forwards_url_and_impersonate_to_both_helpers(self) -> None:
        identity = Mock(return_value=(131, "linux"))
        google = Mock(return_value={})
        with (
            patch.object(fetch_mod, "impersonate_version_platform", identity),
            patch.object(fetch_mod, "_google_headers", google),
            patch.object(fetch_mod, "chrome_navigation_headers", return_value={}),
        ):
            assert (
                _build_headers(
                    method="GET",
                    url="https://x.example/path",
                    content_type=None,
                    extra=None,
                    raw_headers=False,
                    impersonate="chrome131",
                    use_curl=False,
                    accept_ch={},
                )
                == {}
            )
        identity.assert_called_once_with("chrome131")
        google.assert_called_once_with("https://x.example/path", "chrome131")

    def test_fetch_once_forwards_raw_headers_and_impersonate_exactly(self) -> None:
        params = RequestParams(policy=PolicyParams(transport="curl"))
        build = Mock(return_value={})
        backend = Mock(return_value=b"ok")
        with (
            patch.object(fetch_mod, "_build_headers", build),
            patch.object(fetch_mod, "fetch_curl", backend),
        ):
            assert (
                _fetch_once(
                    "https://x.example/",
                    params,
                    headers=None,
                    cookies=None,
                    raw_headers=True,
                    impersonate="chrome131",
                    accept_ch={},
                    on_response=None,
                    session=None,
                )
                == b"ok"
            )
        assert build.call_args.kwargs["raw_headers"] is True
        assert build.call_args.kwargs["impersonate"] == "chrome131"

    def test_fetch_once_joins_two_cookie_values_with_semicolon(self) -> None:
        params = RequestParams(policy=PolicyParams(transport="curl"))
        backend = Mock(return_value=b"ok")
        with patch.object(fetch_mod, "fetch_curl", backend):
            _fetch_once(
                "https://x.example/",
                params,
                headers=None,
                cookies={"a": "1", "b": "2"},
                raw_headers=False,
                impersonate="chrome",
                accept_ch={},
                on_response=None,
                session=None,
            )
        assert _recorded_headers(backend)["Cookie"] == "a=1; b=2"

    def test_fetch_once_does_not_add_attempt_when_retries_zero(self) -> None:
        params = RequestParams(
            policy=PolicyParams(transport="curl"),
            retry=RetryParams(retries=0),
        )
        error = FetchError("https://x.example/", 500, {}, b"error")
        backend = Mock(side_effect=error)
        with patch.object(fetch_mod, "fetch_curl", backend), pytest.raises(FetchError):
            _fetch_once(
                "https://x.example/",
                params,
                headers=None,
                cookies=None,
                raw_headers=False,
                impersonate="chrome",
                accept_ch={},
                on_response=None,
                session=None,
            )
        backend.assert_called_once()


class TestRemainingExactMutationCoverage:
    def test_fetch_once_forwards_existing_session_to_backend(self) -> None:
        params = RequestParams(policy=PolicyParams(transport="curl"))
        session: cc_requests.Session[cc_requests.Response] = cc_requests.Session()
        backend = Mock(return_value=b"ok")
        try:
            with patch.object(fetch_mod, "fetch_curl", backend):
                _fetch_once(
                    "https://x.example/",
                    params,
                    headers=None,
                    cookies=None,
                    raw_headers=False,
                    impersonate="chrome",
                    accept_ch={},
                    on_response=None,
                    session=session,
                )
            assert backend.call_args.kwargs["session"] is session
        finally:
            session.close()

    def test_fetch_once_get_uses_no_content_type(self) -> None:
        params = RequestParams(policy=PolicyParams(transport="stdlib"))
        build = Mock(return_value={})
        backend = Mock(return_value=b"ok")
        with (
            patch.object(fetch_mod, "_build_headers", build),
            patch.object(fetch_mod, "fetch_stdlib", backend),
        ):
            _fetch_once(
                "https://x.example/",
                params,
                headers=None,
                cookies=None,
                raw_headers=False,
                impersonate="chrome",
                accept_ch={},
                on_response=None,
                session=None,
            )
        assert build.call_args.kwargs["content_type"] is None

    def test_browser_fallback_forwards_caller_values_after_bot_block(self) -> None:
        request = _Request(
            url="https://x.example/",
            session=FetchSession(),
            params=RequestParams(policy=PolicyParams(transport="curl-then-zendriver")),
        )
        with (
            patch.object(
                fetch_mod,
                "_send_as",
                side_effect=BotDetectionError("blocked"),
            ),
            patch.object(fetch_mod, "_send_via_zendriver", return_value=b"ok") as send,
            patch.object(transport_routing, "remember_zendriver_domain"),
        ):
            assert (
                _fetch_with_identity(
                    request,
                    caller_headers={"X": "1"},
                    caller_cookies={"c": "2"},
                )
                == b"ok"
            )
        send.assert_called_once_with(
            request,
            headers={"X": "1"},
            cookies={"c": "2"},
        )

    def test_fetch_once_forwards_redirect_callback(self) -> None:
        redirect = Mock()
        params = RequestParams(
            policy=PolicyParams(transport="curl"),
            observe=ObserveParams(on_redirect=redirect),
        )
        backend = Mock(return_value=b"ok")
        with patch.object(fetch_mod, "fetch_curl", backend):
            _fetch_once(
                "https://x.example/",
                params,
                headers=None,
                cookies=None,
                raw_headers=False,
                impersonate="chrome",
                accept_ch={},
                on_response=None,
                session=None,
            )
        assert backend.call_args.kwargs["on_redirect"] is redirect

    def test_request_send_forwards_existing_session_identity(self) -> None:
        request = _Request(
            url="https://x.example/",
            session=FetchSession(),
            params=RequestParams(),
        )
        session: cc_requests.Session[cc_requests.Response] = cc_requests.Session()
        try:
            with patch.object(fetch_mod, "_fetch_once", return_value=b"ok") as send:
                request.send(
                    headers=None,
                    cookies=None,
                    raw_headers=False,
                    curl=session,
                )
            assert send.call_args.kwargs["session"] is session
        finally:
            session.close()

    def test_browser_fallback_forwards_both_caller_values(self) -> None:
        request = _Request(
            url="https://x.example/",
            session=FetchSession(),
            params=RequestParams(policy=PolicyParams(transport="zendriver")),
        )
        with patch.object(fetch_mod, "_send_via_zendriver", return_value=b"ok") as send:
            assert (
                _fetch_with_identity(
                    request,
                    caller_headers={"X": "1"},
                    caller_cookies={"c": "2"},
                )
                == b"ok"
            )
        send.assert_called_once_with(
            request,
            headers={"X": "1"},
            cookies={"c": "2"},
        )

    def test_stdlib_send_has_no_reseat_callback(self) -> None:
        request = _Request(
            url="https://x.example/",
            session=FetchSession(),
            params=RequestParams(policy=PolicyParams(transport="stdlib")),
        )
        with patch.object(fetch_mod._Request, "send", return_value=b"ok") as send:
            _send_as(request, None, "198.51.100.1", None, None)
        assert send.call_args.kwargs["reseat"] is None

    def test_curl_session_pin_and_port_are_forwarded_exactly(self) -> None:
        request = _Request(
            url="https://x.example:8443/",
            session=FetchSession(),
            params=RequestParams(
                policy=PolicyParams(transport="curl", trust="internal"),
            ),
        )
        curl = Mock()
        with (
            patch.object(fetch_mod, "curl_session", return_value=curl) as make,
            patch.object(fetch_mod, "pinned_host", return_value="203.0.113.7") as pin,
            patch.object(fetch_mod._Request, "send", return_value=b"ok"),
        ):
            _send_as(request, None, "198.51.100.1", None, None)
        pin.assert_called_once_with("https://x.example:8443/", "internal")
        make.assert_called_once_with(
            "198.51.100.1",
            "x.example",
            "chrome",
            pin="203.0.113.7",
            port=8443,
        )

    def test_new_profile_saves_drawn_user_agent(self) -> None:
        request = _Request(
            url="https://x.example/",
            session=FetchSession(impersonate="chrome133"),
            params=RequestParams(policy=PolicyParams(transport="stdlib")),
        )
        store = Mock()
        with (
            patch.object(fetch_mod, "ProfileStore", shared=Mock(return_value=store)),
            patch.object(fetch_mod, "draw_user_agent", return_value="drawn"),
            patch.object(
                fetch_mod,
                "kind_for_impersonate",
                return_value="kind",
            ) as kind,
            patch.object(fetch_mod._Request, "send", return_value=b"ok"),
        ):
            _send_as(request, None, "198.51.100.1", None, None)
        kind.assert_called_once_with("chrome133")
        assert _recorded_profile(store.save).ua == "drawn"

    def test_browser_empty_landing_domain_does_not_persist(self) -> None:
        request = _Request(
            url="https://x.example/",
            session=FetchSession(),
            params=RequestParams(),
            observer=Mock(),
        )
        result = BrowserResult(
            body=b"ok",
            cookies={"sid": "1"},
            final_url="https:///path",
        )
        store = Mock()
        with (
            patch.object(fetch_mod, "pinned_host", return_value=None),
            patch.object(fetch_mod, "egress_ip", return_value="198.51.100.1"),
            patch.object(fetch_mod.zendriver, "fetch_zendriver", return_value=result),
            patch.object(fetch_mod, "ProfileStore", shared=Mock(return_value=store)),
        ):
            assert _send_via_zendriver(request, headers=None, cookies=None) == b"ok"
        store.save.assert_not_called()
        store.update_cookies.assert_not_called()

    def test_browser_cookie_free_response_notifies_with_empty_headers(self) -> None:
        observer = Mock()
        request = _Request(
            url="https://x.example/",
            session=FetchSession(),
            params=RequestParams(),
            observer=observer,
        )
        result = BrowserResult(body=b"ok", cookies={}, final_url="https://x.example/")
        with (
            patch.object(fetch_mod, "pinned_host", return_value=None),
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch.object(fetch_mod.zendriver, "fetch_zendriver", return_value=result),
        ):
            _send_via_zendriver(request, headers=None, cookies=None)
        observer.assert_called_once_with(200, {}, "https://x.example/")

    def test_browser_profile_uses_request_impersonate_for_user_agent(self) -> None:
        request = _Request(
            url="https://x.example/",
            session=FetchSession(impersonate="chrome133"),
            params=RequestParams(),
        )
        result = BrowserResult(
            body=b"ok",
            cookies={"sid": "1"},
            final_url="https://x.example/",
        )
        store = Mock()
        store.load.return_value = None
        with (
            patch.object(fetch_mod, "pinned_host", return_value=None),
            patch.object(fetch_mod, "egress_ip", return_value="198.51.100.1"),
            patch.object(fetch_mod.zendriver, "fetch_zendriver", return_value=result),
            patch.object(fetch_mod, "ProfileStore", shared=Mock(return_value=store)),
            patch.object(fetch_mod, "draw_user_agent", return_value="drawn"),
            patch.object(
                fetch_mod,
                "kind_for_impersonate",
                return_value="kind",
            ) as kind,
        ):
            _send_via_zendriver(request, headers=None, cookies=None)
        kind.assert_called_once_with("chrome133")
        assert _recorded_profile(store.save).ua == "drawn"


class TestRetryAttemptCounterMutationCoverage:
    def test_retry_counter_advances_across_three_fetch_error_attempts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        params = RequestParams(
            policy=PolicyParams(transport="curl"),
            retry=RetryParams(retries=3),
        )
        error = FetchError("https://x.example/", 500, {}, b"retry")
        backend = Mock(side_effect=[error, error, error, b"ok"])
        sleeps = Mock()
        monkeypatch.setattr(fetch_mod.time, "sleep", sleeps)
        with (
            patch("wesearch.types.params.random.uniform", return_value=0.0),
            patch.object(fetch_mod, "fetch_curl", backend),
        ):
            assert (
                _fetch_once(
                    "https://x.example/",
                    params,
                    headers=None,
                    cookies=None,
                    raw_headers=False,
                    impersonate="chrome",
                    accept_ch={},
                    on_response=None,
                    session=None,
                )
                == b"ok"
            )
        assert backend.call_count == 4
        assert sleeps.call_args_list == [((1.0,), {}), ((2.0,), {}), ((4.0,), {})]

    def test_retry_counter_advances_across_three_oserror_attempts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        params = RequestParams(
            policy=PolicyParams(transport="curl"),
            retry=RetryParams(retries=3),
        )
        backend = Mock(
            side_effect=[
                OSError("offline"),
                OSError("offline"),
                OSError("offline"),
                b"ok",
            ],
        )
        sleeps = Mock()
        monkeypatch.setattr(fetch_mod.time, "sleep", sleeps)
        with (
            patch("wesearch.types.params.random.uniform", return_value=0.0),
            patch.object(fetch_mod, "fetch_curl", backend),
        ):
            assert (
                _fetch_once(
                    "https://x.example/",
                    params,
                    headers=None,
                    cookies=None,
                    raw_headers=False,
                    impersonate="chrome",
                    accept_ch={},
                    on_response=None,
                    session=None,
                )
                == b"ok"
            )
        assert backend.call_count == 4
        assert sleeps.call_args_list == [((1.0,), {}), ((2.0,), {}), ((4.0,), {})]


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
