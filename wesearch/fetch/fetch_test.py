"""Tests for wesearch.fetch.

Transports are mocked at their seam: curl_cffi's ``Session.request``, the
stdlib ``_connect``, or the transport function itself. Every fetch runs on a
session that already knows its egress, so none probes the network.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from http import client
from typing import TYPE_CHECKING
from unittest.mock import ANY, Mock, call, patch

import base64
import importlib
import inspect
import math

from curl_cffi import requests as cc_requests
from curl_cffi.requests import Headers
from treekle import from_plain

import pytest

from wesearch import fetch
from wesearch.chrome.headers import chrome_identity_headers, chrome_platform
from wesearch.fetch import (
    ContentParams,
    FetchSession,
    ObserveParams,
    PolicyParams,
    RequestParams,
    RetryParams,
)
from wesearch.fetch.fetch import (
    _send,
    _without_userinfo,
    egress_ip,
    resolve_transport,
)
from wesearch.fetch.session import default_session
from wesearch.fetch.transport import transport_routing, zendriver
from wesearch.fetch.types import Closable, CurlKey, Fetch
from wesearch.lib import zstd_compat
from wesearch.profile import Profile
from wesearch.types.errors import (
    BotDetectionError,
    CloudflareChallengeError,
    CooldownError,
    FetchError,
    GoogleJavascriptRequiredError,
    GoogleSorryError,
    PuzzleChallengeError,
)
from wesearch.types.params import ignore


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator, Mapping

    from wesearch.types.params import HttpMethod, Transport


# A public egress, so ``fetch`` never probes for one.
_EGRESS = "203.0.113.1"


@pytest.fixture(name="session")
def session_fixture() -> Iterator[FetchSession]:
    """Return a session that already knows its egress; close it after."""
    session = FetchSession(egress=_EGRESS)
    yield session
    session.close()


def test_fetch_uses_transport_package_layout() -> None:
    assert fetch.__file__ is not None
    assert fetch.__file__.endswith("/fetch/__init__.py")
    assert callable(fetch.fetch)
    for module in ("common", "fetch", "types"):
        importlib.import_module(f"wesearch.fetch.{module}")
    for module in ("curl", "stdlib", "zendriver", "transport_routing"):
        importlib.import_module(f"wesearch.fetch.transport.{module}")


class TestWithoutUserinfo:
    def test_no_userinfo(self) -> None:
        assert _without_userinfo("https://example.com/p?q=1") == (
            "https://example.com/p?q=1",
            "",
        )

    def test_user_pass_stripped_and_encoded(self) -> None:
        url, auth = _without_userinfo("https://u:p@example.com:8443/x")
        assert url == "https://example.com:8443/x"
        assert auth == "Basic " + base64.b64encode(b"u:p").decode()

    def test_pct_decoded_credentials(self) -> None:
        url, auth = _without_userinfo("https://u%40x:p%3Aw@example.com/")
        assert url == "https://example.com/"
        assert auth == "Basic " + base64.b64encode(b"u@x:p:w").decode()

    def test_user_only(self) -> None:
        url, auth = _without_userinfo("https://u@example.com/")
        assert url == "https://example.com/"
        assert auth == "Basic " + base64.b64encode(b"u:").decode()

    def test_password_only_and_at_sign_in_password(self) -> None:
        assert _without_userinfo("https://:pass@example.com/path")[1] == (
            "Basic OnBhc3M="
        )
        assert _without_userinfo("https://u:p@ss@example.com/path")[0] == (
            "https://example.com/path"
        )


class TestUrlParams:
    def test_params_precede_a_fragment(self, session: FetchSession) -> None:
        # A fragment ends the URL and is never sent to the server, so appending
        # the query after it silently dropped every parameter from the wire.
        backend = Mock(return_value=b"ok")
        with patch("wesearch.fetch.fetch.fetch_curl", backend):
            _send(
                "https://e.example/p#section",
                _curl(content=ContentParams(params={"q": "x"})),
                session=session,
                profile_cookies=False,
            )
        assert backend.call_args.args[0] == "https://e.example/p?q=x#section"

    def test_params_merge_into_an_existing_query(self, session: FetchSession) -> None:
        backend = Mock(return_value=b"ok")
        with patch("wesearch.fetch.fetch.fetch_curl", backend):
            _send(
                "https://e.example/p?a=1#s",
                _curl(content=ContentParams(params={"q": "x"})),
                session=session,
                profile_cookies=False,
            )
        assert backend.call_args.args[0] == "https://e.example/p?a=1&q=x#s"


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
        # Retry-After may be an HTTP-date, not just delta-seconds. A near-future
        # date must produce a positive delay, not fall through to backoff.
        future = datetime.now(tz=UTC) + timedelta(seconds=10)
        delay = RetryParams().backoff_delay(0, {"retry-after": format_datetime(future)})
        assert 5 <= delay <= 30

    def test_retry_after_past_date_is_zero(self) -> None:
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
        # negative and never wakes on a NaN.
        delay = RetryParams().backoff_delay(0, {"retry-after": value})
        assert math.isfinite(delay)
        assert delay >= 0.0


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
        # Status 0 is the "no HTTP response" sentinel; there is no HTTP 0.
        err = FetchError("https://x.com", 0, {}, b"Failed to connect to x.com port 443")
        msg = str(err)
        assert "HTTP 0" not in msg
        assert msg.startswith("connection failed: https://x.com")
        assert "connect" in msg.lower()


class TestFetchInputValidation:
    """Invalid numeric args are rejected where they are written."""

    def test_negative_retries_rejected(self) -> None:
        with pytest.raises(ValueError, match="retries"):
            RetryParams(retries=-1)

    def test_negative_max_redirects_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_redirects"):
            RetryParams(max_redirects=-1)

    def test_nonpositive_timeout_rejected(self) -> None:
        # `curl` reads 0 as "no timeout", stdlib as "non-blocking".
        with pytest.raises(ValueError, match="timeout_sec"):
            RetryParams(timeout_sec=0)


class TestFetchClassifiesBlockAtBoundary:
    """``fetch`` raises the specific :class:`BotDetectionError` for a block."""

    def test_cloudflare_403_raises_cloudflare_challenge_error(
        self,
        session: FetchSession,
    ) -> None:
        response = _response(
            403,
            b"<!DOCTYPE html><html><head><title>Just a moment...</title>"
            b'<div class="challenge-platform"></div></head></html>',
            [("server", "cloudflare"), ("cf-ray", "a1-LAX")],
        )
        with _serve(response), pytest.raises(CloudflareChallengeError) as exc:
            fetch.fetch("https://x.com", request=_curl(), session=session)
        assert isinstance(exc.value, FetchError)
        assert isinstance(exc.value, BotDetectionError)
        assert exc.value.status == 403
        assert exc.value.headers == {"server": "cloudflare", "cf-ray": "a1-LAX"}
        assert b"challenge-platform" in exc.value.body
        assert "cloudflare" in exc.value.guidance.lower()

    def test_recaptcha_403_raises_puzzle_challenge_error(
        self,
        session: FetchSession,
    ) -> None:
        response = _response(
            403,
            b'<div class="g-recaptcha" data-sitekey="x"></div>',
            [("content-type", "text/html")],
        )
        with _serve(response), pytest.raises(PuzzleChallengeError) as exc:
            fetch.fetch("https://x.com", request=_curl(), session=session)
        assert exc.value.status == 403
        assert "captcha" in exc.value.guidance.lower()

    def test_genuine_404_raises_plain_fetch_error_not_bot_flag(
        self,
        session: FetchSession,
    ) -> None:
        response = _response(
            404,
            b"<html><body><h1>404 Not Found</h1></body></html>",
            [("server", "nginx")],
        )
        with _serve(response), pytest.raises(FetchError) as exc:
            fetch.fetch("https://x.com", request=_curl(), session=session)
        assert not isinstance(exc.value, BotDetectionError)
        assert exc.value.status == 404

    def test_except_fetch_error_catches_the_specific_subclass(
        self,
        session: FetchSession,
    ) -> None:
        response = _response(
            403,
            b"<html><head><title>Just a moment...</title></head></html>",
            [("server", "cloudflare"), ("cf-ray", "b2-LAX")],
        )
        caught: FetchError | None = None
        with _serve(response):
            try:
                fetch.fetch("https://x.com", request=_curl(), session=session)
            except FetchError as e:
                caught = e
        assert isinstance(caught, CloudflareChallengeError)


class TestFetchRetry:
    """Retries on the stdlib transport, mocked at its connection."""

    def test_retries_on_500(self, session: FetchSession) -> None:
        connection = _stdlib_connection(
            _http_response(500, b"ISE"),
            _http_response(body=b"ok"),
        )
        with (
            patch(_CONNECT, return_value=connection),
            patch("wesearch.fetch.fetch.time.sleep"),
        ):
            assert (
                fetch.fetch(
                    "https://example.com",
                    request=_stdlib(retry=RetryParams(retries=1)),
                    session=session,
                )
                == b"ok"
            )

    def test_no_retry_on_404(self, session: FetchSession) -> None:
        connection = _stdlib_connection(_http_response(404, b"NF"))
        with (
            patch(_CONNECT, return_value=connection),
            pytest.raises(FetchError, match="404"),
        ):
            fetch.fetch(
                "https://example.com",
                request=_stdlib(retry=RetryParams(retries=3)),
                session=session,
            )
        assert connection.getresponse.call_count == 1

    def test_error_body_is_decompressed(self, session: FetchSession) -> None:
        # A Cloudflare 403 challenge page is compressed like any other body; a raw
        # FetchError.body cannot be told from a plain 404.
        html = b"<!DOCTYPE html><html>Just a moment...</html>"
        connection = _stdlib_connection(
            _http_response(
                403,
                zstd_compat.compress(html),
                [("content-encoding", "zstd"), ("server", "cloudflare")],
            ),
        )
        with patch(_CONNECT, return_value=connection), pytest.raises(FetchError) as exc:
            fetch.fetch("https://x.com", request=_stdlib(), session=session)
        assert exc.value.body == html

    def test_retries_on_network_error(self, session: FetchSession) -> None:
        connection = _stdlib_connection(OSError("refused"), _http_response(body=b"ok"))
        with (
            patch(_CONNECT, return_value=connection),
            patch("wesearch.fetch.fetch.time.sleep"),
        ):
            assert (
                fetch.fetch(
                    "https://example.com",
                    request=_stdlib(retry=RetryParams(retries=1)),
                    session=session,
                )
                == b"ok"
            )


class TestHeaderOrder:
    """Lock the canonical Chrome header order on the stdlib wire.

    http.client emits headers in dict insertion order, so the dict's key order
    is the wire order.
    """

    def _capture_headers(
        self,
        session: FetchSession,
        content: ContentParams | None = None,
    ) -> dict[str, str]:
        connection = _stdlib_connection(_http_response(body=b"ok"))
        with patch(_CONNECT, return_value=connection):
            fetch.fetch(
                "https://example.com/",
                request=_stdlib(content=content),
                session=session,
            )
        return from_plain(
            connection.request.call_args.kwargs["headers"],
            dict[str, str],
        )

    def test_get_navigation_order(self, session: FetchSession) -> None:
        # The order a real Chrome navigation sends over HTTP/1.1: no Connection,
        # no Priority (an HTTP/2 construct). Host leads: the pinned connection
        # states it itself, and servers that check order 403 a trailing one.
        headers = self._capture_headers(session)
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

    def test_user_agent_agrees_with_the_client_hints_beside_it(
        self,
        session: FetchSession,
    ) -> None:
        # A browser that contradicts itself is provably not a browser.
        headers = self._capture_headers(session)
        user_agent = headers["User-Agent"]
        platform = headers["sec-ch-ua-platform"].strip('"')
        token = {"Windows": "Windows NT", "Linux": "X11; Linux", "macOS": "Macintosh"}
        assert token[platform] in user_agent
        major = user_agent.split("Chrome/", 1)[1].split(".", 1)[0]
        assert f'v="{major}"' in headers["sec-ch-ua"]

    def test_post_xhr_order_with_json(self, session: FetchSession) -> None:
        headers = self._capture_headers(
            session,
            ContentParams(method="POST", json={"q": "x"}),
        )
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

    def test_post_xhr_order_with_form(self, session: FetchSession) -> None:
        headers = self._capture_headers(
            session,
            ContentParams(method="POST", data={"q": "x"}),
        )
        assert headers["Content-Type"] == "application/x-www-form-urlencoded"
        keys = list(headers)
        assert keys.index("Content-Type") == keys.index("Accept") + 1
        assert keys.index("Origin") == keys.index("Content-Type") + 1

    def test_post_without_body_omits_content_type(self, session: FetchSession) -> None:
        headers = self._capture_headers(session, ContentParams(method="POST"))
        assert "Content-Type" not in headers

    def test_caller_override_preserves_slot(self, session: FetchSession) -> None:
        headers = self._capture_headers(
            session,
            ContentParams(headers={"Accept-Language": "de-DE"}),
        )
        keys = list(headers)
        assert headers["Accept-Language"] == "de-DE"
        assert keys.index("Accept-Encoding") + 1 == keys.index("Accept-Language")

    def test_caller_user_agent_replaces_the_default_in_its_slot(
        self,
        session: FetchSession,
    ) -> None:
        bare = self._capture_headers(session, ContentParams())
        overridden = self._capture_headers(
            session,
            ContentParams(headers={"User-Agent": "Custom/1.0"}),
        )
        assert overridden == {**bare, "User-Agent": "Custom/1.0"}
        assert list(overridden) == list(bare)

    def test_caller_none_deletes_a_default(self, session: FetchSession) -> None:
        headers = self._capture_headers(
            session,
            ContentParams(headers={"User-Agent": None}),
        )
        assert "User-Agent" not in headers

    def test_caller_new_header_appended(self, session: FetchSession) -> None:
        headers = self._capture_headers(
            session,
            ContentParams(headers={"X-Trace": "abc"}),
        )
        assert list(headers)[-1] == "X-Trace"

    def test_pinned_host_puts_host_first(self, session: FetchSession) -> None:
        captured = self._capture_headers(session)
        assert next(iter(captured)) == "Host"
        assert captured["Host"] == "example.com"


class TestOnResponse:
    """``on_response(status, headers)`` fires once per received response."""

    def test_curl_success_reports_status_and_headers(
        self,
        session: FetchSession,
    ) -> None:
        seen: list[tuple[int, dict[str, str]]] = []
        with _serve(_response(headers=[("set-cookie", "GSP=xyz")])):
            fetch.fetch(
                "https://x.com",
                request=RequestParams(
                    observe=ObserveParams(on_response=lambda s, h: seen.append((s, h))),
                ),
                session=session,
            )
        assert seen == [(200, {"set-cookie": "GSP=xyz"})]

    def test_a_caller_observer_gets_no_hop_url(self, session: FetchSession) -> None:
        seen = Mock()
        with patch(
            "wesearch.fetch.fetch.fetch_curl",
            side_effect=_reports(200, {"h": "v"}, "https://x.example/"),
        ):
            _send(
                "https://x.example/",
                _curl(observe=ObserveParams(on_response=seen)),
                session=session,
                profile_cookies=False,
            )
        seen.assert_called_once_with(200, {"h": "v"})


class TestRedirectIdentityScoping:
    """Cross-origin redirects re-scope every origin-bound identity element."""

    def test_302_post_downgrades_to_bodyless_get(self, session: FetchSession) -> None:
        # A 301/302 POST becomes a bodyless GET; only 307/308 keep the method.
        with _serve(
            _response(302, b"", [("location", "https://a.com/land")]),
            _response(content=b"done"),
        ) as request:
            fetch.fetch(
                "https://a.com/start",
                request=RequestParams(
                    content=ContentParams(method="POST", data={"x": "1"}),
                ),
                session=session,
            )
        second = request.call_args_list[1]
        assert second.args[0] == "GET"
        assert second.kwargs["data"] is None

    def test_cross_origin_redirect_drops_cookie_header(
        self,
        session: FetchSession,
    ) -> None:
        with _serve(
            _response(302, b"", [("location", "https://b.com/next")]),
            _response(content=b"done"),
        ) as request:
            fetch.fetch(
                "https://a.com/start",
                request=RequestParams(content=ContentParams(cookies={"SID": "secret"})),
                session=session,
            )
        assert "cookie" in _lower(request, 0)
        assert request.call_args_list[1].args[1] == "https://b.com/next"
        assert "cookie" not in _lower(request, 1)

    def test_same_origin_redirect_keeps_cookie_header(
        self,
        session: FetchSession,
    ) -> None:
        with _serve(
            _response(302, b"", [("location", "https://a.com/next")]),
            _response(content=b"done"),
        ) as request:
            fetch.fetch(
                "https://a.com/start",
                request=RequestParams(content=ContentParams(cookies={"SID": "secret"})),
                session=session,
            )
        assert _lower(request, 1).get("cookie") == "SID=secret"

    def test_one_cookie_header_carries_every_jar_across_hops(
        self,
        session: FetchSession,
    ) -> None:
        # The stored profile, the caller's cookies, and a cookie the first hop
        # sets ride ONE Cookie header; curl_cffi's own jar is never used.
        session.profiles.save(_EGRESS, "a.com", Profile(cookies={"S": "1"}))
        with _serve(
            _response(302, b"", [("location", "/next"), ("set-cookie", "n=3")]),
            _response(content=b"done"),
        ) as request:
            fetch.fetch(
                "https://a.com/start",
                request=RequestParams(content=ContentParams(cookies={"c": "2"})),
                session=session,
            )
        for index, expected in ((0, "S=1; c=2"), (1, "S=1; c=2; n=3")):
            sent = _sent_headers(request, index)
            assert [v for k, v in sent.items() if k.lower() == "cookie"] == [expected]
            assert request.call_args_list[index].kwargs["discard_cookies"] is True
        stored = session.profiles.load(_EGRESS, "a.com")
        assert stored is not None
        assert stored.cookies == {"S": "1", "n": "3"}

    def test_a_cross_host_hop_sends_the_targets_stored_cookies(
        self,
        session: FetchSession,
    ) -> None:
        session.profiles.save(_EGRESS, "a.com", Profile(cookies={"A": "1"}))
        session.profiles.save(_EGRESS, "b.com", Profile(cookies={"B": "2"}))
        with _serve(
            _response(302, b"", [("location", "https://b.com/next")]),
            _response(content=b"done"),
        ) as request:
            fetch.fetch("https://a.com/start", session=session)
        assert _lower(request, 1)["cookie"] == "B=2"

    def test_raw_headers_send_no_stored_cookies_on_a_cross_host_hop(
        self,
        session: FetchSession,
    ) -> None:
        session.profiles.save(_EGRESS, "b.com", Profile(cookies={"B": "2"}))
        with _serve(
            _response(302, b"", [("location", "https://b.com/next")]),
            _response(content=b"done"),
        ) as request:
            fetch.fetch(
                "https://a.com/start",
                request=RequestParams(content=ContentParams(raw_headers=True)),
                session=session,
            )
        assert "cookie" not in _lower(request, 1)

    def test_cross_origin_target_cookie_not_persisted_to_source_profile(
        self,
        session: FetchSession,
    ) -> None:
        with _serve(
            _response(302, b"", [("location", "https://b.com/next")]),
            _response(content=b"done", headers=[("set-cookie", "FOREIGN=1; Path=/")]),
        ):
            fetch.fetch("https://a.com/start", session=session)
        assert session.profiles.load(_EGRESS, "a.com") is None
        landed = session.profiles.load(_EGRESS, "b.com")
        assert landed is not None
        assert landed.cookies == {"FOREIGN": "1"}

    def test_browser_target_cookie_is_not_attributed_to_the_source(
        self,
        session: FetchSession,
    ) -> None:
        # Chrome follows its own redirects; its cookies belong to where it landed.
        with patch.object(
            zendriver,
            "fetch_zendriver",
            _landed(
                b"<html>ok</html>",
                "B_SESSION=secret",
                "https://b.example/landing",
            ),
        ):
            fetch.fetch(
                "https://a.example/start",
                request=_zendriver(),
                session=session,
            )
        landed = session.profiles.load(_EGRESS, "b.example")
        assert landed is not None
        assert landed.cookies == {"B_SESSION": "secret"}
        assert session.profiles.load(_EGRESS, "a.example") is None


class TestIdentityLayer:
    """Each fetch sends this egress's stored cookies for the host and saves new ones.

    On a bot block of a KNOWN identity it discards it and retries once fresh.
    """

    def test_stored_cookies_ride_the_cookie_header(self, session: FetchSession) -> None:
        session.profiles.save(_EGRESS, "x.com", Profile(cookies={"GSP": "s"}))
        with _serve(_response()) as request:
            fetch.fetch("https://x.com/p", request=_curl(), session=session)
        sent = _sent_headers(request)
        assert sent["User-Agent"] == _curl_user_agent()
        assert sent["Cookie"] == "GSP=s"

    def test_caller_cookie_and_user_agent_override_the_defaults(
        self,
        session: FetchSession,
    ) -> None:
        session.profiles.save(_EGRESS, "x.com", Profile(cookies={"GSP": "s"}))
        with _serve(_response()) as request:
            fetch.fetch(
                "https://x.com/p",
                request=RequestParams(
                    content=ContentParams(
                        headers={"User-Agent": "Mine/1"},
                        cookies={"GSP": "caller"},
                    ),
                ),
                session=session,
            )
        sent = _sent_headers(request)
        assert sent["User-Agent"] == "Mine/1"
        assert sent["Cookie"] == "GSP=caller"

    def test_no_profile_still_presents_the_curl_chrome(
        self,
        session: FetchSession,
    ) -> None:
        with _serve(_response()) as request:
            fetch.fetch("https://x.com/p", request=_curl(), session=session)
        sent = _sent_headers(request)
        assert sent["User-Agent"] == _curl_user_agent()
        assert "Cookie" not in sent

    def test_set_cookie_is_persisted(self, session: FetchSession) -> None:
        with _serve(_response(headers=[("set-cookie", "GSP=minted; Path=/")])):
            fetch.fetch("https://x.com/p", request=_curl(), session=session)
        got = session.profiles.load(_EGRESS, "x.com")
        assert got is not None
        assert got.cookies == {"GSP": "minted"}

    def test_a_response_without_set_cookie_saves_no_profile(
        self,
        session: FetchSession,
    ) -> None:
        with patch("wesearch.fetch.fetch.fetch_stdlib", return_value=b"ok"):
            fetch.fetch("https://x.example/", request=_stdlib(), session=session)
        assert session.profiles.load(_EGRESS, "x.example") is None

    def test_a_failed_cookie_save_still_returns_the_body(
        self,
        session: FetchSession,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with (
            _serve(_response(headers=[("set-cookie", "a=1")])),
            patch.object(session.profiles, "save", side_effect=OSError("disk full")),
        ):
            body = fetch.fetch("https://x.com/p", session=session)
        assert body == b"ok"
        assert "disk full" in caplog.text

    def test_caller_on_response_still_fires(self, session: FetchSession) -> None:
        seen: list[int] = []
        with _serve(_response(headers=[("set-cookie", "a=1")])):
            fetch.fetch(
                "https://x.com/p",
                request=RequestParams(
                    observe=ObserveParams(on_response=lambda s, _h: seen.append(s)),
                ),
                session=session,
            )
        assert seen == [200]

    def test_burn_on_known_identity_discards_and_retries_fresh(
        self,
        session: FetchSession,
    ) -> None:
        session.profiles.save(_EGRESS, "x.com", Profile(cookies={"GSP": "old"}))
        blocked = _response(
            403,
            b'<div class="g-recaptcha" data-sitekey="x"></div>',
            [("content-type", "text/html")],
        )
        with (
            _serve(blocked, _response(content=b"ok")) as request,
            patch("wesearch.fetch.fetch.egress_ip", _known_egress),
        ):
            body = fetch.fetch("https://x.com/p", session=session)
        assert body == b"ok"
        assert request.call_count == 2
        assert _sent_headers(request, 0)["Cookie"] == "GSP=old"
        assert "Cookie" not in _sent_headers(request, 1)
        assert session.profiles.load(_EGRESS, "x.com") is None

    def test_second_burn_raises(self, session: FetchSession) -> None:
        session.profiles.save(_EGRESS, "x.com", Profile(cookies={"GSP": "x"}))
        blocked = _response(
            403,
            b'<div class="g-recaptcha" data-sitekey="x"></div>',
            [("content-type", "text/html")],
        )
        with (
            _serve(blocked, blocked),
            patch("wesearch.fetch.fetch.egress_ip", _known_egress),
            pytest.raises(PuzzleChallengeError),
        ):
            fetch.fetch("https://x.com/p", request=_curl(), session=session)

    def test_first_contact_burn_does_not_retry(self, session: FetchSession) -> None:
        blocked = _response(
            403,
            b'<div class="g-recaptcha" data-sitekey="x"></div>',
            [("content-type", "text/html")],
        )
        with _serve(blocked) as request, pytest.raises(PuzzleChallengeError):
            fetch.fetch("https://x.com/p", request=_curl(), session=session)
        assert request.call_count == 1

    def test_raw_headers_bypasses_identity(self, session: FetchSession) -> None:
        session.profiles.save(_EGRESS, "x.com", Profile(cookies={"GSP": "s"}))
        with _serve(_response()) as request:
            fetch.fetch(
                "https://x.com/p",
                request=RequestParams(
                    content=ContentParams(
                        headers={"User-Agent": "raw"},
                        cookies={"new": "2"},
                        raw_headers=True,
                    ),
                ),
                session=session,
            )
        sent = _sent_headers(request)
        assert sent["User-Agent"] == "raw"
        assert sent["Cookie"] == "new=2"

    def test_no_egress_persists_nothing(self) -> None:
        session = FetchSession()
        try:
            with patch(
                "wesearch.fetch.fetch.fetch_curl",
                side_effect=_reports(200, {"set-cookie": "GSP=z"}, "https://x.com/p"),
            ):
                body = _send(
                    "https://x.com/p",
                    _curl(),
                    session=session,
                    profile_cookies=True,
                )
        finally:
            session.close()
        assert body == b"ok"
        assert session.profiles.load("", "x.com") is None

    def test_fetch_probes_the_egress_once_per_call(self) -> None:
        session = FetchSession()
        egress = Mock(return_value="")
        try:
            with (
                patch("wesearch.fetch.fetch.egress_ip", egress),
                patch("wesearch.fetch.fetch.fetch_curl", return_value=b"ok"),
            ):
                assert fetch.fetch("https://x.example/", session=session) == b"ok"
        finally:
            session.close()
        assert egress.call_args_list == [call(session)]

    def test_without_a_session_fetch_shares_the_default_one(self) -> None:
        backend = Mock(return_value=b"ok")
        with (
            patch("wesearch.fetch.fetch.egress_ip", return_value=_EGRESS),
            patch("wesearch.fetch.fetch.fetch_curl", backend),
            patch.object(FetchSession, "close", autospec=True) as close,
        ):
            assert fetch.fetch("https://x.example/", request=_curl()) == b"ok"
            assert fetch.fetch("https://y.example/", request=_curl()) == b"ok"
        first, second = (c.kwargs["session"] for c in backend.call_args_list)
        assert first is default_session()
        assert second is first
        close.assert_not_called()

    def test_a_caller_session_stays_open(self, session: FetchSession) -> None:
        with (
            patch("wesearch.fetch.fetch.fetch_curl", return_value=b"ok"),
            patch.object(FetchSession, "close", autospec=True) as close,
        ):
            fetch.fetch("https://x.example/", request=_curl(), session=session)
        close.assert_not_called()

    def test_known_profile_burn_discards_exact_identity_before_retry(
        self,
        session: FetchSession,
    ) -> None:
        session.egress = "198.51.100.8"
        session.profiles.save("198.51.100.8", "x.example", Profile(cookies={"o": "1"}))
        backend = Mock(side_effect=[BotDetectionError("blocked"), b"ok"])
        egress = Mock(side_effect=_rolling_egress("198.51.100.9"))
        with (
            patch("wesearch.fetch.fetch.fetch_curl", backend),
            patch("wesearch.fetch.fetch.egress_ip", egress),
        ):
            assert (
                fetch.fetch(
                    "https://x.example/path",
                    request=_curl(
                        content=ContentParams(headers={"H": "v"}, cookies={"c": "1"}),
                    ),
                    session=session,
                )
                == b"ok"
            )
        assert session.profiles.load("198.51.100.8", "x.example") is None
        assert egress.call_args_list == [call(session), call(session, refresh=True)]
        assert [_sent_headers(backend, i) for i in (0, 1)] == [
            {"H": "v", "Cookie": "o=1; c=1"},
            {"H": "v", "Cookie": "c=1"},
        ]

    def test_a_burned_browser_identity_is_the_answer(
        self,
        session: FetchSession,
    ) -> None:
        session.profiles.save(_EGRESS, "x.example", Profile(cookies={"old": "1"}))
        browser = Mock(side_effect=BotDetectionError("blocked"))
        with (
            patch.object(zendriver, "fetch_zendriver", browser),
            pytest.raises(BotDetectionError),
        ):
            fetch.fetch("https://x.example/path", request=_zendriver(), session=session)
        browser.assert_called_once()
        assert session.profiles.load(_EGRESS, "x.example") is not None

    def test_stored_and_caller_cookies_merge_on_every_transport(
        self,
        session: FetchSession,
    ) -> None:
        session.profiles.save(_EGRESS, "x.example", Profile(cookies={"old": "1"}))
        backend = Mock(
            side_effect=_reports(
                200,
                {"set-cookie": "new=3"},
                "https://x.example:8443/path",
            ),
        )
        with patch("wesearch.fetch.fetch.fetch_stdlib", backend):
            fetch.fetch(
                "https://x.example:8443/path",
                request=_stdlib(
                    content=ContentParams(headers={"H": "v"}, cookies={"new": "2"}),
                ),
                session=session,
            )
        assert _sent_headers(backend) == {"H": "v", "Cookie": "old=1; new=2"}
        stored = session.profiles.load(_EGRESS, "x.example")
        assert stored is not None
        assert stored.cookies == {"old": "1", "new": "3"}


class TestEgressIp:
    """``egress_ip`` probes an echo cascade and keeps the IPv4 answer on the session."""

    def _probe(
        self,
        fetch_mock: Mock,
        *,
        ipv6: bool = False,
        session: FetchSession | None = None,
    ) -> str:
        with patch("wesearch.fetch.fetch.fetch", fetch_mock):
            return egress_ip(session or FetchSession(), refresh=True, ipv6=ipv6)

    def test_first_echo_returned(self) -> None:
        assert self._probe(Mock(return_value=b" 203.0.113.7\n")) == "203.0.113.7"

    def test_a_failed_probe_is_not_repeated_until_the_backoff_ends(self) -> None:
        session = FetchSession()
        echo = Mock(side_effect=FetchError("https://echo.example", 0, {}, b""))
        clock = [1000.0]
        with (
            patch("wesearch.fetch.fetch.fetch", echo),
            patch("wesearch.fetch.fetch.time.monotonic", lambda: clock[0]),
        ):
            assert egress_ip(session, v4_echoes=("https://echo.example",)) == ""
            clock[0] += 59.0
            assert egress_ip(session, v4_echoes=("https://echo.example",)) == ""
            assert echo.call_count == 1
            clock[0] += 2.0
            assert egress_ip(session, v4_echoes=("https://echo.example",)) == ""
        assert echo.call_count == 2

    def test_refresh_probes_through_the_backoff(self) -> None:
        session = FetchSession()
        echo = Mock(
            side_effect=[
                FetchError("https://echo.example", 0, {}, b""),
                b"203.0.113.7",
            ],
        )
        with (
            patch("wesearch.fetch.fetch.fetch", echo),
            patch("wesearch.fetch.fetch.time.monotonic", return_value=1000.0),
        ):
            egress_ip(session, v4_echoes=("https://echo.example",))
            assert (
                egress_ip(session, refresh=True, v4_echoes=("https://echo.example",))
                == "203.0.113.7"
            )

    def test_probe_builds_exact_raw_echo_request(self) -> None:
        session = FetchSession()
        echo = Mock(return_value=b"203.0.113.7")
        with patch("wesearch.fetch.fetch.fetch", echo):
            assert (
                egress_ip(
                    session,
                    v4_echoes=("https://echo.example",),
                    timeout_sec=7.0,
                )
                == "203.0.113.7"
            )
        assert echo.call_args.args == ("https://echo.example",)
        assert echo.call_args.kwargs["session"] is session
        request = echo.call_args.kwargs["request"]
        assert isinstance(request, RequestParams)
        assert request.content.headers == {}
        assert request.content.raw_headers is True
        assert request.retry.timeout_sec == 7.0

    def test_probe_default_timeout_is_five_seconds(self) -> None:
        echo = Mock(return_value=b"203.0.113.7")
        with patch("wesearch.fetch.fetch.fetch", echo):
            egress_ip(FetchSession(), v4_echoes=("https://echo.example",))
        request = echo.call_args.kwargs["request"]
        assert isinstance(request, RequestParams)
        assert request.retry.timeout_sec == 5.0

    def test_probe_selects_v4_echoes_when_ipv6_is_false(self) -> None:
        echo = Mock(return_value=b"203.0.113.7")
        with patch("wesearch.fetch.fetch.fetch", echo):
            assert (
                egress_ip(
                    FetchSession(),
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

    def test_all_fail_resolves_empty(self) -> None:
        offline = FetchError("u", 0, {}, b"offline")
        assert self._probe(Mock(side_effect=offline)) == ""

    def test_v6_echo_returned(self) -> None:
        assert (
            self._probe(Mock(return_value=b"2606:4700:4700::1111\n"), ipv6=True)
            == "2606:4700:4700::1111"
        )

    def test_v4_reply_rejected_for_v6_request(self) -> None:
        assert self._probe(Mock(return_value=b"203.0.113.7"), ipv6=True) == ""

    def test_uses_v6_endpoints(self) -> None:
        mock = Mock(return_value=b"2001:db8::5")
        self._probe(mock, ipv6=True)
        url = mock.call_args.args[0]
        assert isinstance(url, str)
        assert "ipv6" in url or "api64" in url

    def test_malformed_v6_reply_rejected(self) -> None:
        assert self._probe(Mock(return_value=b"::::"), ipv6=True) == ""
        assert self._probe(Mock(return_value=b"ff:"), ipv6=True) == ""

    def test_probe_records_the_ipv4_answer(self) -> None:
        session = FetchSession()
        self._probe(Mock(return_value=b"203.0.113.7"), session=session)
        assert session.egress == "203.0.113.7"

    def test_an_ipv6_answer_is_never_kept(self) -> None:
        session = FetchSession(egress="203.0.113.7")
        assert (
            self._probe(Mock(return_value=b"2001:db8::5"), ipv6=True, session=session)
            == "2001:db8::5"
        )
        assert session.egress == "203.0.113.7"

    def test_a_known_egress_returns_without_probing(self) -> None:
        echo = Mock()
        with patch("wesearch.fetch.fetch.fetch", echo):
            assert egress_ip(FetchSession(egress="9.9.9.9")) == "9.9.9.9"
        echo.assert_not_called()

    def test_an_unknown_egress_is_probed(self) -> None:
        session = FetchSession()
        echo = Mock(return_value=b"1.2.3.4")
        with patch("wesearch.fetch.fetch.fetch", echo):
            assert egress_ip(session) == "1.2.3.4"
        assert echo.call_count == 1
        assert session.egress == "1.2.3.4"

    def test_refresh_always_probes_and_refreshes(self) -> None:
        session = FetchSession(egress="1.1.1.1")
        assert self._probe(Mock(return_value=b"2.2.2.2"), session=session) == "2.2.2.2"
        assert session.egress == "2.2.2.2"

    def test_failed_probe_leaves_egress_untouched(self) -> None:
        session = FetchSession(egress="192.0.2.9")
        offline = FetchError("u", 0, {}, b"x")
        assert self._probe(Mock(side_effect=offline), session=session) == ""
        assert session.egress == "192.0.2.9"

    def test_an_egress_roll_closes_only_the_old_ips_connections(self) -> None:
        # Connections out the old address would present the new identity from
        # the old IP. Other threads' requests on the new IP, and the browser,
        # which follows the host's routing, must keep running.
        old, new, browsers = _Counted(), _Counted(), _Counted()
        curl: dict[CurlKey, Closable] = {
            ("203.0.113.9", "a.example", None, 443): old,
            ("203.0.113.10", "a.example", None, 443): new,
        }
        session = FetchSession(egress="203.0.113.9", browsers=browsers, curl=curl)
        self._probe(Mock(return_value=b"203.0.113.10"), session=session)
        self._probe(Mock(return_value=b"203.0.113.10"), session=session)
        assert (old.closes, new.closes, browsers.shutdowns) == (1, 0, 0)
        assert list(session.curl) == [("203.0.113.10", "a.example", None, 443)]
        assert session.egress == "203.0.113.10"

    def test_a_first_probe_closes_the_connections_opened_before_it(self) -> None:
        # Keyed by an unknown egress, they would never be reused again.
        early = _Counted()
        curl: dict[CurlKey, Closable] = {("", "a.example", None, 443): early}
        session = FetchSession(curl=curl)
        self._probe(Mock(return_value=b"203.0.113.10"), session=session)
        assert early.closes == 1
        assert session.curl == {}
        assert session.egress == "203.0.113.10"


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

    def test_accepts_public_only_hosts(self) -> None:
        # Chrome owns its DNS, so a public-only browser request checks the host
        # and declines to pin; it does not refuse the request.
        params = RequestParams(policy=PolicyParams(transport="zendriver"))
        assert params.policy.allow_private_hosts is False

    def test_default_transport_is_auto(self) -> None:
        assert RequestParams().policy.transport == "auto"

    def test_auto_uses_general_curl_then_browser_fallback(self) -> None:
        assert resolve_transport("auto") == "curl-then-zendriver"

    def test_auto_uses_curl_for_post(self) -> None:
        assert resolve_transport("auto", method="POST") == "curl"

    @pytest.mark.parametrize(
        ("transport", "method", "resolved"),
        [
            ("auto", "POST", "stdlib"),
            ("curl", "GET", "stdlib"),
            ("auto", "GET", "curl-then-zendriver"),
            ("zendriver", "GET", "zendriver"),
        ],
    )
    def test_stdlib_stands_in_for_curl_when_curl_cannot_load(
        self,
        transport: Transport,
        method: HttpMethod,
        resolved: Transport,
    ) -> None:
        with patch("wesearch.fetch.fetch.curl_available", return_value=False):
            assert resolve_transport(transport, method=method) == resolved

    def test_the_curl_leg_of_a_browser_fallback_also_falls_back(
        self,
        session: FetchSession,
    ) -> None:
        stdlib = Mock(return_value=b"ok")
        with (
            patch("wesearch.fetch.fetch.curl_available", return_value=False),
            patch("wesearch.fetch.fetch.fetch_stdlib", stdlib),
            patch.object(zendriver, "fetch_zendriver") as browser,
        ):
            assert (
                fetch.fetch(
                    "https://x.example/",
                    request=RequestParams(
                        policy=PolicyParams(transport="curl-then-zendriver"),
                    ),
                    session=session,
                )
                == b"ok"
            )
        stdlib.assert_called_once()
        browser.assert_not_called()

    def test_auto_uses_curl_for_get_body(self, session: FetchSession) -> None:
        with (
            patch("wesearch.fetch.fetch.fetch_curl", return_value=b"ok") as curl,
            patch.object(zendriver, "fetch_zendriver") as browser,
        ):
            body = fetch.fetch(
                "https://google.com/api",
                request=RequestParams(content=ContentParams(json={"query": "value"})),
                session=session,
            )
        assert body == b"ok"
        curl.assert_called_once()
        browser.assert_not_called()

    def test_auto_post_to_learned_domain_uses_curl(self, session: FetchSession) -> None:
        # A domain learned to need the browser must not override method/body
        # eligibility: the browser leg is GET-only.
        with (
            patch.object(
                transport_routing,
                "zendriver_domains",
                return_value=frozenset({"walled.example"}),
            ),
            patch("wesearch.fetch.fetch.fetch_curl", return_value=b"ok") as curl,
            patch.object(zendriver, "fetch_zendriver") as browser,
        ):
            body = fetch.fetch(
                "https://walled.example/api",
                request=RequestParams(content=ContentParams(json={"q": "v"})),
                session=session,
            )
        assert body == b"ok"
        curl.assert_called_once()
        browser.assert_not_called()

    def test_browser_fetch_observes_a_cookie_free_response(
        self,
        session: FetchSession,
    ) -> None:
        # ObserveParams.on_response promises a callback for EVERY response.
        observer = Mock()
        with patch.object(
            zendriver,
            "fetch_zendriver",
            _landed(b"ok", "", "https://walled.example/x"),
        ):
            fetch.fetch(
                "https://walled.example/x",
                request=_zendriver(observe=ObserveParams(on_response=observer)),
                session=session,
            )
        observer.assert_called_once_with(200, {})

    def test_browser_fetch_forwards_url_params_headers_and_cookies(
        self,
        session: FetchSession,
    ) -> None:
        with patch.object(zendriver, "fetch_zendriver", return_value=b"ok") as via:
            fetch.fetch(
                "https://google.com/search?hl=en",
                request=_zendriver(
                    content=ContentParams(
                        params={"q": "test query"},
                        headers={"X-Test": "yes"},
                        cookies={"CONSENT": "YES+"},
                    ),
                ),
                session=session,
            )
        assert via.call_args.args == ("https://google.com/search?hl=en&q=test+query",)
        assert via.call_args.kwargs["headers"] == {
            "X-Test": "yes",
            "Cookie": "CONSENT=YES+",
        }
        assert set(via.call_args.kwargs) == _fetch_keywords()

    def test_browser_fetch_forwards_the_redirect_budget(
        self,
        session: FetchSession,
    ) -> None:
        # RetryParams documents 0 as "no redirects"; Chrome's own default is 10.
        with patch.object(zendriver, "fetch_zendriver", return_value=b"ok") as via:
            fetch.fetch(
                "https://walled.example/x",
                request=_zendriver(retry=RetryParams(max_redirects=0)),
                session=session,
            )
        assert via.call_args.kwargs["max_redirects"] == 0

    def test_browser_call_forwards_exact_arguments(self, session: FetchSession) -> None:
        redirect = Mock()
        request = RequestParams(
            policy=PolicyParams(transport="zendriver", allow_private_hosts=True),
            retry=RetryParams(timeout_sec=9.0),
            observe=ObserveParams(on_redirect=redirect),
        )
        with patch.object(zendriver, "fetch_zendriver", return_value=b"ok") as browser:
            assert (
                fetch.fetch("https://x.example/path", request=request, session=session)
                == b"ok"
            )
        browser.assert_called_once_with(
            "https://x.example/path",
            method="GET",
            headers={},
            body=b"",
            timeout_sec=9.0,
            connect_timeout_sec=10.0,
            max_redirects=10,
            on_redirect=redirect,
            on_response=ANY,
            allow_private_hosts=True,
            session=session,
            cookies_for_host=ANY,
        )

    def test_browser_fetch_without_egress_persists_nothing(self) -> None:
        session = FetchSession()
        try:
            with (
                patch("wesearch.fetch.fetch.egress_ip", return_value=""),
                patch.object(
                    zendriver,
                    "fetch_zendriver",
                    _landed(b"<html>rendered</html>", "SID=xyz", "https://w.example/x"),
                ),
            ):
                body = fetch.fetch(
                    "https://w.example/x",
                    request=_zendriver(),
                    session=session,
                )
        finally:
            session.close()
        assert body == b"<html>rendered</html>"
        assert session.profiles.load("", "w.example") is None

    def test_browser_fetch_persists_cookies_to_profile_store(
        self,
        session: FetchSession,
    ) -> None:
        with patch.object(
            zendriver,
            "fetch_zendriver",
            _landed(b"ok", "cf_clearance=tok", "https://walled.example/x"),
        ):
            fetch.fetch(
                "https://walled.example/x",
                request=_zendriver(),
                session=session,
            )
        saved = session.profiles.load(_EGRESS, "walled.example")
        assert saved is not None
        assert saved.cookies == {"cf_clearance": "tok"}

    def test_browser_existing_profile_is_updated_in_place(
        self,
        session: FetchSession,
    ) -> None:
        session.profiles.save(_EGRESS, "x.example", Profile(cookies={"old": "1"}))
        with patch.object(
            zendriver,
            "fetch_zendriver",
            _landed(b"ok", "sid=new", "https://x.example/path"),
        ):
            fetch.fetch("https://x.example/path", request=_zendriver(), session=session)
        stored = session.profiles.load(_EGRESS, "x.example")
        assert stored is not None
        assert stored.cookies == {"old": "1", "sid": "new"}

    def test_browser_empty_landing_domain_does_not_persist(
        self,
        session: FetchSession,
    ) -> None:
        with patch.object(
            zendriver,
            "fetch_zendriver",
            _landed(b"ok", "sid=1", "https:///path"),
        ):
            assert (
                fetch.fetch("https://x.example/", request=_zendriver(), session=session)
                == b"ok"
            )
        assert session.profiles.load(_EGRESS, "x.example") is None
        assert session.profiles.load(_EGRESS, "") is None

    def test_zendriver_fetch_forwards_exact_arguments_and_files_landing(
        self,
        session: FetchSession,
    ) -> None:
        session.egress = "198.51.100.8"
        observer = Mock()
        redirect = Mock()
        request = _zendriver(
            content=ContentParams(
                params={"q": "x y"},
                headers={"X": "h"},
                cookies={"c": "v"},
            ),
            retry=RetryParams(timeout_sec=11.0),
            observe=ObserveParams(on_redirect=redirect, on_response=observer),
        )
        browser = Mock(
            side_effect=_landed(
                b"rendered",
                "sid=1\ntoken=2",
                "https://b.example/landed",
            ),
        )
        with patch.object(zendriver, "fetch_zendriver", browser):
            assert (
                fetch.fetch("https://a.example/path", request=request, session=session)
                == b"rendered"
            )
        browser.assert_called_once()
        assert browser.call_args.args[0] == "https://a.example/path?q=x+y"
        kwargs = browser.call_args.kwargs
        assert kwargs["timeout_sec"] == 11.0
        assert kwargs["headers"] == {"X": "h", "Cookie": "c=v"}
        assert kwargs["allow_private_hosts"] is False
        assert kwargs["max_redirects"] == 10
        assert kwargs["on_redirect"] is redirect
        observer.assert_called_once_with(200, {"set-cookie": "sid=1\ntoken=2"})
        stored = session.profiles.load("198.51.100.8", "b.example")
        assert stored is not None
        assert stored.cookies == {"sid": "1", "token": "2"}


class TestCurlThenZendriverBackend:
    """``transport="curl-then-zendriver"``: curl first, zendriver only on a bot block."""

    def test_curl_then_zendriver_inherits_zendriver_restrictions(self) -> None:
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
        session: FetchSession,
    ) -> None:
        with (
            patch("wesearch.fetch.fetch.fetch_curl", return_value=b"curl body"),
            patch.object(zendriver, "fetch_zendriver") as via,
        ):
            body = fetch.fetch(
                "https://ok.example/",
                request=_curl_then_zendriver(),
                session=session,
            )
        assert body == b"curl body"
        via.assert_not_called()

    def test_curl_then_zendriver_falls_back_to_zendriver_on_bot_block(
        self,
        session: FetchSession,
    ) -> None:
        with (
            patch(
                "wesearch.fetch.fetch.fetch_curl",
                side_effect=CloudflareChallengeError(),
            ) as curl,
            patch.object(zendriver, "fetch_zendriver", return_value=b"rendered") as via,
            patch.object(transport_routing, "remember_zendriver_domain") as remember,
        ):
            body = fetch.fetch(
                "https://walled.example/",
                request=_curl_then_zendriver(),
                session=session,
            )
        assert body == b"rendered"
        curl.assert_called_once()
        via.assert_called_once()
        remember.assert_called_once_with("walled.example")

    def test_raw_headers_request_still_runs_the_body_validator(
        self,
        session: FetchSession,
    ) -> None:
        # A challenge page served with HTTP 200 must reach a raw-header caller as
        # a typed error, not an ordinary body.
        seen: list[bytes] = []

        def validate_body(body: bytes) -> None:
            seen.append(body)
            raise PuzzleChallengeError("challenge served")

        with (
            patch("wesearch.fetch.fetch.fetch_curl", return_value=b"<form id=x>"),
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
                session=session,
            )
        assert seen == [b"<form id=x>"]

    def test_success_body_challenge_falls_back_and_remembers_domain(
        self,
        session: FetchSession,
    ) -> None:
        # The validator runs on each leg, so a 200 that is really a block sends
        # the request to the browser.
        def validate_body(body: bytes) -> None:
            if b"enablejs" in body:
                raise GoogleJavascriptRequiredError("JavaScript required")

        with (
            patch("wesearch.fetch.fetch.fetch_curl", return_value=b"enablejs"),
            patch.object(transport_routing, "remember_zendriver_domain") as remember,
            patch.object(zendriver, "fetch_zendriver", return_value=b"rendered") as via,
        ):
            body = fetch.fetch(
                "https://walled.example/",
                request=_curl_then_zendriver(
                    observe=ObserveParams(body_validator=validate_body),
                ),
                session=session,
            )
        assert body == b"rendered"
        via.assert_called_once()
        remember.assert_called_once_with("walled.example")

    @pytest.mark.parametrize(
        "block",
        [
            CooldownError(),
            PuzzleChallengeError(),
            GoogleSorryError(),
            BotDetectionError(),
        ],
        ids=type,
    )
    def test_a_block_no_browser_clears_is_not_remembered(
        self,
        session: FetchSession,
        block: BotDetectionError,
    ) -> None:
        with (
            patch("wesearch.fetch.fetch.fetch_curl", side_effect=block),
            patch.object(transport_routing, "remember_zendriver_domain") as remember,
            patch.object(zendriver, "fetch_zendriver", return_value=b"rendered"),
        ):
            fetch.fetch(
                "https://walled.example/",
                request=_curl_then_zendriver(),
                session=session,
            )
        remember.assert_not_called()

    def test_bot_block_remembers_domain_even_when_browser_still_fails(
        self,
        session: FetchSession,
    ) -> None:
        with (
            patch(
                "wesearch.fetch.fetch.fetch_curl",
                side_effect=CloudflareChallengeError(),
            ),
            patch.object(transport_routing, "remember_zendriver_domain") as remember,
            patch.object(
                zendriver,
                "fetch_zendriver",
                side_effect=PuzzleChallengeError("human required"),
            ),
            pytest.raises(PuzzleChallengeError, match="human required"),
        ):
            fetch.fetch(
                "https://walled.example/",
                request=_curl_then_zendriver(),
                session=session,
            )
        remember.assert_called_once_with("walled.example")

    def test_auto_reuses_persisted_zendriver_fallback(
        self,
        session: FetchSession,
    ) -> None:
        domains: set[str] = set()
        with (
            patch(
                "wesearch.fetch.fetch.fetch_curl",
                side_effect=CloudflareChallengeError(),
            ) as curl,
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
            patch.object(zendriver, "fetch_zendriver", return_value=b"rendered") as via,
        ):
            first = fetch.fetch("https://walled.example/", session=session)
            second = fetch.fetch("https://walled.example/", session=session)
        assert first == second == b"rendered"
        curl.assert_called_once()
        assert via.call_count == 2

    def test_curl_then_zendriver_does_not_fall_back_on_non_block_error(
        self,
        session: FetchSession,
    ) -> None:
        # A plain 404 propagates; the browser would not help.
        with (
            patch(
                "wesearch.fetch.fetch.fetch_curl",
                side_effect=FetchError("https://x.example/", 404, {}, b""),
            ),
            patch.object(zendriver, "fetch_zendriver") as via,
            pytest.raises(FetchError),
        ):
            fetch.fetch(
                "https://x.example/",
                request=_curl_then_zendriver(),
                session=session,
            )
        via.assert_not_called()

    def test_browser_fallback_forwards_caller_values_after_bot_block(
        self,
        session: FetchSession,
    ) -> None:
        with (
            patch(
                "wesearch.fetch.fetch.fetch_curl",
                side_effect=BotDetectionError("blocked"),
            ),
            patch.object(zendriver, "fetch_zendriver", return_value=b"ok") as browser,
        ):
            assert (
                fetch.fetch(
                    "https://x.example/",
                    request=_curl_then_zendriver(
                        content=ContentParams(headers={"X": "1"}, cookies={"c": "2"}),
                    ),
                    session=session,
                )
                == b"ok"
            )
        assert _sent_headers(browser) == {"X": "1", "Cookie": "c=2"}

    def test_domainless_url_sends_and_validates(self, session: FetchSession) -> None:
        validate = Mock()
        with patch("wesearch.fetch.fetch.fetch_curl", return_value=b"body"):
            assert (
                fetch.fetch(
                    "/relative",
                    request=RequestParams(
                        observe=ObserveParams(body_validator=validate),
                    ),
                    session=session,
                )
                == b"body"
            )
        validate.assert_called_once_with(b"body")


class TestSend:
    """``_send``: the request it builds, its retries, and what it forwards."""

    def test_builds_exact_request_and_forwards_all_options(
        self,
        session: FetchSession,
    ) -> None:
        params = RequestParams(
            content=ContentParams(
                method="POST",
                params={"q": "a b"},
                data={"x": "y"},
                headers={"X-Test": "yes", "Cookie": "h=1"},
                cookies={"sid": "cookie"},
            ),
            policy=PolicyParams(transport="curl", allow_private_hosts=True),
            retry=RetryParams(
                timeout_sec=7.0,
                connect_timeout_sec=2.0,
                max_redirects=3,
            ),
        )
        backend = Mock(return_value=b"body")
        with patch("wesearch.fetch.fetch.fetch_curl", backend):
            got = _send(
                "https://u:p@example.com/path#frag",
                params,
                session=session,
                profile_cookies=False,
            )
        assert got == b"body"
        backend.assert_called_once_with(
            "https://example.com/path?q=a+b#frag",
            method="POST",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": "Basic dTpw",
                "X-Test": "yes",
                "Cookie": "h=1; sid=cookie",
            },
            body=b"x=y",
            timeout_sec=7.0,
            connect_timeout_sec=2.0,
            max_redirects=3,
            on_redirect=ignore,
            on_response=ANY,
            allow_private_hosts=True,
            session=session,
            cookies_for_host=ANY,
        )

    def test_without_profile_cookies_a_redirect_gets_no_stored_cookies(
        self,
        session: FetchSession,
    ) -> None:
        session.profiles.save(_EGRESS, "b.com", Profile(cookies={"B": "2"}))
        with _serve(
            _response(302, b"", [("location", "https://b.com/next")]),
            _response(content=b"done"),
        ) as request:
            _send(
                "https://a.com/start",
                _curl(),
                session=session,
                profile_cookies=False,
            )
        assert "cookie" not in _lower(request, 1)

    def test_forwards_the_redirect_callback(self, session: FetchSession) -> None:
        redirect = Mock()
        backend = Mock(return_value=b"ok")
        with patch("wesearch.fetch.fetch.fetch_curl", backend):
            _send(
                "https://x.example/",
                _curl(observe=ObserveParams(on_redirect=redirect)),
                session=session,
                profile_cookies=False,
            )
        assert backend.call_args.kwargs["on_redirect"] is redirect

    def test_retries_status_zero_and_logs_exactly(
        self,
        session: FetchSession,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        backend = Mock(side_effect=[FetchError("https://x/", 0, {}, b"down"), b"ok"])
        sleeps = Mock()
        caplog.set_level("DEBUG", logger="wesearch.fetch.fetch")
        with (
            patch("wesearch.fetch.fetch.time.sleep", sleeps),
            patch("wesearch.types.params.random.uniform", return_value=0.0),
            patch("wesearch.fetch.fetch.fetch_curl", backend),
        ):
            assert (
                _send(
                    "https://x/",
                    _curl(retry=RetryParams(retries=1)),
                    session=session,
                    profile_cookies=False,
                )
                == b"ok"
            )
        assert backend.call_count == 2
        assert sleeps.call_args_list == [call(1.0)]
        assert [r.getMessage() for r in caplog.records] == [
            "fetch https://x/ -> 0, retry in 1.0s",
        ]

    def test_raises_a_connection_failure_once_retries_are_spent(
        self,
        session: FetchSession,
    ) -> None:
        error = FetchError("https://x/", 0, {}, b"offline")
        backend = Mock(side_effect=error)
        with (
            patch("wesearch.fetch.fetch.fetch_curl", backend),
            pytest.raises(FetchError) as raised,
        ):
            _send("https://x/", _curl(), session=session, profile_cookies=False)
        assert raised.value is error
        backend.assert_called_once()

    def test_does_not_retry_nonretryable_status(self, session: FetchSession) -> None:
        error = FetchError("https://x/", 404, {"h": "v"}, b"missing")
        backend = Mock(side_effect=error)
        with (
            patch("wesearch.fetch.fetch.fetch_curl", backend),
            pytest.raises(FetchError) as raised,
        ):
            _send(
                "https://x/",
                _curl(retry=RetryParams(retries=1)),
                session=session,
                profile_cookies=False,
            )
        assert raised.value is error
        backend.assert_called_once()

    def test_uses_all_of_the_retry_budget_and_no_more(
        self,
        session: FetchSession,
    ) -> None:
        error = FetchError("https://x/", 500, {}, b"retry")
        backend = Mock(side_effect=[error, error, b"ok"])
        with (
            patch("wesearch.fetch.fetch.time.sleep"),
            patch("wesearch.fetch.fetch.fetch_curl", backend),
            pytest.raises(FetchError),
        ):
            _send(
                "https://x/",
                _curl(retry=RetryParams(retries=1)),
                session=session,
                profile_cookies=False,
            )
        assert backend.call_count == 2

    def test_retries_zero_sends_once(self, session: FetchSession) -> None:
        backend = Mock(side_effect=FetchError("https://x.example/", 500, {}, b""))
        with (
            patch("wesearch.fetch.fetch.fetch_curl", backend),
            pytest.raises(FetchError),
        ):
            _send("https://x.example/", _curl(), session=session, profile_cookies=False)
        backend.assert_called_once()

    def test_retry_counter_advances_across_three_attempts(
        self,
        session: FetchSession,
    ) -> None:
        error = FetchError("https://x.example/", 500, {}, b"retry")
        backend = Mock(side_effect=[error, error, error, b"ok"])
        sleeps = Mock()
        with (
            patch("wesearch.fetch.fetch.time.sleep", sleeps),
            patch("wesearch.types.params.random.uniform", return_value=0.0),
            patch("wesearch.fetch.fetch.fetch_curl", backend),
        ):
            assert (
                _send(
                    "https://x.example/",
                    _curl(retry=RetryParams(retries=3)),
                    session=session,
                    profile_cookies=False,
                )
                == b"ok"
            )
        assert backend.call_count == 4
        assert sleeps.call_args_list == [call(1.0), call(2.0), call(4.0)]

    def test_json_body_is_serialized(self, session: FetchSession) -> None:
        backend = Mock(return_value=b"ok")
        with patch("wesearch.fetch.fetch.fetch_curl", backend):
            _send(
                "https://x/",
                _curl(content=ContentParams(method="POST", json={"a": 1})),
                session=session,
                profile_cookies=False,
            )
        assert backend.call_args.kwargs["body"] == b'{"a": 1}'
        assert _sent_headers(backend) == {"Content-Type": "application/json"}

    def test_case_variant_cookie_headers_collapse_to_the_last(
        self,
        session: FetchSession,
    ) -> None:
        backend = Mock(return_value=b"ok")
        with patch("wesearch.fetch.fetch.fetch_curl", backend):
            _send(
                "https://x/",
                _curl(
                    content=ContentParams(
                        headers={"cookie": "a=header", "COOKIE": "b=header"},
                        cookies={"c": "param"},
                    ),
                ),
                session=session,
                profile_cookies=False,
            )
        assert _sent_headers(backend) == {"Cookie": "b=header; c=param"}

    def test_raw_headers_imply_no_content_type(self, session: FetchSession) -> None:
        backend = Mock(return_value=b"ok")
        with patch("wesearch.fetch.fetch.fetch_curl", backend):
            _send(
                "https://x.example/",
                _curl(
                    content=ContentParams(
                        method="POST",
                        json={"a": 1},
                        raw_headers=True,
                    ),
                ),
                session=session,
                profile_cookies=False,
            )
        assert backend.call_args.kwargs["headers"] == {}
        assert backend.call_args.kwargs["body"] == b'{"a": 1}'

    def test_joins_two_cookie_values_with_semicolon(
        self,
        session: FetchSession,
    ) -> None:
        backend = Mock(return_value=b"ok")
        with patch("wesearch.fetch.fetch.fetch_curl", backend):
            _send(
                "https://x.example/",
                _curl(content=ContentParams(cookies={"a": "1", "b": "2"})),
                session=session,
                profile_cookies=False,
            )
        assert _sent_headers(backend)["Cookie"] == "a=1; b=2"

    def test_a_get_sends_no_body_and_no_content_type(
        self,
        session: FetchSession,
    ) -> None:
        backend = Mock(return_value=b"ok")
        with patch("wesearch.fetch.fetch.fetch_stdlib", backend):
            _send(
                "https://x.example/",
                _stdlib(),
                session=session,
                profile_cookies=False,
            )
        assert backend.call_args.kwargs["headers"] == {}
        assert backend.call_args.kwargs["body"] == b""
        assert backend.call_args.kwargs["session"] is session


class TestGate:
    def test_a_final_block_locks_the_site_for_the_next_fetch(
        self,
        session: FetchSession,
    ) -> None:
        def challenge(body: bytes) -> None:
            raise PuzzleChallengeError(body.decode(), url="https://gated.example/")

        blocked = _curl(observe=ObserveParams(body_validator=challenge))
        with (
            patch("wesearch.fetch.fetch.fetch_curl", return_value=b"wall"),
            pytest.raises(PuzzleChallengeError),
        ):
            fetch.fetch("https://gated.example/", request=blocked, session=session)
        with (
            patch("wesearch.fetch.fetch.fetch_curl", return_value=b"ok") as sent,
            pytest.raises(CooldownError),
        ):
            fetch.fetch("https://gated.example/", session=session)
        sent.assert_not_called()

    def test_a_curl_block_the_browser_clears_does_not_lock(
        self,
        session: FetchSession,
    ) -> None:
        def wall_on_curl(body: bytes) -> None:
            if body == b"wall":
                raise PuzzleChallengeError("challenge")

        request = RequestParams(observe=ObserveParams(body_validator=wall_on_curl))
        with (
            patch("wesearch.fetch.fetch.fetch_curl", return_value=b"wall"),
            patch.object(zendriver, "fetch_zendriver", return_value=b"page"),
        ):
            assert (
                fetch.fetch("https://gated.example/", request=request, session=session)
                == b"page"
            )
        with patch("wesearch.fetch.fetch.fetch_curl", return_value=b"ok"):
            assert (
                fetch.fetch("https://gated.example/", request=_curl(), session=session)
                == b"ok"
            )


_CONNECT = "wesearch.fetch.transport.stdlib._connect"


def _curl(
    *,
    content: ContentParams | None = None,
    retry: RetryParams | None = None,
    observe: ObserveParams | None = None,
) -> RequestParams:
    """Return request params on the curl transport."""
    return _on("curl", content=content, retry=retry, observe=observe)


def _stdlib(
    *,
    content: ContentParams | None = None,
    retry: RetryParams | None = None,
) -> RequestParams:
    """Return request params on the stdlib transport."""
    return _on("stdlib", content=content, retry=retry)


def _zendriver(
    *,
    content: ContentParams | None = None,
    retry: RetryParams | None = None,
    observe: ObserveParams | None = None,
) -> RequestParams:
    """Return request params on the browser transport."""
    return _on("zendriver", content=content, retry=retry, observe=observe)


def _curl_then_zendriver(
    *,
    content: ContentParams | None = None,
    observe: ObserveParams | None = None,
) -> RequestParams:
    """Return request params on curl with the browser fallback."""
    return _on("curl-then-zendriver", content=content, observe=observe)


def _on(
    transport: Transport,
    *,
    content: ContentParams | None = None,
    retry: RetryParams | None = None,
    observe: ObserveParams | None = None,
) -> RequestParams:
    """Return request params on ``transport``."""
    return RequestParams(
        content=content or ContentParams(),
        retry=retry or RetryParams(),
        observe=observe or ObserveParams(),
        policy=PolicyParams(transport=transport),
    )


def _response(
    status: int = 200,
    content: bytes = b"ok",
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


class _Counted:
    """A connection and a browser pool that count their closes."""

    def __init__(self) -> None:
        self.closes = 0
        self.shutdowns = 0

    def close(self) -> None:
        self.closes += 1

    def shutdown(self) -> None:
        self.shutdowns += 1


def _http_response(
    status: int = 200,
    body: bytes = b"hello",
    headers: list[tuple[str, str]] | None = None,
) -> Mock:
    """Return an ``http.client`` response stand-in."""
    response = Mock(spec=client.HTTPResponse)
    response.status = status
    response.read.side_effect = [body, b""]
    response.isclosed.return_value = False
    response.getheaders.return_value = headers or [("content-encoding", "identity")]
    return response


def _stdlib_connection(*responses: Mock | Exception) -> Mock:
    """Return a connection whose ``getresponse`` answers ``responses`` in order."""
    connection = Mock()
    connection.getresponse.side_effect = list(responses)
    return connection


def _sent_headers(mock: Mock, index: int = 0) -> dict[str, str]:
    """Return the headers of a recorded transport call, ``None`` values dropped."""
    headers = from_plain(
        mock.call_args_list[index].kwargs["headers"],
        dict[str, str | None],
    )
    return {k: v for k, v in headers.items() if v is not None}


def _lower(mock: Mock, index: int) -> dict[str, str]:
    """Return :func:`_sent_headers` with names lower-cased."""
    return {k.lower(): v for k, v in _sent_headers(mock, index).items()}


def _curl_user_agent() -> str:
    """Return the User-Agent curl presents: curl_cffi's newest Chrome."""
    major = int(cc_requests.impersonate.REAL_TARGET_MAP["chrome"][len("chrome") :])
    return chrome_identity_headers(major=major, platform=chrome_platform("chrome"))[
        "User-Agent"
    ]


def _fetch_keywords() -> set[str]:
    """Return the keyword parameters every transport takes."""
    return set(inspect.signature(Fetch.__call__).parameters) - {"self", "url"}


def _known_egress(session: FetchSession, **_kw: object) -> str:
    """Return the session's egress, as a probe that answers the same IP would."""
    return session.egress


def _rolling_egress(new: str) -> Callable[..., str]:
    """Return an ``egress_ip`` stand-in whose refresh answers ``new``."""

    def egress(session: FetchSession, *, refresh: bool = False) -> str:
        if refresh:
            session.egress = new
        return session.egress

    return egress


def _reports(
    status: int,
    headers: Mapping[str, str],
    url: str,
) -> Callable[..., bytes]:
    """Return a transport stand-in that reports one response and returns ``ok``."""

    def transport(*_args: object, **kwargs: object) -> bytes:
        callback = kwargs["on_response"]
        assert callable(callback)
        callback(status, dict(headers), url)
        return b"ok"

    return transport


def _landed(body: bytes, set_cookie: str, url: str) -> Callable[..., bytes]:
    """Return a ``fetch_zendriver`` stand-in that lands on ``url`` with cookies."""

    def browser(*_args: object, **kwargs: object) -> bytes:
        observe = kwargs["on_response"]
        assert callable(observe)
        observe(200, {"set-cookie": set_cookie} if set_cookie else {}, url)
        return body

    return browser


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
