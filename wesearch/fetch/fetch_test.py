"""Tests for wesearch.fetch."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from typing import TYPE_CHECKING, cast
from unittest.mock import ANY, Mock, call, patch

import importlib
import logging

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
    _send,
    egress_ip,
    resolve_transport,
)
from wesearch.fetch.gate import gated
from wesearch.fetch.session import default_session
from wesearch.fetch.transport import transport_routing, zendriver
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


if TYPE_CHECKING:
    from collections.abc import Iterator

    from wesearch.fetch.types import Closable, CurlKey
    from wesearch.types.params import Transport


# A public egress, so ``fetch`` never probes for one.
_EGRESS = "203.0.113.1"

_APP_SHELL = (
    b'<!DOCTYPE html><html><head><script src="/app.js"></script></head>'
    b'<body><div id="root"></div></body></html>'
)
_REDIRECT_STUB = (
    b"<!DOCTYPE html><title>Redirecting&hellip;</title>"
    b'<meta http-equiv="refresh" content="0; url=../2.15/x.html">'
)
_AWS_WAF_WALL = (
    b"<!DOCTYPE html><html><head><title></title></head><body>"
    b"<script>AwsWafIntegration.getToken()</script></body></html>"
)


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


class TestFetchEntryAndGate:
    """``fetch`` resolves session/egress, gates the bare URL, and tags blocks."""

    def test_defaults_share_default_session_and_probe_egress_once(self) -> None:
        backend = Mock(return_value=b"ok")
        egress = Mock(return_value="198.51.100.4")
        with (
            patch("wesearch.fetch.fetch.egress_ip", egress),
            patch("wesearch.fetch.fetch.fetch_curl", backend),
            patch("wesearch.fetch.fetch.gated", wraps=gated) as gate,
            patch.object(FetchSession, "close", autospec=True) as close,
        ):
            assert fetch.fetch("https://x.example/") == b"ok"
            assert fetch.fetch("https://y.example/") == b"ok"
        shared = default_session()
        assert [c.kwargs["session"] for c in backend.call_args_list] == [shared, shared]
        assert egress.call_args_list == [call(shared), call(shared)]
        assert gate.call_args.kwargs == {"egress": "198.51.100.4", "gate": shared.gate}
        close.assert_not_called()

    def test_credentialed_url_and_validator_block_strip_userinfo(
        self,
        session: FetchSession,
    ) -> None:
        def validate_body(body: bytes) -> None:
            raise PuzzleChallengeError(
                "challenge served",
                status=200,
                headers={"server": "edge"},
                body=body,
            )

        with (
            patch(
                "wesearch.fetch.fetch.fetch_curl",
                return_value=b"<form id=x>",
            ) as sent,
            patch("wesearch.fetch.fetch.egress_ip") as probe,
            patch("wesearch.fetch.fetch.gated", wraps=gated) as gate,
            pytest.raises(PuzzleChallengeError) as error,
        ):
            fetch.fetch(
                "https://u:secret@raw.example/",
                request=RequestParams(
                    content=ContentParams(
                        headers={"User-Agent": "x"},
                        raw_headers=True,
                    ),
                    observe=ObserveParams(body_validator=validate_body),
                ),
                session=session,
            )
        probe.assert_not_called()
        assert sent.call_args.args == ("https://raw.example/",)
        assert sent.call_args.kwargs["session"] is session
        assert gate.call_args.args == ("https://raw.example/",)
        assert gate.call_args.kwargs == {"egress": _EGRESS, "gate": session.gate}
        assert "secret" not in str(error.value)
        assert (
            error.value.url,
            error.value.status,
            error.value.headers,
            error.value.body,
        ) == ("https://raw.example/", 200, {"server": "edge"}, b"<form id=x>")
        assert str(error.value) == (
            "challenge served Run `webrowser https://raw.example/` and solve the "
            "challenge; the site unlocks once its page loads."
        )

    def test_gate_locks_only_the_egress_of_a_final_block(
        self,
        session: FetchSession,
    ) -> None:
        already_named = PuzzleChallengeError(
            "wall",
            url="https://gated.example/challenge",
        )
        with (
            patch("wesearch.fetch.fetch.fetch_curl", side_effect=already_named),
            pytest.raises(PuzzleChallengeError) as raised,
        ):
            fetch.fetch("https://gated.example/", request=_curl(), session=session)
        assert raised.value is already_named

        with (
            patch("wesearch.fetch.fetch.fetch_curl", return_value=b"ok") as sent,
            pytest.raises(CooldownError),
        ):
            fetch.fetch("https://gated.example/", request=_curl(), session=session)
        sent.assert_not_called()

        session.egress = "198.51.100.9"
        with patch("wesearch.fetch.fetch.fetch_curl", return_value=b"ok"):
            assert (
                fetch.fetch("https://gated.example/", request=_curl(), session=session)
                == b"ok"
            )

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


class TestResolveTransport:
    """``resolve_transport`` maps ``auto``, learned domains, and missing curl."""

    @pytest.mark.parametrize(
        ("transport", "available", "expected"),
        [
            ("curl", True, "curl"),
            ("stdlib", True, "stdlib"),
            ("zendriver", True, "browser"),
            ("curl", False, "stdlib"),
            ("curl-then-zendriver", False, "stdlib"),
        ],
    )
    def test_selected_transport_sends_the_request(
        self,
        session: FetchSession,
        transport: Transport,
        available: bool,
        expected: str,
    ) -> None:
        curl, stdlib = Mock(return_value=b"curl"), Mock(return_value=b"stdlib")
        with (
            patch("wesearch.fetch.fetch.curl_available", return_value=available),
            patch("wesearch.fetch.fetch.fetch_curl", curl),
            patch("wesearch.fetch.fetch.fetch_stdlib", stdlib),
            patch.object(
                zendriver,
                "fetch_zendriver",
                return_value=b"browser",
            ) as browser,
        ):
            assert (
                fetch.fetch(
                    "https://x.example/",
                    request=RequestParams(policy=PolicyParams(transport=transport)),
                    session=session,
                )
                == expected.encode()
            )
        for name, backend in (("curl", curl), ("stdlib", stdlib), ("browser", browser)):
            assert [c.args for c in backend.call_args_list] == (
                [("https://x.example/",)] if name == expected else []
            )

    def test_resolves_auto_domains_and_curl_availability(self) -> None:
        queried: list[str] = []

        class _Domains:
            def __contains__(self, domain: object) -> bool:
                assert isinstance(domain, str)
                queried.append(domain)
                return domain == "walled.example"

        with patch.object(
            transport_routing,
            "zendriver_domains",
            return_value=_Domains(),
        ):
            assert resolve_transport("auto") == "curl-then-zendriver"
            assert queried == [""]
            assert resolve_transport("auto", domain="walled.example") == "zendriver"
            assert resolve_transport("auto", method="POST") == "curl"
            assert resolve_transport("auto", raw_headers=True) == "curl"
            assert resolve_transport("auto", has_body=True) == "curl"
            assert (
                resolve_transport("auto", method="POST", domain="walled.example")
                == "curl"
            )

        with patch("wesearch.fetch.fetch.curl_available", return_value=False):
            assert resolve_transport("auto", method="POST") == "stdlib"
            assert resolve_transport("curl") == "stdlib"
            assert resolve_transport("auto") == "curl-then-zendriver"
            assert resolve_transport("zendriver") == "zendriver"


class TestEgressIp:
    """``egress_ip`` probes an echo cascade and keeps the IPv4 answer on the session."""

    def test_default_echoes_and_caching_and_backoff(self) -> None:
        target = FetchSession()
        seen: list[tuple[str, float]] = []

        def fail_echo(
            url: str,
            *,
            request: RequestParams,
            session: FetchSession,
        ) -> bytes:
            assert session is target
            assert request.content.raw_headers is True
            assert request.content.headers == {}
            seen.append((url, request.retry.timeout_sec))
            raise FetchError(url, 0, {}, b"offline")

        clock = [1000.0]
        with (
            patch("wesearch.fetch.fetch.fetch", side_effect=fail_echo) as echo,
            patch("wesearch.fetch.fetch.time.monotonic", lambda: clock[0]),
        ):
            assert egress_ip(target) == ""
            assert seen == [
                ("https://ipv4.icanhazip.com", 5.0),
                ("https://api.ipify.org", 5.0),
            ]
            assert target.egress_retry_at == 1060.0

            # Within backoff: does not re-probe unless refresh=True.
            clock[0] = 1059.0
            assert egress_ip(target) == ""
            assert echo.call_count == 2

            # At exact boundary (1060.0 < 1060.0 is False): probes again.
            clock[0] = 1060.0
            assert (
                egress_ip(
                    target,
                    v4_echoes=("https://echo.example",),
                    timeout_sec=7.0,
                    retry_after_sec=2.0,
                )
                == ""
            )
            assert target.egress_retry_at == 1062.0
            assert seen[-1] == ("https://echo.example", 7.0)

        # Refresh overrides backoff; known egress returns without probing.
        with (
            patch(
                "wesearch.fetch.fetch.fetch",
                side_effect=[b"2001:db8::1", b" 203.0.113.7\n"],
            ) as echo,
            patch("wesearch.fetch.fetch.time.monotonic", return_value=1061.0),
        ):
            assert egress_ip(target, refresh=True) == "203.0.113.7"
            assert target.egress == "203.0.113.7"
            assert egress_ip(target) == "203.0.113.7"
            assert echo.call_count == 2

    def test_ipv6_probes_v6_echoes_without_mutating_session(self) -> None:
        session = FetchSession(egress="203.0.113.7")
        seen_v6: list[str] = []

        def v6_cascade(
            url: str,
            *,
            request: RequestParams,
            session: FetchSession,
        ) -> bytes:
            del request, session
            seen_v6.append(url)
            if len(seen_v6) == 1:
                return b"203.0.113.7"
            return b" 2606:4700:4700::1111\n"

        with patch("wesearch.fetch.fetch.fetch", side_effect=v6_cascade):
            assert egress_ip(session, ipv6=True) == "2606:4700:4700::1111"
        assert seen_v6 == [
            "https://ipv6.icanhazip.com",
            "https://api64.ipify.org",
        ]
        assert session.egress == "203.0.113.7"

        empty = FetchSession()
        with patch(
            "wesearch.fetch.fetch.fetch",
            side_effect=[b"::::", b"ff:"],
        ):
            assert egress_ip(empty, ipv6=True) == ""
        assert empty.egress == ""
        assert empty.egress_retry_at == 0.0

    def test_egress_roll_closes_only_stale_curl_connections(self) -> None:
        early, old, current, browsers = (
            _Counted(),
            _Counted(),
            _Counted(),
            _Counted(),
        )
        curl: dict[CurlKey, Closable] = {
            ("", "a.example", None, 443): early,
            ("203.0.113.9", "a.example", None, 443): old,
            ("203.0.113.10", "a.example", None, 443): current,
        }
        session = FetchSession(egress="203.0.113.9", browsers=browsers, curl=curl)
        with patch(
            "wesearch.fetch.fetch.fetch",
            return_value=b"203.0.113.10",
        ):
            assert egress_ip(session, refresh=True) == "203.0.113.10"
            assert egress_ip(session, refresh=True) == "203.0.113.10"
        assert (early.closes, old.closes, current.closes, browsers.shutdowns) == (
            1,
            1,
            0,
            0,
        )
        assert list(session.curl) == [("203.0.113.10", "a.example", None, 443)]
        assert session.egress == "203.0.113.10"


class TestProfileAndFallbackOrchestration:
    """``_fetch_as_profile`` and ``_render_shell``: identity burn and browser fallback."""

    def test_known_identity_burn_discards_and_retries_once_on_non_browser(
        self,
        session: FetchSession,
    ) -> None:
        session.egress = "198.51.100.8"
        session.profiles.save("198.51.100.8", "x.example", Profile(cookies={"o": "1"}))
        calls = 0

        def attempt(*args: object, **kwargs: object) -> bytes:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise PuzzleChallengeError("blocked")
            return _reports(200, {"set-cookie": "n=2"}, "https://x.example/path")(
                *args,
                **kwargs,
            )

        backend = Mock(side_effect=attempt)
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
        assert [c.args[0] for c in backend.call_args_list] == [
            "https://x.example/path",
            "https://x.example/path",
        ]
        assert [_sent_headers(backend, i) for i in (0, 1)] == [
            {"H": "v", "Cookie": "o=1; c=1"},
            {"H": "v", "Cookie": "c=1"},
        ]
        rotated = session.profiles.load("198.51.100.9", "x.example")
        assert rotated is not None
        assert rotated.cookies == {"n": "2"}

    def test_second_identity_burn_raises_without_another_retry(
        self,
        session: FetchSession,
    ) -> None:
        session.profiles.save(_EGRESS, "x.example", Profile(cookies={"old": "1"}))
        with (
            patch(
                "wesearch.fetch.fetch.fetch_curl",
                side_effect=PuzzleChallengeError("blocked"),
            ) as backend,
            patch("wesearch.fetch.fetch.egress_ip", return_value=_EGRESS),
            pytest.raises(PuzzleChallengeError),
        ):
            fetch.fetch("https://x.example/", request=_curl(), session=session)
        assert backend.call_count == 2
        assert session.profiles.load(_EGRESS, "x.example") is None

    def test_unrecoverable_burns_raise_without_retry(
        self,
        session: FetchSession,
    ) -> None:
        # First-contact burn (no stored profile) does not retry.
        first_contact = Mock(side_effect=PuzzleChallengeError("blocked"))
        with (
            patch("wesearch.fetch.fetch.fetch_curl", first_contact),
            pytest.raises(PuzzleChallengeError),
        ):
            fetch.fetch("https://x.example/", request=_curl(), session=session)
        first_contact.assert_called_once()

        # Browser burn never discards profile or retries.
        session.profiles.save(_EGRESS, "y.example", Profile(cookies={"old": "1"}))
        browser = Mock(side_effect=BotDetectionError("blocked"))
        with (
            patch.object(zendriver, "fetch_zendriver", browser),
            pytest.raises(BotDetectionError),
        ):
            fetch.fetch("https://y.example/", request=_zendriver(), session=session)
        browser.assert_called_once()
        assert session.profiles.load(_EGRESS, "y.example") is not None

    def test_curl_then_zendriver_clears_blocks_and_remembers_domain(
        self,
        session: FetchSession,
    ) -> None:
        domains: set[str] = set()
        with (
            patch(
                "wesearch.fetch.fetch.fetch_curl",
                side_effect=CloudflareChallengeError(),
            ) as curl_leg,
            patch.object(
                transport_routing,
                "zendriver_domains",
                side_effect=lambda: frozenset(domains),
            ),
            patch.object(
                transport_routing,
                "remember_zendriver_domain",
                side_effect=domains.add,
            ) as remember,
            patch.object(
                zendriver,
                "fetch_zendriver",
                return_value=b"rendered",
            ) as via,
        ):
            first = fetch.fetch(
                "https://walled.example/",
                request=RequestParams(
                    content=ContentParams(headers={"X": "1"}, cookies={"c": "2"}),
                    policy=PolicyParams(allow_private_hosts=True),
                ),
                session=session,
            )
            second = fetch.fetch("https://walled.example/", session=session)
            curl_leg.side_effect = None
            curl_leg.return_value = b"post-ok"
            assert (
                fetch.fetch(
                    "https://walled.example/api",
                    request=RequestParams(content=ContentParams(method="POST")),
                    session=session,
                )
                == b"post-ok"
            )
            assert (
                fetch.fetch(
                    "https://walled.example/api",
                    request=RequestParams(content=ContentParams(json={"q": "v"})),
                    session=session,
                )
                == b"post-ok"
            )
        assert first == second == b"rendered"
        assert curl_leg.call_count == 3
        assert via.call_count == 2
        assert via.call_args_list[0].kwargs["allow_private_hosts"] is True
        assert _sent_headers(via, 0) == {"X": "1", "Cookie": "c=2"}
        remember.assert_called_once_with("walled.example")

    @pytest.mark.parametrize(
        ("outcome", "remembered", "browser"),
        [
            (CloudflareChallengeError(), True, b"rendered"),
            (CooldownError(), False, b"rendered"),
            (PuzzleChallengeError(), False, b"rendered"),
            (GoogleSorryError(), False, b"rendered"),
            (BotDetectionError(), False, b"rendered"),
            (_AWS_WAF_WALL, True, b"rendered"),
            (CloudflareChallengeError(), True, PuzzleChallengeError("human required")),
        ],
    )
    def test_domain_memory_depends_on_whether_browser_clears_wall(
        self,
        session: FetchSession,
        outcome: bytes | BotDetectionError,
        remembered: bool,
        browser: bytes | PuzzleChallengeError,
    ) -> None:
        with (
            patch(
                "wesearch.fetch.fetch.fetch_curl",
                side_effect=[outcome],
            ) as curl_leg,
            patch.object(transport_routing, "remember_zendriver_domain") as remember,
            patch.object(zendriver, "fetch_zendriver", side_effect=[browser]) as via,
            pytest.raises(PuzzleChallengeError, match="human required")
            if isinstance(browser, PuzzleChallengeError)
            else nullcontext(),
        ):
            assert (
                fetch.fetch(
                    "https://walled.example/",
                    request=_curl_then_zendriver(),
                    session=session,
                )
                == b"rendered"
            )
        assert curl_leg.call_args.args == ("https://walled.example/",)
        assert via.call_args.args == ("https://walled.example/",)
        assert remember.call_count == remembered

    @pytest.mark.parametrize(
        ("url", "error", "browser", "expected"),
        [
            (
                "https://akamai.example/",
                FetchError("u", 403, {"Content-Type": "text/html"}, b"Denied"),
                b"rendered",
                (True, True),
            ),
            (
                "https://akamai.example/",
                FetchError("u", 403, {"Content-Type": "text/html"}, b"Denied"),
                FetchError("u", 403, {}, b"no"),
                (True, False),
            ),
            (
                "https://akamai.example/",
                FetchError("u", 403, {"Content-Type": "application/json"}, b"{}"),
                b"unused",
                (False, False),
            ),
            (
                "https://ops.example/",
                FetchError(
                    "https://ops.example/v2/ops?_rsc",
                    404,
                    {"content-type": "text/html"},
                    b"Not found",
                ),
                b"rendered",
                (True, True),
            ),
            (
                "https://ops.example/",
                FetchError(
                    "https://ops.example/",
                    404,
                    {"content-type": "text/html"},
                    b"Not found",
                ),
                b"unused",
                (False, False),
            ),
            (
                "https:///x",
                FetchError("https:///x", 403, {"content-type": "text/html"}, b"Denied"),
                b"rendered",
                (True, False),
            ),
            (
                "https:///x",
                CloudflareChallengeError(),
                b"rendered",
                (True, False),
            ),
        ],
    )
    def test_unmarked_block_and_hostless_fallback_rules(
        self,
        session: FetchSession,
        url: str,
        error: FetchError,
        browser: bytes | FetchError,
        expected: tuple[bool, bool],
    ) -> None:
        falls_back, remembers = expected
        with (
            patch("wesearch.fetch.fetch.fetch_curl", side_effect=error),
            patch.object(zendriver, "fetch_zendriver", side_effect=[browser]) as via,
            patch.object(transport_routing, "remember_zendriver_domain") as remember,
        ):
            if isinstance(browser, bytes) and falls_back:
                assert (
                    fetch.fetch(url, request=_curl_then_zendriver(), session=session)
                    == browser
                )
            else:
                with pytest.raises(FetchError):
                    fetch.fetch(url, request=_curl_then_zendriver(), session=session)
        assert [c.args[0] for c in via.call_args_list] == ([url] if falls_back else [])
        assert [c.args[0] for c in remember.call_args_list] == (
            [url.split("/")[2]] if remembers else []
        )

    def test_shell_rendering_and_unavailable_browser_fallback(
        self,
        session: FetchSession,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Non-shell page never touches the browser.
        page = b'<!doctype html><p>Comments</p><div class="g-recaptcha"></div>'
        with (
            patch("wesearch.fetch.fetch.fetch_curl", return_value=page),
            patch.object(zendriver, "fetch_zendriver") as via,
        ):
            assert (
                fetch.fetch(
                    "https://blog.example/",
                    request=_curl_then_zendriver(),
                    session=session,
                )
                == page
            )
        via.assert_not_called()

        # Fuller render wins; equal-length or shorter render keeps the shell.
        fuller = b"<html><body>" + b"word " * 200 + b"</body></html>"
        same_length = _REDIRECT_STUB.replace(b"Redirecting", b"Replacement")
        for candidate, expected in (
            (fuller, fuller),
            (same_length, _REDIRECT_STUB),
            (b"<html><body>tiny</body></html>", _REDIRECT_STUB),
        ):
            with (
                patch(
                    "wesearch.fetch.fetch.fetch_curl",
                    return_value=_REDIRECT_STUB,
                ),
                patch.object(
                    zendriver,
                    "fetch_zendriver",
                    return_value=candidate,
                ) as via,
                patch.object(
                    transport_routing,
                    "remember_zendriver_domain",
                ) as remember,
            ):
                assert (
                    fetch.fetch(
                        "https://docs.example/",
                        request=_curl_then_zendriver(),
                        session=session,
                    )
                    == expected
                )
            assert via.call_args.args == ("https://docs.example/",)
            remember.assert_not_called()

        # Unavailable browser logs warning with exc_info and returns the shell.
        with (
            patch("wesearch.fetch.fetch.fetch_curl", return_value=_REDIRECT_STUB),
            patch.object(
                zendriver,
                "fetch_zendriver",
                side_effect=zendriver.BrowserUnavailableError("no Chrome"),
            ),
            caplog.at_level(logging.WARNING, logger="wesearch.fetch.fetch"),
        ):
            assert (
                fetch.fetch(
                    "https://docs.example/",
                    request=_curl_then_zendriver(),
                    session=session,
                )
                == _REDIRECT_STUB
            )
        assert [(r.getMessage(), bool(r.exc_info)) for r in caplog.records] == [
            ("No browser to render https://docs.example/; keeping its shell.", True),
        ]

        # Any other browser error on an empty shell propagates.
        with (
            patch("wesearch.fetch.fetch.fetch_curl", return_value=_APP_SHELL),
            patch.object(
                zendriver,
                "fetch_zendriver",
                side_effect=FetchError("https://app.example/", 0, {}, b"timed out"),
            ),
            pytest.raises(FetchError, match="connection failed"),
        ):
            fetch.fetch(
                "https://app.example/",
                request=_curl_then_zendriver(),
                session=session,
            )

    def test_validator_challenge_on_curl_falls_back_to_browser(
        self,
        session: FetchSession,
    ) -> None:
        def validate_body(body: bytes) -> None:
            if b"enablejs" in body:
                raise GoogleJavascriptRequiredError("JavaScript required")

        with (
            patch("wesearch.fetch.fetch.fetch_curl", return_value=b"enablejs"),
            patch.object(transport_routing, "remember_zendriver_domain") as remember,
            patch.object(zendriver, "fetch_zendriver", return_value=b"rendered") as via,
        ):
            assert (
                fetch.fetch(
                    "https://walled.example/",
                    request=_curl_then_zendriver(
                        observe=ObserveParams(body_validator=validate_body),
                    ),
                    session=session,
                )
                == b"rendered"
            )
        via.assert_called_once()
        remember.assert_called_once_with("walled.example")


class TestSend:
    """``_send`` builds the wire request, merges cookies, retries, and saves jars."""

    def test_execution_transport_is_independent_of_caller_policy(
        self,
        session: FetchSession,
    ) -> None:
        params = RequestParams()
        backend = Mock(return_value=b"body")
        assert (
            _send("https://x.example/", params, transport=backend, session=session)
            == b"body"
        )
        backend.assert_called_once()
        assert backend.call_args.kwargs["session"] is session
        assert params.policy.transport == "auto"

    def test_builds_exact_request_and_forwards_all_options(
        self,
        session: FetchSession,
    ) -> None:
        redirect = Mock()
        params = RequestParams(
            content=ContentParams(
                method="POST",
                params={"q": "a b"},
                data={"x": "y"},
                headers={"X-Test": "yes", "Cookie": "h=1", "Drop": None},
                cookies={"sid": "cookie"},
            ),
            policy=PolicyParams(transport="curl", allow_private_hosts=True),
            retry=RetryParams(
                timeout_sec=7.0,
                connect_timeout_sec=2.0,
                max_redirects=3,
            ),
            observe=ObserveParams(on_redirect=redirect),
        )
        backend = Mock(return_value=b"body")
        assert (
            _send(
                "https://u:p@example.com/path#frag",
                params,
                transport=backend,
                session=session,
            )
            == b"body"
        )
        backend.assert_called_once_with(
            "https://example.com/path?q=a+b#frag",
            method="POST",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": "Basic dTpw",
                "X-Test": "yes",
                "Drop": None,
                "Cookie": "h=1; sid=cookie",
            },
            body=b"x=y",
            timeout_sec=7.0,
            connect_timeout_sec=2.0,
            max_redirects=3,
            on_redirect=redirect,
            on_response=ANY,
            allow_private_hosts=True,
            session=session,
            cookies_for_host=ANY,
        )

    def test_query_merging_json_body_and_raw_headers(
        self,
        session: FetchSession,
    ) -> None:
        backend = Mock(return_value=b"ok")
        _send(
            "https://e.example/p?a=1#s",
            _curl(
                content=ContentParams(method="POST", params={"q": "x"}, json={"a": 1}),
            ),
            transport=backend,
            session=session,
        )
        _send(
            "https://e.example/p",
            _curl(
                content=ContentParams(method="POST", json={"a": 1}, raw_headers=True),
            ),
            transport=backend,
            session=session,
        )
        first, second = backend.call_args_list
        assert first.args[0] == "https://e.example/p?a=1&q=x#s"
        assert first.kwargs["body"] == b'{"a": 1}'
        assert first.kwargs["headers"] == {"Content-Type": "application/json"}
        assert second.kwargs["body"] == b'{"a": 1}'
        assert second.kwargs["headers"] == {}

    def test_cookie_precedence_and_cross_host_lookup(
        self,
        session: FetchSession,
    ) -> None:
        session.profiles.save(
            _EGRESS,
            "x.example",
            Profile(cookies={"GSP": "s", "A": "p", "K": "p"}),
        )
        session.profiles.save(_EGRESS, "b.example", Profile(cookies={"B": "2"}))
        backend = Mock(return_value=b"ok")
        _send(
            "https://x.example:8443/p",
            _stdlib(
                content=ContentParams(
                    headers={"cookie": "GSP=h; A=h", "COOKIE": "GSP=h2; A=h2"},
                    cookies={"A": "param"},
                ),
            ),
            transport=backend,
            session=session,
        )
        # Raw headers ignore stored cookies both initially and on redirect lookup.
        _send(
            "https://x.example/p",
            _stdlib(content=ContentParams(raw_headers=True)),
            transport=backend,
            session=session,
        )
        # Relative URL has no hostname, so no profile is loaded.
        _send("/relative", _stdlib(), transport=backend, session=session)

        first, raw_call, rel_call = backend.call_args_list
        assert first.kwargs["headers"] == {"Cookie": "GSP=h2; A=param; K=p"}
        cookies_for_host = cast(
            Callable[[str], Mapping[str, str]],
            first.kwargs["cookies_for_host"],
        )
        assert dict(cookies_for_host("b.example")) == {"B": "2"}
        assert dict(cookies_for_host("unknown.example")) == {}

        raw_lookup = cast(
            Callable[[str], Mapping[str, str]],
            raw_call.kwargs["cookies_for_host"],
        )
        assert raw_call.kwargs["headers"] == {}
        assert dict(raw_lookup("b.example")) == {}
        assert rel_call.kwargs["headers"] == {}
        assert rel_call.kwargs["body"] == b""

    def test_on_response_persists_per_hop_cookies_and_tolerates_oserror(
        self,
        session: FetchSession,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        session.profiles.save(
            _EGRESS,
            "existing.example",
            Profile(cookies={"old": "1"}),
        )
        seen: list[tuple[int, dict[str, str]]] = []

        def transport(*_args: object, **kwargs: object) -> bytes:
            on_response = cast(
                Callable[[int, dict[str, str], str], None],
                kwargs["on_response"],
            )
            on_response(
                302,
                {"set-cookie": "new=2\nother=3"},
                "https://existing.example:8443/hop",
            )
            on_response(200, {"set-cookie": "fresh=1"}, "https://new.example/land")
            on_response(200, {"set-cookie": "ignored=1"}, "https:///no-host")
            on_response(200, {"x-test": "1"}, "https://none.example/")
            return b"done"

        with patch.object(zendriver, "fetch_zendriver", side_effect=transport):
            assert (
                fetch.fetch(
                    "https://start.example/",
                    request=_zendriver(
                        observe=ObserveParams(
                            on_response=lambda s, h: seen.append((s, h)),
                        ),
                    ),
                    session=session,
                )
                == b"done"
            )
        assert seen == [
            (302, {"set-cookie": "new=2\nother=3"}),
            (200, {"set-cookie": "fresh=1"}),
            (200, {"set-cookie": "ignored=1"}),
            (200, {"x-test": "1"}),
        ]
        existing = session.profiles.load(_EGRESS, "existing.example")
        created = session.profiles.load(_EGRESS, "new.example")
        assert existing is not None
        assert existing.cookies == {"old": "1", "new": "2", "other": "3"}
        assert created is not None
        assert created.cookies == {"fresh": "1"}
        assert session.profiles.load(_EGRESS, "start.example") is None
        assert session.profiles.load(_EGRESS, "") is None

        # Without egress or with raw_headers, nothing is saved; OSError logs warning.
        no_egress = FetchSession()
        try:
            backend = _reports(200, {"set-cookie": "a=1"}, "https://x.example/")
            _send("https://x.example/", _curl(), transport=backend, session=no_egress)
            _send(
                "https://x.example/",
                _curl(content=ContentParams(raw_headers=True)),
                transport=backend,
                session=session,
            )
            assert no_egress.profiles.load("", "x.example") is None
        finally:
            no_egress.close()
        assert session.profiles.load(_EGRESS, "x.example") is None

        with (
            patch.object(session.profiles, "save", side_effect=OSError("disk full")),
            caplog.at_level(logging.WARNING, logger="wesearch.fetch.fetch"),
        ):
            assert (
                _send(
                    "https://jar.example/",
                    _curl(),
                    transport=_reports(
                        200,
                        {"set-cookie": "a=1"},
                        "https://jar.example/",
                    ),
                    session=session,
                )
                == b"ok"
            )
        assert [(r.getMessage(), bool(r.exc_info)) for r in caplog.records] == [
            ("could not save cookies for jar.example", True),
        ]

    @pytest.mark.parametrize(
        ("status", "retried"),
        [
            (0, True),
            (429, True),
            (500, True),
            (502, True),
            (503, True),
            (504, True),
            (403, False),
            (404, False),
            (430, False),
            (501, False),
            (505, False),
        ],
    )
    def test_only_transient_statuses_retry(
        self,
        session: FetchSession,
        status: int,
        retried: bool,
    ) -> None:
        backend = Mock(side_effect=[FetchError("u", status, {}, b""), b"ok"])
        with patch("wesearch.fetch.fetch.time.sleep"):
            if retried:
                assert (
                    _send(
                        "https://example.com",
                        _curl(retry=RetryParams(retries=1)),
                        transport=backend,
                        session=session,
                    )
                    == b"ok"
                )
            else:
                with pytest.raises(FetchError):
                    _send(
                        "https://example.com",
                        _curl(retry=RetryParams(retries=1)),
                        transport=backend,
                        session=session,
                    )
        assert backend.call_count == 1 + retried

    def test_retry_budget_backoff_headers_and_debug_log(
        self,
        session: FetchSession,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        errors = [
            FetchError("https://x.example/", 0, {}, b"down"),
            FetchError("https://x.example/", 429, {"retry-after": "3"}, b"slow"),
            FetchError("https://x.example/", 500, {}, b"err"),
        ]
        backend = Mock(side_effect=[*errors, b"ok"])
        sleeps = Mock()
        caplog.set_level("DEBUG", logger="wesearch.fetch.fetch")
        with (
            patch("wesearch.fetch.fetch.time.sleep", sleeps),
            patch("wesearch.types.params.random.uniform", return_value=0.0),
        ):
            assert (
                _send(
                    "https://x.example/",
                    _curl(retry=RetryParams(retries=3)),
                    transport=backend,
                    session=session,
                )
                == b"ok"
            )
        assert backend.call_count == 4
        assert sleeps.call_args_list == [call(1.0), call(3.0), call(4.0)]
        assert [r.getMessage() for r in caplog.records] == [
            "fetch https://x.example/ -> 0, retry in 1.0s",
            "fetch https://x.example/ -> 429, retry in 3.0s",
            "fetch https://x.example/ -> 500, retry in 4.0s",
        ]

        # Exhausting retry budget raises the last FetchError.
        exhausted = Mock(side_effect=[errors[0], errors[2]])
        with (
            patch("wesearch.fetch.fetch.time.sleep"),
            pytest.raises(FetchError) as raised,
        ):
            _send(
                "https://x.example/",
                _curl(retry=RetryParams(retries=1)),
                transport=exhausted,
                session=session,
            )
        assert raised.value is errors[2]
        assert exhausted.call_count == 2


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


class _Counted:
    """A connection and a browser pool that count their closes."""

    def __init__(self) -> None:
        self.closes = 0
        self.shutdowns = 0

    def close(self) -> None:
        self.closes += 1

    def shutdown(self) -> None:
        self.shutdowns += 1


def _sent_headers(mock: Mock, index: int = 0) -> dict[str, str]:
    """Return the headers of a recorded transport call, ``None`` values dropped."""
    headers = cast(
        dict[str, str | None],
        mock.call_args_list[index].kwargs["headers"],
    )
    return {k: v for k, v in headers.items() if v is not None}


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


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
