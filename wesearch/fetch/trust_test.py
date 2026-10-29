"""Cross-transport tests for the ``allow_private_hosts`` contract.

The bugs this file pins all lived in one blind spot: every other test fixes
one axis and varies the other, so nothing covered transport x cookies x
private-host policy together. A grid does. Each cell must behave identically --
if a transport needs a special case here, the abstraction is still wrong.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import TYPE_CHECKING, cast, override
from unittest.mock import Mock, patch

import inspect

from curl_cffi import requests as cc_requests
from curl_cffi.requests import Headers

import pytest

from wesearch.fetch import FetchSession, fetch
from wesearch.fetch.transport import curl, stdlib, transport_routing, zendriver
from wesearch.fetch.types import Fetch
from wesearch.profile import Profile
from wesearch.types.errors import CloudflareChallengeError
from wesearch.types.params import PolicyParams, RequestParams


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from wesearch.types.params import Transport


# A public egress, so ``fetch`` never probes for one.
_EGRESS = "203.0.113.1"


class _Echo(BaseHTTPRequestHandler):
    """Answers every GET with ``ok``."""

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    @override
    def log_message(self, format: str, *args: object) -> None:
        del format, args


@pytest.fixture(name="session")
def session_fixture() -> Iterator[FetchSession]:
    """Return a session with a known egress and a stored ``example.com`` cookie."""
    session = FetchSession(egress=_EGRESS)
    session.profiles.save(_EGRESS, "example.com", Profile(cookies={"stored": "S"}))
    yield session
    session.close()


class TestCookiesSurviveThePolicy:
    """A stored cookie reaches the wire under every transport and host policy.

    The policy is about security, not transport capability: choosing it must not
    cost the session identity. The pinned curl fork once dropped the cookie jar,
    so every public-only fetch presented as a brand-new anonymous client -- the
    bot signature profiles exist to prevent.
    """

    @pytest.mark.parametrize("transport", ["curl", "stdlib"])
    @pytest.mark.parametrize("allow_private_hosts", [False, True])
    def test_stored_cookie_reaches_wire(
        self,
        session: FetchSession,
        transport: Transport,
        allow_private_hosts: bool,
    ) -> None:
        backend = Mock(return_value=b"ok")
        with patch(f"wesearch.fetch.fetch.fetch_{transport}", backend):
            fetch(
                "https://example.com/",
                request=RequestParams(
                    policy=PolicyParams(
                        transport=transport,
                        allow_private_hosts=allow_private_hosts,
                    ),
                ),
                session=session,
            )
        backend.assert_called_once()
        assert backend.call_args.kwargs["headers"] == {"Cookie": "stored=S"}
        assert backend.call_args.kwargs["allow_private_hosts"] is allow_private_hosts


class TestBrowserUnderPublicOnly:
    """Browser transports run under the default public-only policy.

    Chrome owns its DNS, so the policy checks the host and declines to pin.
    Rejecting the request instead left the one caller that asked for safety with
    no browser path at all, and silently downgraded its ``auto`` to plain curl.
    """

    @pytest.mark.parametrize("transport", ["zendriver", "curl-then-zendriver"])
    def test_browser_transport_reaches_chrome(
        self,
        session: FetchSession,
        transport: Transport,
    ) -> None:
        browser = Mock(return_value=b"<html>ok</html>")
        with (
            patch.object(zendriver, "fetch_zendriver", browser),
            patch("wesearch.fetch.fetch.fetch_curl", _walled),
        ):
            body = fetch(
                "https://example.com/",
                request=RequestParams(policy=PolicyParams(transport=transport)),
                session=session,
            )
        assert body == b"<html>ok</html>"
        browser.assert_called_once()
        assert browser.call_args.kwargs["allow_private_hosts"] is False

    def test_auto_keeps_browser_leg_under_public_only(
        self,
        session: FetchSession,
    ) -> None:
        # ``auto`` once short-circuited to plain curl whenever private hosts were
        # refused, so the browser fallback was unreachable for the default caller.
        session.profiles.discard(_EGRESS, "example.com")
        curl_leg = Mock(side_effect=_walled)
        browser = Mock(return_value=b"rendered")
        with (
            patch("wesearch.fetch.fetch.fetch_curl", curl_leg),
            patch.object(zendriver, "fetch_zendriver", browser),
            patch.object(transport_routing, "remember_zendriver_domain"),
        ):
            assert fetch("https://example.com/", session=session) == b"rendered"
        curl_leg.assert_called_once()
        browser.assert_called_once()

    def test_learned_domain_routes_to_browser_under_public_only(
        self,
        session: FetchSession,
    ) -> None:
        curl_leg = Mock(return_value=b"curl")
        with (
            patch.object(
                transport_routing,
                "zendriver_domains",
                return_value=frozenset({"learned.example"}),
            ),
            patch("wesearch.fetch.fetch.fetch_curl", curl_leg),
            patch.object(zendriver, "fetch_zendriver", return_value=b"rendered"),
        ):
            assert fetch("https://learned.example/", session=session) == b"rendered"
        curl_leg.assert_not_called()


class TestPrivateHostEnforcement:
    """The default refuses a private address; ``allow_private_hosts`` permits it."""

    @pytest.mark.parametrize("transport", ["curl", "stdlib", "zendriver"])
    def test_default_refuses_loopback(
        self,
        session: FetchSession,
        transport: Transport,
    ) -> None:
        with pytest.raises(ValueError, match="non-public"):
            fetch(
                "http://127.0.0.1:1/",
                request=RequestParams(policy=PolicyParams(transport=transport)),
                session=session,
            )
        assert session.curl == {}
        assert session.browsers is None

    def test_allowing_private_hosts_permits_loopback(
        self,
        session: FetchSession,
    ) -> None:
        server = HTTPServer(("127.0.0.1", 0), _Echo)
        port = server.server_address[1]
        thread = Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        thread.start()
        try:
            body = fetch(
                f"http://127.0.0.1:{port}/",
                request=RequestParams(
                    policy=PolicyParams(transport="curl", allow_private_hosts=True),
                ),
                session=session,
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()
        assert body == b"ok"


class TestOneTransportContract:
    """One curl implementation, and one signature every transport shares.

    The pinned fork diverged twice from the pooled path -- headers, then cookies
    -- because a security option selected an implementation rather than setting
    an option. ``Session(curl_options={CurlOpt.RESOLVE: ...})`` pins the pooled
    Session, so the fork has no reason to exist.
    """

    def test_no_pinned_curl_function(self) -> None:
        assert not hasattr(curl, "_fetch_curl_pinned")

    def test_no_simple_curl_function(self) -> None:
        assert not hasattr(curl, "_fetch_curl_simple")

    @pytest.mark.parametrize(
        "transport",
        [curl.fetch_curl, stdlib.fetch_stdlib, zendriver.fetch_zendriver],
        ids=["curl", "stdlib", "zendriver"],
    )
    def test_every_transport_has_the_fetch_signature(
        self,
        transport: Callable[..., bytes],
    ) -> None:
        # The dispatcher picks any backend by name, so a divergence here is what
        # let the pinned path quietly stop accepting a pooled session.
        protocol = inspect.signature(Fetch.__call__)
        expected = protocol.replace(
            parameters=[p for p in protocol.parameters.values() if p.name != "self"],
        )
        assert inspect.signature(transport) == expected


class TestCurlConnectionReuse:
    """A session reuses one pinned curl connection per key, and it holds no jar.

    The key is ``(egress, host, pin, port)``. A connection opened per request
    re-handshakes TLS every time, which no browser does; a connection that kept
    its own cookie jar would resend a burned identity's cookies after the
    profile was discarded.
    """

    def test_fetches_to_one_host_share_one_connection(
        self,
        session: FetchSession,
    ) -> None:
        with patch.object(
            cc_requests.Session,
            "request",
            Mock(side_effect=[_response(), _response()]),
        ) as request:
            fetch("https://example.com/a", request=_curl(), session=session)
            opened = list(session.curl.values())
            fetch("https://example.com/b", request=_curl(), session=session)
        assert len(opened) == 1
        assert list(session.curl.values()) == opened
        assert [c.kwargs["discard_cookies"] for c in request.call_args_list] == [
            True,
            True,
        ]

    def test_a_burned_identity_sends_no_cookie_on_the_reused_connection(
        self,
        session: FetchSession,
    ) -> None:
        blocked = _response(403, b'<div class="g-recaptcha" data-sitekey="x"></div>')
        with (
            patch.object(
                cc_requests.Session,
                "request",
                Mock(side_effect=[blocked, _response()]),
            ) as request,
            patch("wesearch.fetch.fetch.egress_ip", _known_egress),
        ):
            assert fetch("https://example.com/", session=session) == b"ok"
        assert len(session.curl) == 1
        cookies = [
            cast(dict[str, str | None], c.kwargs["headers"]).get("Cookie")
            for c in request.call_args_list
        ]
        assert cookies == ["stored=S", None]


def _curl() -> RequestParams:
    """Return request params on the curl transport."""
    return RequestParams(policy=PolicyParams(transport="curl"))


def _response(status: int = 200, content: bytes = b"ok") -> Mock:
    """Return a curl_cffi response stand-in with real ``Headers``."""
    return Mock(status_code=status, content=content, headers=Headers([]))


def _walled(*_args: object, **_kwargs: object) -> bytes:
    """Raise the block that sends curl-then-zendriver to the browser."""
    raise CloudflareChallengeError(url="https://example.com/", status=403)


def _known_egress(session: FetchSession, **_kw: object) -> str:
    """Return the session's egress, as a probe that answers the same IP would."""
    return session.egress


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
