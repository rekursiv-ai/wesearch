"""Cross-transport tests for the ``trust`` contract.

The bugs this file pins all lived in one blind spot: every existing test fixes
one axis and varies the other, so nothing covered transport x cookies x SSRF
together. A grid does. Each cell must behave identically -- if a transport
needs a special case here, the abstraction is still wrong.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, override
from unittest.mock import patch

import importlib
import inspect

import pytest

from wesearch.fetch.common import ValidatedHost
from wesearch.fetch.transport.zendriver import BrowserResult
from wesearch.profile import Profile, ProfileStore
from wesearch.types.errors import CloudflareChallengeError
from wesearch.types.params import PolicyParams, RequestParams, Transport, Trust


fetch_mod = importlib.import_module("wesearch.fetch.fetch")
curl_mod = importlib.import_module("wesearch.fetch.transport.curl")
stdlib_mod = importlib.import_module("wesearch.fetch.transport.stdlib")


class _Echo(BaseHTTPRequestHandler):
    """Records the Cookie header of each request."""

    seen: list[str] = []  # noqa: RUF012 -- test-local recorder.

    def do_GET(self) -> None:
        _Echo.seen.append(self.headers.get("Cookie", ""))
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    @override
    def log_message(self, format: str, *args: Any) -> None:
        del format, args


@pytest.fixture(name="profiled")
def profiled_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Pin an egress and seat a stored cookie for ``example.com``."""
    store = ProfileStore(base_dir=tmp_path)

    def shared(_cls: type[ProfileStore]) -> ProfileStore:
        return store

    def fixed_egress(**_kw: object) -> str:
        return "203.0.113.1"

    monkeypatch.setattr(ProfileStore, "shared", classmethod(shared))
    monkeypatch.setattr(fetch_mod, "egress_ip", fixed_egress)
    store.save(
        "203.0.113.1", "example.com", Profile(ua="UA/1", cookies={"stored": "S"})
    )
    return "203.0.113.1"


class TestCookiesSurviveTrust:
    """A stored cookie reaches the wire under every transport and trust level.

    ``trust`` is a security policy, not a transport capability: choosing it
    must not cost the session identity. The SSRF-pinned curl fork dropped the
    pooled Session -- and with it the jar -- so every sagent fetch presented as
    a brand-new anonymous client, the exact bot signature ProfileStore exists
    to prevent.
    """

    @pytest.mark.parametrize("transport", ["curl", "stdlib"])
    @pytest.mark.parametrize("trust", ["untrusted", "internal"])
    def test_stored_cookie_reaches_wire(
        self, profiled: str, transport: Transport, trust: Trust
    ) -> None:
        del profiled
        sent: dict[str, str] = {}

        def spy(url: str, **kwargs: Any) -> bytes:
            del url
            jar = kwargs.get("session")
            header = kwargs["headers"].get("Cookie", "")
            names = [c.name for c in jar.cookies.jar] if jar is not None else []
            sent["cookies"] = header or ",".join(names)
            return b"ok"

        with (
            patch.object(fetch_mod, "fetch_curl", spy),
            patch.object(fetch_mod, "fetch_stdlib", spy),
        ):
            fetch_mod.fetch(
                "https://example.com/",
                request=RequestParams(
                    policy=PolicyParams(transport=transport, trust=trust)
                ),
            )
        assert "stored" in sent["cookies"], (
            f"transport={transport} trust={trust} sent no stored cookie: "
            f"{sent['cookies']!r}"
        )


class TestBrowserUnderUntrusted:
    """Browser transports run under the default trust.

    Chrome owns its DNS, so ``untrusted`` validates the host and declines to
    pin. Rejecting the request instead left sagent -- the one caller that asked
    for SSRF safety -- with no browser path at all, and silently downgraded its
    ``auto`` to plain curl.
    """

    @pytest.mark.parametrize("transport", ["zendriver", "curl-then-zendriver"])
    def test_browser_transport_reaches_chrome(
        self, profiled: str, transport: Transport
    ) -> None:
        del profiled
        reached: list[str] = []

        def browser(url: str, **_kw: Any) -> BrowserResult:
            reached.append(url)
            return BrowserResult(body=b"<html>ok</html>", cookies={})

        def walled(*_a: Any, **_kw: Any) -> bytes:
            # ``zendriver`` must never consult curl at all; ``curl-then-zendriver``
            # consults it first and escalates ONLY on a bot block, so a block is
            # what puts both transports in front of the browser.
            raise CloudflareChallengeError(url="https://example.com/", status=403)

        with (
            patch.object(fetch_mod.zendriver_backend, "fetch_zendriver", browser),
            patch.object(fetch_mod, "fetch_curl", walled),
        ):
            body, _ = fetch_mod.fetch(
                "https://example.com/",
                request=RequestParams(policy=PolicyParams(transport=transport)),
            )
        assert body == b"<html>ok</html>"
        assert reached, f"{transport} never reached the browser under untrusted"

    def test_auto_keeps_browser_leg_under_untrusted(self, profiled: str) -> None:
        # BUG 4: ``auto`` short-circuited to plain curl whenever SSRF was
        # requested, so the learned-domain routing below it never ran and the
        # browser fallback was unreachable for the default caller.
        del profiled
        resolved: list[str] = []

        def spy(request: Any, **_kw: Any) -> bytes:
            resolved.append(request.params.policy.transport)
            return b"ok"

        with patch.object(fetch_mod, "_fetch_with_identity", spy):
            fetch_mod.fetch("https://example.com/", request=RequestParams())
        assert resolved == ["curl-then-zendriver"]

    def test_learned_domain_routes_to_browser_under_untrusted(
        self, profiled: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        del profiled
        monkeypatch.setattr(
            fetch_mod.transport_routing,
            "zendriver_domains",
            lambda: frozenset({"learned.example"}),
        )
        resolved: list[str] = []

        def spy(request: Any, **_kw: Any) -> bytes:
            resolved.append(request.params.policy.transport)
            return b"ok"

        with patch.object(fetch_mod, "_fetch_with_identity", spy):
            fetch_mod.fetch("https://learned.example/", request=RequestParams())
        assert resolved == ["zendriver"]


class TestTrustEnforcement:
    """``untrusted`` refuses a private address; ``internal`` permits it."""

    @pytest.mark.parametrize("transport", ["curl", "stdlib"])
    def test_untrusted_refuses_loopback(self, transport: Transport) -> None:
        with pytest.raises(ValueError, match="non-public"):
            fetch_mod.fetch(
                "http://127.0.0.1:1/",
                request=RequestParams(policy=PolicyParams(transport=transport)),
            )

    def test_internal_permits_loopback(self) -> None:
        _Echo.seen.clear()
        server = HTTPServer(("127.0.0.1", 0), _Echo)
        port = server.server_address[1]
        thread = Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        thread.start()
        try:
            body, _ = fetch_mod.fetch(
                f"http://127.0.0.1:{port}/",
                request=RequestParams(
                    policy=PolicyParams(transport="curl", trust="internal")
                ),
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
        assert body == b"ok"


class TestPinnedForkIsGone:
    """One curl implementation, not two.

    The fork diverged twice from the pooled path -- headers in July, cookies
    after -- because ``validated_hosts`` selected an implementation rather than
    setting an option. ``Session(curl_options={CurlOpt.RESOLVE: ...})`` pins on
    the pooled Session (measured: thread-safe and concurrency-safe), so the
    fork has no reason to exist.
    """

    def test_no_pinned_curl_function(self) -> None:
        assert not hasattr(curl_mod, "_fetch_curl_pinned")

    def test_no_simple_curl_function(self) -> None:
        assert not hasattr(curl_mod, "_fetch_curl_simple")

    def test_fetch_curl_takes_no_validated_hosts(self) -> None:
        assert (
            "validated_hosts" not in inspect.signature(curl_mod.fetch_curl).parameters
        )

    def test_transports_keep_identical_signatures(self) -> None:
        # The dispatcher picks either backend by name, so a divergence here is
        # what let the pinned path quietly stop accepting a pooled session.
        curl = set(inspect.signature(curl_mod.fetch_curl).parameters)
        stdlib = set(inspect.signature(stdlib_mod.fetch_stdlib).parameters)
        assert curl == stdlib


class TestBurnDropsEveryPinnedSession:
    """A burned identity loses ALL its pooled sessions, pinned or not.

    The pin is part of the pool key, so a burn that matched one exact key would
    leave the same identity's other pinned sessions alive -- still holding the
    cookies that just got it blocked.
    """

    def test_close_drops_all_pins_for_the_identity(self) -> None:
        closed: list[str] = []

        class _Session:
            def __init__(self, label: str) -> None:
                self.label = label

            def close(self) -> None:
                closed.append(self.label)

        pool = {
            ("1.2.3.4", "example.com", "chrome", None, 443): _Session("unpinned"),
            (
                "1.2.3.4",
                "example.com",
                "chrome",
                ValidatedHost(host="example.com", ip="93.184.216.34"),
                443,
            ): _Session("pinned"),
            ("5.6.7.8", "example.com", "chrome", None, 443): _Session("other-egress"),
        }
        with patch.object(curl_mod, "_curl_sessions", pool):
            curl_mod.close_curl_session("1.2.3.4", "example.com", "chrome")
            survivors = list(pool)
        assert sorted(closed) == ["pinned", "unpinned"]
        assert survivors == [("5.6.7.8", "example.com", "chrome", None, 443)]


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
