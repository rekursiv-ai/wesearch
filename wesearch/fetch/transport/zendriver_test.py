"""Tests for ``wesearch.fetch.transport.zendriver`` (zendriver headless fetch backend).

Hermetic: a fake async browser stands in for zendriver, so the transport logic
(page status, cookie-domain filtering, challenge settling, redirect guard, pool
reuse) is exercised with no Chrome and no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import GeneratorType, MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, Final, cast, override

import asyncio
import contextlib
import inspect
import os
import re
import selectors
import socket
import subprocess
import sys
import threading
import time
import warnings

from treekle import from_plain
from zendriver import Browser, Config, Tab
from zendriver.cdp import fetch, network, page, security, target
from zendriver.core.connection import Transaction

import asyncio_atexit
import pytest

from wesearch.fetch.session import default_session
from wesearch.fetch.transport import zendriver
from wesearch.fetch.transport.zendriver import BrowserPool, _load
from wesearch.fetch.types import FetchSession
from wesearch.lib.userdirs import data_dir
from wesearch.types.errors import CloudflareChallengeError, FetchError
from wesearch.types.params import ignore


if TYPE_CHECKING:
    from collections.abc import (
        Awaitable,
        Callable,
        Coroutine,
        Generator,
        Mapping,
        Sequence,
    )

    from wesearch.fetch.types import Fetch
    from wesearch.types.params import HttpMethod


_CWD: Final = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def _stub_browser_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep fake-browser tests independent of installed Chrome binaries."""

    # Never executed: every test fakes the launch, so nothing needs to exist here.
    def find_fake_browser(browser: object) -> str:
        del browser
        return "/fake/chrome"

    # A string target: `zendriver` names this package's transport module here.
    monkeypatch.setattr("zendriver.core.config.find_executable", find_fake_browser)


# A fake profile dir; the browser is mocked in every test, so it is never
# touched on disk.
_PROFILE = Path("test-profile")

# Captured at IMPORT, which is the only moment it is still reachable: arming is
# class-wide and permanent, and earlier tests in this module launch browsers, so
# a test that read ``Transaction.__call__`` at call time would restore the guard
# it means to remove and assert nothing.
_VENDOR_TRANSACTION_CALL = Transaction[object].__call__


@pytest.mark.cli_python_subprocess
def test_direct_executable_reexecutes_as_module() -> None:
    script = _CWD / "zendriver.py"
    result = subprocess.run(  # noqa: S603 -- fixed argv: this repo's own script.
        ["/bin/sh", "-x", str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "python3 -m wesearch.fetch.transport.zendriver --help" in result.stderr
    assert "RuntimeWarning" not in result.stderr


@dataclass(slots=True, kw_only=True)
class _FakeCookie:
    name: str
    value: str
    domain: str


class _FakeCookieJar:
    def __init__(self, cookies: list[_FakeCookie]) -> None:
        self._cookies = cookies
        self.seeded: list[network.CookieParam] = []

    async def get_all(self) -> list[_FakeCookie]:
        return self._cookies

    async def set_all(self, cookies: list[network.CookieParam]) -> None:
        self.seeded = cookies


# The genuine CDP dataclass, not a look-alike: the transport guards on ``isinstance`` (a
# live run delivered a foreign event type to the handler), so a stand-in would satisfy
# the fake and be rejected in production -- the exact direction a test must never fail
# in. Only the two fields the transport reads carry meaning; the rest are the shape the
# class requires.
def _main_frame_navigated() -> page.FrameNavigated:
    """Return a real ``FrameNavigated`` for the MAIN frame (``parent_id is None``)."""
    frame = page.Frame(
        id_=page.FrameId("main"),
        loader_id=network.LoaderId("loader"),
        url="https://walled.example/",
        domain_and_registry="walled.example",
        security_origin="https://walled.example",
        mime_type="text/html",
        secure_context_type=page.SecureContextType.SECURE,
        cross_origin_isolated_context_type=(
            page.CrossOriginIsolatedContextType.NOT_ISOLATED
        ),
        gated_api_features=[],
        parent_id=None,
    )
    return page.FrameNavigated(
        frame=frame,
        type_=page.NavigationType.NAVIGATION,
    )


# The tab's target id; ``_load`` records only page responses for this frame.
_TAB_ID = "tab-1"


def _document_response(
    url: str,
    status: int,
    headers: dict[str, str] | None = None,
    *,
    frame: str = _TAB_ID,
    kind: network.ResourceType = network.ResourceType.DOCUMENT,
) -> network.ResponseReceived:
    """Return a real ``ResponseReceived`` for a page load in ``frame``."""
    return network.ResponseReceived(
        request_id=network.RequestId("r"),
        loader_id=network.LoaderId("l"),
        timestamp=network.MonotonicTime(0.0),
        type_=kind,
        response=network.Response(
            url=url,
            status=status,
            status_text="",
            headers=network.Headers(headers or {}),
            mime_type="text/html",
            charset="utf-8",
            connection_reused=False,
            connection_id=0.0,
            encoded_data_length=0.0,
            security_state=security.SecurityState.SECURE,
        ),
        has_extra_info=False,
        frame_id=page.FrameId(frame),
    )


class _FakeTab:
    """A tab that replays a scripted document sequence, driven by navigations.

    ``documents`` is the measured Cloudflare handoff, one entry per main-frame
    navigation, the last repeating forever. Live capture of one walled URL::

        nav 1:   5516 bytes  challenge  <- the interstitial, fully loaded
        (the interstitial's JS navigates)
        nav 2: 380404 bytes  clear      <- the real page

    The 386-byte ``readyState == "loading"`` phase between them is modelled by
    ``parsing``: the body a read catches when it lands after the navigation
    commits but before the document parses. A transport that harvests there
    returns a ``<head>`` with the right title and no body, so the fake must be
    able to hand that out or no test can catch it.
    """

    def __init__(
        self,
        *,
        content: str,
        href: str,
        documents: list[str] | None = None,
        parsing: str | None = None,
        paused_events: list[object] | None = None,
        responses: list[object] | None = None,
    ) -> None:
        self._documents = documents if documents is not None else [content]
        self._parsing = parsing
        self._href = href
        self.target_id = _TAB_ID
        self.closed = False
        self.navigations: list[str] = []
        self.commands: list[object] = []
        self.handlers: list[Callable[[object], None]] = []
        self.paused_handlers: list[Callable[[object], None]] = []
        self.response_handlers: list[Callable[[object], None]] = []
        self.continued_requests: list[str] = []
        self.failed_requests: list[str] = []
        self._paused_events: list[object] = list(paused_events or [])
        self._responses: list[object] = list(responses or [])
        self.wire_commands: list[dict[str, object]] = []
        self.content_reads = 0
        self._index = 0
        # True between a navigation and its ready-state wait -- the window in
        # which the new document exists but has not parsed.
        self._is_parsing = False

    def add_handler(
        self,
        event_type: object,
        handler: Callable[[object], None],
    ) -> None:
        # Routed by event type, mirroring zendriver's own dispatch: the
        # navigation watcher and the per-request guard must not receive each
        # other's events, which a single handler list cannot express.
        if event_type is fetch.RequestPaused:
            self.paused_handlers.append(handler)
            return
        if event_type is network.ResponseReceived:
            self.response_handlers.append(handler)
            return
        self.handlers.append(handler)

    def pause_request(self, event: object) -> None:
        """Deliver one intercepted request to the transport's guard."""
        for handler in self.paused_handlers:
            handler(event)

    def _navigate_main_frame(self) -> None:
        """Advance to the next document and notify the transport's handler."""
        if self._index + 1 < len(self._documents):
            self._index += 1
        self._is_parsing = self._parsing is not None
        for handler in self.handlers:
            handler(_main_frame_navigated())

    async def send(self, command: object) -> None:
        self.commands.append(command)
        # A CDP verb is a generator yielding its wire dict; reading that dict is
        # how the fake records the REAL decision (method + request id) rather
        # than trusting the transport's own account of what it sent.
        raw = next(command, None) if isinstance(command, GeneratorType) else None
        if not isinstance(raw, dict):
            return
        # ``read`` rather than a bare ``.get`` ladder: the CDP verbs are
        # unstubbed, so their wire dict arrives as ``dict[Unknown, Unknown]``
        # and every read off it is partially unknown.
        payload = from_plain(cast(object, raw), dict[str, object])
        # Kept: a generator is single-use, so a test that re-reads ``commands``
        # would find every one exhausted by this very inspection.
        self.wire_commands.append(payload)
        request_id = from_plain(
            from_plain(payload.get("params"), dict[str, object], default={}).get(
                "requestId",
            ),
            str,
            default="",
        )
        if not request_id:
            return
        method = from_plain(payload.get("method"), str, default="")
        if method == "Fetch.continueRequest":
            self.continued_requests.append(request_id)
        elif method == "Fetch.failRequest":
            self.failed_requests.append(request_id)

    async def get(self, url: str) -> _FakeTab:
        self.navigations.append(url)
        if not self._href:
            self._href = url
        # Chrome pauses intercepted requests DURING navigation, so the scripted
        # ones are delivered here rather than after the fetch returns -- the
        # transport answers them on the live loop, which is the only place a
        # CDP reply can be sent.
        for event in self._paused_events:
            self.pause_request(event)
        self._paused_events = []
        for event in self._responses:
            for handler in self.response_handlers:
                handler(event)
        self._responses = []
        # Drained by yielding, never by sleeping: the guard's reply crosses two
        # scheduler stages (``call_soon_threadsafe``, then the task it creates),
        # and these tests run on a 0.02s budget to prove the settle poll gives
        # up -- a wall-clock wait spends half of it and times the fetch out.
        for _ in range(4):
            await asyncio.sleep(0)
        return self

    async def wait_for_ready_state(
        self,
        until: str = "interactive",
        timeout: int = 10,  # noqa: ASYNC109 -- mirrors zendriver's Tab API.
    ) -> bool:
        del until, timeout
        self._is_parsing = False  # Parsing finished; the full document is up.
        return True

    async def evaluate(self, expr: str) -> str:
        del expr
        return self._href

    async def get_content(self) -> str:
        self.content_reads += 1
        if self._is_parsing and self._parsing is not None:
            return self._parsing
        body = self._documents[self._index]
        # A challenge document replaces itself: schedule the handoff the way
        # Chrome does, right after the wall has been observed once.
        if self._index + 1 < len(self._documents):
            self._navigate_main_frame()
        return body

    async def close(self) -> None:
        self.closed = True


class _FakeListener:
    """Stands in for zendriver's websocket listener task."""

    def __init__(self) -> None:
        self.running = True


class _FakeConnection:
    """Stands in for zendriver's browser-level websocket connection."""

    def __init__(self) -> None:
        self.closed = False
        self.listener: _FakeListener | None = _FakeListener()
        self.handlers: dict[type[object], list[Callable[..., object]]] = {}

    def add_handler(
        self,
        event_type: type[object],
        handler: Callable[..., object],
    ) -> None:
        self.handlers.setdefault(event_type, []).append(handler)

    async def emit(self, event: object) -> None:
        """Deliver ``event`` to its handlers as zendriver's listener does."""
        for handler in self.handlers.get(type(event), []):
            result = handler(event, self)
            if inspect.isawaitable(result):
                await result

    async def aclose(self) -> None:
        self.closed = True


class _FakeBrowser:
    """An async stand-in for ``zendriver.Browser`` with scripted content."""

    def __init__(
        self,
        *,
        content: str = "<html>ok</html>",
        href: str = "",
        cookies: list[_FakeCookie] | None = None,
        documents: list[str] | None = None,
        parsing: str | None = None,
        paused_events: list[object] | None = None,
        responses: list[object] | None = None,
    ) -> None:
        self._content = content
        self._href = href
        self._documents = documents
        self._parsing = parsing
        self._paused_events = paused_events or []
        self._responses = responses or []
        self.cookies = _FakeCookieJar(cookies or [])
        self.connection = _FakeConnection()
        self.stopped = False
        self.gets: list[str] = []
        self.stop_calls: int = 0
        self.last_tab: _FakeTab | None = None
        # Mirrors zendriver's ``Popen`` handle on a launched browser; ``None``
        # until a test supplies one, as it is on a browser we never launched.
        self._process: _FakeProcess | None = None

    async def get(self, url: str, new_tab: bool = False) -> _FakeTab:
        del new_tab
        self.gets.append(url)
        self.last_tab = _FakeTab(
            content=self._content,
            href=self._href,
            documents=self._documents,
            parsing=self._parsing,
            # COPIES: the tab drains its lists once delivered.
            paused_events=list(self._paused_events),
            responses=list(self._responses),
        )
        return self.last_tab

    async def stop(self) -> None:
        self.stop_calls += 1
        self.stopped = True


class _FakeProcess:
    """Stands in for zendriver's ``Popen`` handle on a launched browser."""

    def __init__(self) -> None:
        self.kills = 0

    def kill(self) -> None:
        self.kills += 1


class _StubPool:
    """A pool whose ``browser`` always yields one preset fake browser."""

    def __init__(self, browser: _FakeBrowser) -> None:
        self._browser = browser
        self.requested: list[tuple[Path, bool]] = []

    async def browser(
        self,
        profile_dir: Path,
        *,
        headless: bool,
    ) -> _FakeBrowser:
        self.requested.append((profile_dir, headless))
        return self._browser


class _RunningPool(_StubPool):
    """A :class:`_StubPool` that also runs a coroutine and shuts down, as the real one."""

    def __init__(self, browser: _FakeBrowser) -> None:
        super().__init__(browser)
        self.waits: list[float] = []
        self.shutdowns = 0

    def run[T](self, coro: Coroutine[object, object, T], *, timeout_sec: float) -> T:
        self.waits.append(timeout_sec)
        return asyncio.run(coro)

    def shutdown(self) -> None:
        self.shutdowns += 1


def _session(pool: object) -> FetchSession:
    """Return a session whose browser pool is ``pool``."""
    return FetchSession(browsers=cast(BrowserPool, pool), profile_dir=_PROFILE)


async def _drive(
    browser: _FakeBrowser,
    url: str = "https://example.com/",
    *,
    timeout_sec: float = 5.0,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    max_redirects: int = 10,
    on_redirect: Callable[[str], None] = ignore,
    on_response: Callable[[int, dict[str, str], str], None] = ignore,
    allow_private_hosts: bool = False,
) -> bytes:
    """Run :func:`_load` on ``browser`` through a stub pool."""
    return await _load(
        cast(BrowserPool, _StubPool(browser)),
        url,
        profile_dir=_PROFILE,
        headless=True,
        headers=headers or {},
        cookies=cookies or {},
        timeout_sec=timeout_sec,
        max_redirects=max_redirects,
        on_redirect=on_redirect,
        on_response=on_response,
        allow_private_hosts=allow_private_hosts,
    )


def _load_page(
    browser: _FakeBrowser,
    url: str = "https://example.com/",
    *,
    timeout_sec: float = 5.0,
    cookies: dict[str, str] | None = None,
    on_response: Callable[[int, dict[str, str], str], None] = ignore,
) -> bytes:
    """Run :func:`_drive` to completion."""
    return asyncio.run(
        _drive(
            browser,
            url,
            timeout_sec=timeout_sec,
            cookies=cookies,
            on_response=on_response,
        ),
    )


# -- fetch_zendriver: the Fetch contract -------------------------------------


def _send(
    browser: _FakeBrowser,
    *,
    headers: Mapping[str, str | None] = MappingProxyType({}),
    on_response: Callable[[int, dict[str, str], str], None] = ignore,
    url: str = "https://a.example/",
) -> bytes:
    """Call ``fetch_zendriver`` as a :class:`Fetch` on ``browser``."""
    transport: Fetch = zendriver.fetch_zendriver
    return transport(
        url,
        headers=headers,
        timeout_sec=5.0,
        on_response=on_response,
        session=_session(_RunningPool(browser)),
    )


@pytest.fixture
def pooled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let ``fetch_zendriver`` accept a :class:`_RunningPool` as its pool."""
    monkeypatch.setattr(zendriver, "BrowserPool", _RunningPool)


@pytest.mark.usefixtures("pooled")
def test_fetch_zendriver_returns_the_rendered_body() -> None:
    assert _send(_FakeBrowser(content="<html>page</html>")) == b"<html>page</html>"


@pytest.mark.usefixtures("pooled")
def test_fetch_zendriver_reports_landed_cookies_like_a_response() -> None:
    browser = _FakeBrowser(
        href="https://b.example/landing",
        cookies=[_FakeCookie(name="SID", value="1", domain="b.example")],
    )
    seen: list[tuple[int, dict[str, str], str]] = []

    def observe(status: int, headers: dict[str, str], url: str) -> None:
        seen.append((status, headers, url))

    _send(browser, on_response=observe, url="https://a.example/start")
    assert seen == [(200, {"set-cookie": "SID=1"}, "https://b.example/landing")]


@pytest.mark.parametrize(
    ("method", "body"),
    [("POST", b""), ("GET", b"x")],
)
def test_fetch_zendriver_refuses_what_a_navigation_cannot_send(
    method: str,
    body: bytes,
) -> None:
    with pytest.raises(ValueError, match="only a bodyless GET"):
        zendriver.fetch_zendriver(
            "https://a.example/",
            method=cast("HttpMethod", method),
            body=body,
        )


def test_fetch_zendriver_refuses_a_private_host_before_any_browser() -> None:
    session = FetchSession(profile_dir=_PROFILE)
    with pytest.raises(ValueError, match="non-public"):
        zendriver.fetch_zendriver("http://127.0.0.1:1/", session=session)
    assert session.browsers is None


@pytest.mark.usefixtures("pooled")
def test_fetch_zendriver_seeds_the_cookie_header_into_chrome() -> None:
    browser = _FakeBrowser()
    _send(browser, headers={"Cookie": "a=1; b=2", "X-Key": "k"})
    assert {(c.name, c.value, c.url) for c in browser.cookies.seeded} == {
        ("a", "1", "https://a.example/"),
        ("b", "2", "https://a.example/"),
    }


@pytest.mark.usefixtures("pooled")
def test_fetch_zendriver_forwards_caller_headers_except_cookie_and_none() -> None:
    """Caller headers, browser identity included, reach the tab; ``Cookie`` is seeded."""
    browser = _FakeBrowser()
    _send(
        browser,
        headers={
            "X-Test": "yes",
            "User-Agent": "caller",
            "Cookie": "SID=cookie",
            "X-Gone": None,
        },
    )
    assert browser.last_tab is not None
    assert _extra_http_headers(browser.last_tab) == {
        "x-test": "yes",
        "user-agent": "caller",
    }


def test_fetch_zendriver_adopts_the_session_pool_and_bounds_its_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The outer ceiling must exceed ``_load``'s own budget.

    At or below it the outer wait fires first, cancelling a load that was about
    to report a wall and turning it into an opaque timeout.
    """
    monkeypatch.setattr(zendriver, "BrowserPool", _RunningPool)
    pool = _RunningPool(_FakeBrowser())
    session = FetchSession(
        browsers=cast(BrowserPool, pool),
        profile_dir=_PROFILE,
        headless=False,
    )
    zendriver.fetch_zendriver("https://a.example/", timeout_sec=7.0, session=session)

    assert pool.waits == [37.0]
    assert pool.requested == [(_PROFILE, False)]
    assert pool.shutdowns == 0, "a caller's session must stay open"


class _OwnedPool(_RunningPool):
    """A pool ``fetch_zendriver`` builds itself, for a call with no session."""

    made: list[_OwnedPool] = []  # noqa: RUF012 -- per-test record, reset by the test.

    def __init__(self) -> None:
        super().__init__(_FakeBrowser())
        _OwnedPool.made.append(self)


def test_fetch_zendriver_without_a_session_reuses_the_default_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_OwnedPool, "made", [])
    monkeypatch.setattr(zendriver, "BrowserPool", _OwnedPool)
    assert zendriver.fetch_zendriver("https://a.example/") == b"<html>ok</html>"
    assert zendriver.fetch_zendriver("https://b.example/") == b"<html>ok</html>"
    [pool] = _OwnedPool.made
    assert default_session().browsers is pool
    assert pool.shutdowns == 0


def test_fetch_zendriver_reports_a_run_timeout_as_status_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TimingOutPool(_RunningPool):
        @override
        def run[T](
            self,
            coro: Coroutine[object, object, T],
            *,
            timeout_sec: float,
        ) -> T:
            coro.close()
            raise TimeoutError

    monkeypatch.setattr(zendriver, "BrowserPool", _TimingOutPool)
    with pytest.raises(FetchError) as error:
        zendriver.fetch_zendriver(
            "https://a.example/",
            session=_session(_TimingOutPool(_FakeBrowser())),
        )
    assert error.value.status == 0
    assert isinstance(error.value.__cause__, TimeoutError)


def test_fetch_zendriver_defaults_are_the_fetch_contract() -> None:
    parameters = inspect.signature(zendriver.fetch_zendriver).parameters
    assert cast(object, parameters["timeout_sec"].default) == 30.0
    assert cast(object, parameters["max_redirects"].default) == 10
    assert cast(object, parameters["allow_private_hosts"].default) is False
    assert cast(object, parameters["session"].default) is None


# -- _load: status, cookies, body --------------------------------------------


def test_load_raises_on_a_page_status_of_400_or_more() -> None:
    """Chrome renders an error page; the transport must report its status.

    Without it a 403 block page reads as a successful fetch of its own markup.
    """
    browser = _FakeBrowser(
        content="<html>Forbidden</html>",
        responses=[_document_response("https://example.com/", 403)],
    )
    with pytest.raises(FetchError) as error:
        _load_page(browser)
    assert error.value.status == 403
    assert error.value.body == b"<html>Forbidden</html>"


def test_load_classifies_a_challenge_status_page() -> None:
    browser = _FakeBrowser(
        content="<html><title>Just a moment...</title></html>",
        responses=[
            _document_response(
                "https://example.com/",
                403,
                {"cf-mitigated": "challenge"},
            ),
        ],
    )
    with pytest.raises(CloudflareChallengeError):
        _load_page(browser, timeout_sec=0.02)


def test_load_reports_every_page_response_and_the_landed_cookies_last() -> None:
    browser = _FakeBrowser(
        href="https://b.example/landing",
        cookies=[_FakeCookie(name="B", value="2", domain="b.example")],
        responses=[
            _document_response("https://a.example/", 302, {"Location": "/x"}),
            _document_response("https://b.example/landing", 200, {"X-Y": "z"}),
            # Subresources and other frames are not pages.
            _document_response(
                "https://cdn.example/a.js",
                200,
                kind=network.ResourceType.SCRIPT,
            ),
            _document_response("https://ad.example/", 500, frame="iframe"),
        ],
    )
    seen: list[tuple[int, dict[str, str], str]] = []

    def observe(status: int, headers: dict[str, str], url: str) -> None:
        seen.append((status, headers, url))

    _load_page(browser, "https://a.example/", on_response=observe)
    assert seen == [
        (302, {"location": "/x"}, "https://a.example/"),
        (200, {"x-y": "z", "set-cookie": "B=2"}, "https://b.example/landing"),
    ]


def test_load_returns_body_and_reports_only_domain_cookies() -> None:
    browser = _FakeBrowser(
        content="<html>results</html>",
        cookies=[
            _FakeCookie(name="SID", value="abc", domain=".gated.example"),
            _FakeCookie(name="OTHER", value="zzz", domain="example.com"),
            _FakeCookie(name="EMPTY", value="", domain="gated.example"),
            _FakeCookie(name="NODOMAIN", value="v", domain=""),
            _FakeCookie(name="BOUNDARY", value="bad", domain="ated.example"),
        ],
    )
    seen: list[dict[str, str]] = []
    body = _load_page(
        browser,
        "https://gated.example/page?q=x",
        on_response=lambda _status, headers, _url: seen.append(headers),
    )
    assert body == b"<html>results</html>"
    assert seen == [{"set-cookie": "SID=abc\nEMPTY="}]
    # The per-fetch tab is closed after harvest -- the memory-teardown contract
    # (Chrome process stays warm; the scraped page's tab does not).
    assert browser.last_tab is not None
    assert browser.last_tab.closed is True


def test_load_matches_subdomain_and_exact_host_cookies() -> None:
    browser = _FakeBrowser(
        href="https://www.example.com/",
        cookies=[
            _FakeCookie(name="PARENT", value="1", domain=".example.com"),
            _FakeCookie(name="EXACT", value="2", domain="www.example.com"),
            _FakeCookie(name="PREFIX", value="bad", domain="xwww.example.com"),
        ],
    )
    seen: list[dict[str, str]] = []
    _load_page(browser, on_response=lambda _s, headers, _u: seen.append(headers))
    assert seen == [{"set-cookie": "PARENT=1\nEXACT=2"}]


def test_load_reports_no_set_cookie_when_chrome_holds_none() -> None:
    seen: list[dict[str, str]] = []
    _load_page(_FakeBrowser(), on_response=lambda _s, headers, _u: seen.append(headers))
    assert seen == [{}]


def test_load_reports_the_url_its_cookies_belong_to() -> None:
    """The landed response must name the origin its cookies were harvested for.

    Cookies are keyed to the FINAL url, because a cross-origin redirect seats
    the target's. Reporting them under the requested URL filed ``b.example``'s
    session cookie under ``a.example``.
    """
    browser = _FakeBrowser(
        href="https://b.example/landing",
        cookies=[_FakeCookie(name="B_SESSION", value="secret", domain="b.example")],
    )
    seen: list[tuple[str, dict[str, str]]] = []
    _load_page(
        browser,
        "https://a.example/start",
        on_response=lambda _s, headers, url: seen.append((url, headers)),
    )
    assert seen == [("https://b.example/landing", {"set-cookie": "B_SESSION=secret"})]


def test_load_unwraps_chromes_json_viewer() -> None:
    """A JSON response must come back as JSON, not as Chrome's viewer shell.

    ``get_content`` serializes the DOM, and for a non-HTML body Chrome
    SYNTHESIZES a document to display it: the payload is re-wrapped in
    ``<html><head>...</head><body><pre>``. The original bytes are still there,
    inside the ``<pre>``, HTML-escaped.
    """
    payload = '{"query": "x&y", "results": []}'
    browser = _FakeBrowser(
        content=(
            '<html><head><meta name="color-scheme" content="light dark">'
            '<meta charset="utf-8"></head><body>'
            '<pre>{"query": "x&amp;y", "results": []}</pre></body></html>'
        ),
    )
    assert _load_page(browser) == payload.encode()


def test_load_unwraps_the_real_chrome_viewer_markup() -> None:
    """Verbatim shell captured from Chrome, not a hand-written approximation.

    Chrome mounts its JSON formatter in a ``<div>`` AFTER the ``</pre>``, so
    the payload is not the last node in the body.
    """
    payload = '{"query": "opensource", "results": []}'
    browser = _FakeBrowser(
        content=(
            '<html><head><meta name="color-scheme" content="light dark">'
            '<meta charset="utf-8"></head><body>'
            f"<pre>{payload}</pre>"
            '<div class="json-formatter-container"></div>'
            "</body></html>"
        ),
    )
    assert _load_page(browser) == payload.encode()


def test_load_leaves_real_html_alone() -> None:
    """A genuine page containing a ``<pre>`` must not be reduced to it."""
    content = "<html><body><h1>Title</h1><pre>code sample</pre></body></html>"
    assert _load_page(_FakeBrowser(content=content)) == content.encode()


def test_load_seeds_cookies_for_the_requested_url() -> None:
    browser = _FakeBrowser()
    _load_page(browser, "https://google.com/search?q=x", cookies={"CONSENT": "YES+"})
    assert [(c.name, c.value, c.url) for c in browser.cookies.seeded] == [
        ("CONSENT", "YES+", "https://google.com/search?q=x"),
    ]


def test_load_seeds_no_cookies_when_none_given() -> None:
    browser = _FakeBrowser()
    _load_page(browser)
    assert browser.cookies.seeded == []


def test_load_opens_the_agent_tab_before_the_requested_url() -> None:
    url = "https://gated.example/page?q=x"
    browser = _FakeBrowser()
    _load_page(browser, url)
    assert browser.gets == ["about:blank#wesearch-agent"]
    assert browser.last_tab is not None
    assert browser.last_tab.navigations == [url]


def test_load_settles_on_half_the_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A page still walled at half the budget returns its wall, not a timeout."""
    budgets: list[float] = []
    real_wait_for = asyncio.wait_for

    async def record(
        awaitable: Awaitable[object],
        timeout: float | None = None,  # noqa: ASYNC109 -- mirrors asyncio.wait_for.
    ) -> object:
        budgets.append(-1.0 if timeout is None else timeout)
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr(asyncio, "wait_for", record)
    browser = _FakeBrowser(
        documents=["<html><title>Just a moment...</title></html>", "<html>ok</html>"],
    )
    assert _load_page(browser, timeout_sec=8.0) == b"<html>ok</html>"
    assert 3.9 < budgets[0] <= 4.0, budgets


def test_load_timeout_includes_browser_acquisition() -> None:
    class _SlowPool:
        async def browser(self, profile_dir: Path, *, headless: bool) -> _FakeBrowser:
            del profile_dir, headless
            await asyncio.Event().wait()
            raise AssertionError("Browser acquisition escaped the timeout.")

    with pytest.raises(TimeoutError):
        asyncio.run(
            _load(
                cast(BrowserPool, _SlowPool()),
                "https://example.com/",
                profile_dir=_PROFILE,
                headless=True,
                headers={},
                cookies={},
                timeout_sec=0.001,
                max_redirects=10,
                on_redirect=ignore,
                on_response=ignore,
                allow_private_hosts=False,
            ),
        )


class _RaisingCloseTab(_FakeTab):
    """A tab whose ``close`` fails outright rather than hanging."""

    @override
    async def close(self) -> None:
        self.closed = True
        raise RuntimeError("close blew up")


class _RaisingCloseBrowser(_FakeBrowser):
    """A browser handing out :class:`_RaisingCloseTab`."""

    @override
    async def get(self, url: str, new_tab: bool = False) -> _FakeTab:
        del new_tab
        self.gets.append(url)
        self.last_tab = _RaisingCloseTab(
            content="<html><title>Plain</title>ok</html>",
            href="",
        )
        return self.last_tab


def test_a_failing_tab_close_does_not_mask_the_fetch_it_cleans_up_after() -> None:
    """Teardown reports nothing over the caller.

    The tab close runs in ``_load``'s ``finally``, so an exception it let
    escape would replace whatever the fetch was about to return OR raise.
    """
    assert _load_page(_RaisingCloseBrowser()) == b"<html><title>Plain</title>ok</html>"


class _WedgedTab(_FakeTab):
    """A tab whose navigation AND close both never return.

    Models one wedged CDP connection: ``Tab.get`` waits on a load event that
    never arrives, and ``Tab.close`` then awaits a reply with no ceiling.
    """

    @override
    async def get(self, url: str) -> _FakeTab:
        self.navigations.append(url)
        await asyncio.Event().wait()
        raise AssertionError("The navigation was never cancelled.")

    @override
    async def close(self) -> None:
        self.closed = True
        await asyncio.Event().wait()
        raise AssertionError("The close was never bounded.")


class _WedgedBrowser(_FakeBrowser):
    """A browser handing out :class:`_WedgedTab`."""

    @override
    async def get(self, url: str, new_tab: bool = False) -> _FakeTab:
        del new_tab
        self.gets.append(url)
        self.last_tab = _WedgedTab(content="<html>ok</html>", href="")
        return self.last_tab


def test_load_reports_its_own_timeout_when_teardown_also_wedges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled load must still finish, so the caller sees the budget.

    An ``asyncio.timeout`` scope delivers exactly ONE cancellation. When it
    lands on the navigation, the tab close runs with that cancellation already
    spent -- so an unbounded close parks there forever and the coroutine NEVER
    completes. The close's own 5s ceiling is shrunk to keep this fast.
    """
    browser = _WedgedBrowser()
    real_timeout = asyncio.timeout
    budgets: list[float | None] = []

    def timeout(delay: float | None) -> asyncio.Timeout:
        budgets.append(delay)
        return real_timeout(0.01 if delay == 5.0 else delay)

    monkeypatch.setattr(asyncio, "timeout", timeout)

    async def go() -> float:
        before = asyncio.all_tasks()
        started = time.monotonic()
        with pytest.raises(TimeoutError) as error:
            await _drive(browser, timeout_sec=0.01)
        assert isinstance(error.value.__cause__, asyncio.CancelledError)
        assert asyncio.all_tasks() - before == set(), "the close task was not reaped"
        return time.monotonic() - started

    # Outside ``raises``: an unbounded close must fail on the outer deadline.
    elapsed = asyncio.run(real_timeout_wait(go(), 1.0))
    assert elapsed < 0.5, f"teardown outlived its budget: {elapsed:.2f}s"
    assert budgets == [0.01, 5.0]
    assert browser.last_tab is not None
    assert browser.last_tab.navigations == ["https://example.com/"]
    assert browser.last_tab.closed


async def real_timeout_wait[T](coro: Coroutine[object, object, T], delay: float) -> T:
    """Await ``coro`` under ``asyncio.wait_for``, untouched by a patched timeout."""
    return await asyncio.wait_for(coro, timeout=delay)


def test_load_uses_one_overall_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every step wait must shrink against ONE deadline, never restart it.

    A CHALLENGE page, not a clear one: the settle loop returns on its first
    read when the document is already clear, so a clear page never reaches the
    two step waits and the assertion below has nothing to observe.
    """
    browser = _FakeBrowser(
        documents=["<html><title>Just a moment...</title></html>", "<html>ok</html>"],
    )
    budgets: list[float] = []
    real_wait_for = asyncio.wait_for

    async def record_budget(
        awaitable: Awaitable[object],
        timeout: float | None = None,  # noqa: ASYNC109 -- The browser test timeout bounds polling across an awaited operation.
    ) -> object:
        budgets.append(-1.0 if timeout is None else timeout)
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr(asyncio, "wait_for", record_budget)
    _load_page(browser, timeout_sec=5.0)

    assert len(budgets) >= 2, f"the settle path never reached its step waits: {budgets}"
    assert all(timeout >= 0.0 for timeout in budgets), (
        f"a step wait was left unbounded: {budgets}"
    )
    assert budgets == sorted(budgets, reverse=True), (
        f"step waits did not shrink against one deadline: {budgets}"
    )
    assert max(budgets) <= 5.0 / 2, f"a step wait exceeded the settle budget: {budgets}"


def test_load_returns_a_rendered_widget_page_without_classifying_it() -> None:
    body = '<html><div id="cf_chl_widget"></div></html>'
    # Small: this body IS challenge markup, so the settle poll waits out its
    # budget before giving up.
    assert _load_page(_FakeBrowser(content=body), timeout_sec=0.02) == body.encode()


def test_load_waits_out_a_cloudflare_interstitial() -> None:
    # The interstitial reaches readyState "complete" on its own -- it IS a loaded
    # document -- and only then does its JS navigate to the real page (measured
    # live: 5516 bytes at complete, 380404 bytes after the handoff).
    real = "<html><title>Real Page</title>body</html>"
    browser = _FakeBrowser(
        documents=["<html><title>Just a moment...</title></html>", real],
    )
    assert _load_page(browser, "https://walled.example/", timeout_sec=30.0) == (
        real.encode()
    ), "browser transport returned the Cloudflare interstitial"


def test_load_waits_for_the_new_document_to_parse() -> None:
    # A main-frame navigation COMMITS the new document before it parses, so a
    # read right after the event sees a bare <head>: right title, no body.
    real = "<html><title>Real Page</title>the whole body</html>"
    browser = _FakeBrowser(
        documents=["<html><title>Just a moment...</title></html>", real],
        parsing="<html><title>Real Page</title></html>",
    )
    assert _load_page(browser, "https://walled.example/", timeout_sec=30.0) == (
        real.encode()
    ), "harvested the document mid-parse: right title, empty body"


def test_load_returns_promptly_when_no_challenge() -> None:
    # A plain document is harvested on the FIRST read, with no re-poll.
    browser = _FakeBrowser(content="<html><title>Plain</title>ok</html>")
    assert _load_page(browser, timeout_sec=30.0) == (
        b"<html><title>Plain</title>ok</html>"
    )
    assert browser.last_tab is not None
    assert browser.last_tab.content_reads == 1


def test_load_gives_up_on_an_unclearable_challenge() -> None:
    # A challenge that never clears must return what Chrome rendered, never hang.
    walled = "<html><title>Just a moment...</title></html>"
    browser = _FakeBrowser(content=walled)
    assert _load_page(browser, "https://walled.example/", timeout_sec=0.02) == (
        walled.encode()
    )
    assert browser.last_tab is not None
    assert browser.last_tab.closed is True


def test_load_allows_embedded_captcha_on_rendered_page() -> None:
    body = '<html><div class="g-recaptcha"></div></html>'
    assert _load_page(_FakeBrowser(content=body)) == body.encode()


# -- _load page guard: redirect hops, pinning, credentials -------------------


def _continued_headers(tab: _FakeTab) -> dict[str, str]:
    """Return the headers the guard released a continued request with."""
    payload = next(
        (
            c
            for c in tab.wire_commands
            if from_plain(c.get("method"), str, default="") == "Fetch.continueRequest"
        ),
        None,
    )
    assert payload is not None, "the request was never continued"
    entries = from_plain(
        from_plain(payload.get("params"), dict[str, object], default={}).get("headers"),
        list[object],
        default=[],
    )
    return {
        from_plain(
            from_plain(entry, dict[str, object]).get("name"),
            str,
            default="",
        ).lower(): from_plain(
            from_plain(entry, dict[str, object]).get("value"),
            str,
            default="",
        )
        for entry in entries
    }


# Whether an override was sent AT ALL: ``Cookie`` applies intermittently under one and
# no override survives a redirect hop, so sending one is the hazard.
def _continue_override(tab: _FakeTab) -> dict[str, str] | None:
    """Return the header OVERRIDE the guard sent, or ``None`` if it sent none."""
    payload = next(
        (
            c
            for c in tab.wire_commands
            if from_plain(c.get("method"), str, default="") == "Fetch.continueRequest"
        ),
        None,
    )
    assert payload is not None, "the request was never continued"
    params = from_plain(payload.get("params"), dict[str, object], default={})
    if "headers" not in params:
        return None
    return _continued_headers(tab)


def _extra_http_headers(tab: _FakeTab) -> dict[str, str]:
    """Return the headers installed tab-wide via ``setExtraHTTPHeaders``."""
    payload = next(
        (
            c
            for c in tab.wire_commands
            if from_plain(c.get("method"), str, default="")
            == "Network.setExtraHTTPHeaders"
        ),
        None,
    )
    if payload is None:
        return {}
    installed = from_plain(
        from_plain(payload.get("params"), dict[str, object], default={}).get("headers"),
        dict[str, object],
        default={},
    )
    return {name.lower(): from_plain(value, str) for name, value in installed.items()}


# ``headers`` are the ones Chrome reports ALREADY on the paused request, which include
# whatever ``set_extra_http_headers`` installed on the tab.
def _request_paused(
    url: str,
    headers: dict[str, str] | None = None,
    *,
    request_id: str = "req-1",
) -> fetch.RequestPaused:
    """Return a real ``RequestPaused`` for a main-frame document request."""
    request = network.Request(
        url=url,
        method="GET",
        headers=network.Headers(headers or {}),
        initial_priority=network.ResourcePriority.HIGH,
        referrer_policy="no-referrer",
    )
    return fetch.RequestPaused(
        request_id=fetch.RequestId(request_id),
        request=request,
        frame_id=page.FrameId("main"),
        resource_type=network.ResourceType.DOCUMENT,
        response_error_reason=None,
        response_status_code=None,
        response_status_text=None,
        response_headers=None,
        network_id=None,
        redirected_request_id=None,
    )


class TestBrowserHonorsPrivateHostsPerHop:
    """Chrome follows redirects itself, so each hop must be checked before it runs.

    The header transports re-validate every hop; the browser leg once validated
    only the URL the CALLER passed and then handed navigation to Chrome, which
    fetched every subsequent hop with nothing watching.
    """

    @classmethod
    def _run(
        cls,
        browser: _FakeBrowser,
        *,
        url: str = "https://public.example/start",
        allow_private_hosts: bool = False,
        on_redirect: Callable[[str], None] = ignore,
        headers: dict[str, str] | None = None,
        max_redirects: int = 10,
    ) -> _FakeTab:
        """Drive one load and return the tab that served it."""
        asyncio.run(
            _drive(
                browser,
                url,
                headers=headers,
                allow_private_hosts=allow_private_hosts,
                on_redirect=on_redirect,
                max_redirects=max_redirects,
            ),
        )
        assert browser.last_tab is not None
        return browser.last_tab

    def test_a_hop_adding_no_header_keeps_chromes_own_header_set(self) -> None:
        """A guard that adds nothing must not rewrite the request's headers.

        An override on ``Fetch.continueRequest`` is unreliable: ``Cookie`` is
        overridden INTERMITTENTLY (crbug 40762053), and overrides do not extend
        to later redirect hops. Measured live, an echo override got Cloudflare's
        wall where the no-override control got the page.
        """
        browser = _FakeBrowser(
            paused_events=[
                _request_paused("https://public.example/next", {"accept": "text/html"}),
            ],
        )
        tab = self._run(browser)
        assert tab.continued_requests == ["req-1"]
        assert _continue_override(tab) is None

    def test_caller_headers_do_not_cross_an_origin_boundary(self) -> None:
        """A caller credential must not follow a redirect to another origin.

        ``set_extra_http_headers`` is TAB-scoped, so Chrome re-sends whatever
        was installed on every hop that tab makes. The credential is withheld
        from it and the cross-origin hop continues without an override.
        """
        browser = _FakeBrowser(
            paused_events=[
                _request_paused("https://evil.example/steal", {"accept": "text/html"}),
            ],
        )
        tab = self._run(
            browser,
            headers={"Authorization": "Bearer secret", "Accept": "text/html"},
        )
        assert _continue_override(tab) is None
        assert "authorization" not in _extra_http_headers(tab)
        # A non-credential header is not the hazard and must still travel.
        assert _extra_http_headers(tab).get("accept") == "text/html"

    def test_caller_headers_survive_a_same_origin_hop(self) -> None:
        # Same origin is what the header was seeded FOR.
        browser = _FakeBrowser(
            paused_events=[_request_paused("https://public.example/next")],
        )
        tab = self._run(browser, headers={"Authorization": "Bearer secret"})
        assert _continued_headers(tab)["authorization"] == "Bearer secret"

    def test_an_entitled_header_replaces_chromes_row_rather_than_adding_one(
        self,
    ) -> None:
        """One header name must yield ONE row, whatever case the caller used.

        Chrome reports the paused request's headers Title-Cased, so a caller
        spelling one lower-case produced BOTH keys and two rows on the wire.
        """
        browser = _FakeBrowser(
            paused_events=[
                _request_paused(
                    "https://public.example/next",
                    {"Authorization": "chrome-own", "Accept": "text/html"},
                ),
            ],
        )
        tab = self._run(browser, headers={"authorization": "caller"})
        payload = next(
            c
            for c in tab.wire_commands
            if from_plain(c.get("method"), str, default="") == "Fetch.continueRequest"
        )
        entries = from_plain(
            from_plain(payload.get("params"), dict[str, object], default={}).get(
                "headers",
            ),
            list[object],
            default=[],
        )
        names = [
            from_plain(
                from_plain(e, dict[str, object]).get("name"),
                str,
                default="",
            ).lower()
            for e in entries
        ]
        assert names.count("authorization") == 1, f"duplicate row: {names}"
        assert _continued_headers(tab) == {
            "authorization": "caller",
            "accept": "text/html",
        }

    def test_origin_bound_headers_are_never_installed_tab_wide(self) -> None:
        """An origin-bound header must be attached per hop, never tab-wide.

        ``set_extra_http_headers`` applies to EVERY request the tab makes,
        subresources included, and those are not intercepted.
        """
        browser = _FakeBrowser()
        tab = self._run(
            browser,
            headers={"Authorization": "Bearer secret", "Accept": "text/html"},
        )
        installed = _extra_http_headers(tab)
        assert "authorization" not in installed
        assert installed.get("accept") == "text/html"

    def test_origin_bound_headers_use_the_shared_redirect_contract(self) -> None:
        """The browser leg must withhold what the header transports drop.

        ``ORIGIN_BOUND`` includes the extended client hints, not just
        credentials; a hint treated as origin-free would be installed tab-wide.
        """
        browser = _FakeBrowser(
            paused_events=[
                _request_paused("https://public.example/next", {"accept": "text/html"}),
            ],
        )
        tab = self._run(
            browser,
            headers={"Sec-CH-UA-Model": "Pixel", "Accept": "text/html"},
        )
        assert _extra_http_headers(tab).get("sec-ch-ua-model") is None
        assert _continued_headers(tab).get("sec-ch-ua-model") == "Pixel"

    def test_no_extra_headers_are_installed_without_ambient_headers(self) -> None:
        tab = self._run(_FakeBrowser(), headers={"Authorization": "x"})
        methods = [c["method"] for c in tab.wire_commands]
        assert methods == ["Fetch.enable", "Network.enable"]

    def test_private_redirect_target_is_refused_before_it_is_fetched(self) -> None:
        browser = _FakeBrowser(
            paused_events=[_request_paused("http://127.0.0.1:1/secret")],
        )
        tab = self._run(browser)
        # Failed, not continued: the loopback hop must never reach the socket.
        assert tab.failed_requests == ["req-1"]
        assert tab.continued_requests == []

    def test_public_redirect_target_is_allowed(self) -> None:
        browser = _FakeBrowser(
            paused_events=[_request_paused("https://example.com/next")],
        )
        tab = self._run(browser)
        assert tab.continued_requests == ["req-1"]
        assert tab.failed_requests == []

    def test_allow_private_hosts_permits_a_private_target(self) -> None:
        # The caller's statement that it authored the URL: a loopback SearXNG
        # instance must still be reachable through the browser.
        browser = _FakeBrowser(
            paused_events=[_request_paused("http://127.0.0.1:8888/next")],
        )
        tab = self._run(
            browser,
            url="http://127.0.0.1:8888/search",
            allow_private_hosts=True,
        )
        assert tab.continued_requests == ["req-1"]
        assert tab.failed_requests == []

    def test_on_redirect_fires_before_the_hop_is_followed(self) -> None:
        browser = _FakeBrowser(
            paused_events=[_request_paused("https://example.com/next")],
        )
        seen: list[str] = []
        tab = self._run(browser, on_redirect=seen.append)
        assert seen == ["https://example.com/next"]
        assert tab.continued_requests == ["req-1"]

    def test_on_redirect_raising_aborts_the_hop(self) -> None:
        def refuse(url: str) -> None:
            del url
            raise RuntimeError("caller refused the hop")

        browser = _FakeBrowser(
            paused_events=[_request_paused("https://example.com/next")],
        )
        tab = self._run(browser, on_redirect=refuse)
        assert tab.failed_requests == ["req-1"]
        assert tab.continued_requests == []

    def test_a_canonicalized_initial_request_is_not_a_hop(self) -> None:
        """Chrome normalizes the URL it was given; that is not a redirect."""
        browser = _FakeBrowser(
            href="https://example.com/",
            paused_events=[_request_paused("https://example.com/")],
        )
        seen: list[str] = []
        self._run(browser, url="https://example.com", on_redirect=seen.append)
        assert seen == []

    def test_a_fragment_only_difference_is_not_a_hop(self) -> None:
        """A fragment is never sent, so its absence on the wire is not a hop."""
        browser = _FakeBrowser(
            paused_events=[_request_paused("https://example.com/page")],
        )
        seen: list[str] = []
        self._run(
            browser,
            url="https://example.com/page#section",
            on_redirect=seen.append,
        )
        assert seen == []

    def test_a_hop_back_to_the_starting_url_is_still_a_hop(self) -> None:
        """``A -> B -> A`` must report both hops, including the return."""
        browser = _FakeBrowser(
            href="https://a.example/start",
            paused_events=[
                _request_paused("https://b.example/next"),
                _request_paused("https://a.example/start"),
            ],
        )
        seen: list[str] = []
        self._run(browser, url="https://a.example/start", on_redirect=seen.append)
        assert seen == ["https://b.example/next", "https://a.example/start"]

    def test_no_hop_reports_nothing(self) -> None:
        seen: list[str] = []
        self._run(
            _FakeBrowser(href="https://example.com/x"),
            url="https://example.com/x",
            on_redirect=seen.append,
        )
        assert seen == []

    def test_a_redirect_budget_of_zero_refuses_the_hop(self) -> None:
        """``max_redirects=0`` must stop the browser leg too."""
        browser = _FakeBrowser(
            paused_events=[_request_paused("https://public.example/next")],
        )
        tab = self._run(browser, max_redirects=0)
        assert tab.failed_requests == ["req-1"]
        assert tab.continued_requests == []

    def test_a_redirect_budget_bounds_the_chain(self) -> None:
        """One hop is allowed at ``max_redirects=1``; the second is refused."""
        browser = _FakeBrowser(
            paused_events=[
                _request_paused("https://public.example/one", request_id="hop-1"),
                _request_paused("https://public.example/two", request_id="hop-2"),
            ],
        )
        tab = self._run(browser, max_redirects=1)
        assert tab.continued_requests == ["hop-1"]
        assert tab.failed_requests == ["hop-2"]

    def test_interception_is_scoped_to_documents(self) -> None:
        """Only navigations are paused, and that scoping is load-bearing.

        Intercepting every request pauses each subresource until the handler
        answers; measured against live Google, the page never reached
        ``readyState == "complete"``.
        """
        tab = self._run(_FakeBrowser())
        assert tab.wire_commands[0] == {
            "method": "Fetch.enable",
            "params": {
                "patterns": [
                    {
                        "requestStage": "Request",
                        "resourceType": "Document",
                        "urlPattern": "*",
                    },
                ],
            },
        }


def test_guard_ignores_events_other_than_paused_requests() -> None:
    browser = _FakeBrowser(paused_events=[object()], responses=[object()])
    _load_page(browser)
    assert browser.last_tab is not None
    assert browser.last_tab.continued_requests == []
    assert browser.last_tab.failed_requests == []


def test_load_closes_the_tab_when_loading_fails() -> None:
    class _RefusingTab(_FakeTab):
        @override
        async def get(self, url: str) -> _FakeTab:
            del url
            raise RuntimeError("navigation refused")

    class _Browser(_FakeBrowser):
        @override
        async def get(self, url: str, new_tab: bool = False) -> _FakeTab:
            del new_tab
            self.last_tab = _RefusingTab(content="", href="")
            return self.last_tab

    browser = _Browser()
    with pytest.raises(RuntimeError, match="navigation refused"):
        _load_page(browser)
    assert browser.last_tab is not None
    assert browser.last_tab.closed


def test_wire_url_removes_fragment_and_supplies_root_path() -> None:
    assert zendriver._wire_url("https://example.com#frag") == "https://example.com/"
    assert zendriver._wire_url("https://example.com/a#frag") == "https://example.com/a"


def test_send_on_a_closed_loop_drops_the_command() -> None:
    loop = asyncio.new_event_loop()
    loop.close()
    sent: list[object] = []

    async def send(command: object) -> None:
        sent.append(command)

    class SendingTab:
        def send(self, command: object) -> Coroutine[object, object, None]:
            return send(command)

    zendriver._send(loop, cast(Tab, SendingTab()), object())
    assert sent == []


# -- _load settle loop --------------------------------------------------------


class _ScriptedTab(_FakeTab):
    """A :class:`_FakeTab` whose content, handlers, and ready state a test scripts."""

    def __init__(self) -> None:
        super().__init__(content="", href="https://example.com/")


def _browser_with(tab: _FakeTab) -> _FakeBrowser:
    """Return a browser whose every ``get`` hands out ``tab``."""

    class _Browser(_FakeBrowser):
        @override
        async def get(self, url: str, new_tab: bool = False) -> _FakeTab:
            del new_tab
            self.gets.append(url)
            self.last_tab = tab
            return tab

    return _Browser()


def test_load_returns_promptly_when_the_wall_clears_off_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A challenge that clears must be observed then, not at the deadline.

    zendriver dispatches sync handlers off-loop, and a cross-thread
    ``Event.set()`` does not wake the selector.
    """
    wall = "<html><title>Just a moment...</title></html>"
    cleared = threading.Event()
    waiting = threading.Event()
    selector = selectors.DefaultSelector()
    real_select = selector.select

    def select(timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        if timeout is not None and timeout > 0:
            waiting.set()
        return real_select(timeout)

    # Signal only once the loop commits to blocking, so a bare set cannot pass.
    monkeypatch.setattr(selector, "select", select)

    class _ClearsOffLoopTab(_ScriptedTab):
        """Serves the wall until an off-loop navigation event says otherwise.

        Its content never advances on the loop thread, which is the arrangement
        that would hide a wakeup delivered from anywhere else.
        """

        @override
        async def get_content(self) -> str:
            return "<html>ok</html>" if cleared.is_set() else wall

    tab = _ClearsOffLoopTab()

    def clear_from_another_thread() -> None:
        assert waiting.wait(timeout=1.0), "the settle wait never blocked"
        cleared.set()
        for handler in tab.handlers:
            handler(_main_frame_navigated())

    async def go() -> float:
        started = time.monotonic()
        thread = threading.Thread(target=clear_from_another_thread)
        thread.start()
        try:
            body = await _drive(_browser_with(tab), timeout_sec=2.0)
            assert body == b"<html>ok</html>"
            return time.monotonic() - started
        finally:
            thread.join(timeout=1.0)
            assert not thread.is_alive()

    with asyncio.Runner(
        loop_factory=partial(asyncio.SelectorEventLoop, selector=selector),
    ) as runner:
        assert runner.run(go()) < 0.5


def test_load_bounds_a_stalled_document_parse() -> None:
    """The settle budget must bound the whole settle, parse included."""
    wall = "<html><title>Just a moment...</title></html>"

    class _StalledParseTab(_ScriptedTab):
        def __init__(self) -> None:
            super().__init__()
            self.loaded = False

        @override
        async def get_content(self) -> str:
            # Commit a navigation on every read, so the loop always advances to
            # the ready-state wait that has no ceiling of its own.
            for handler in self.handlers:
                handler(_main_frame_navigated())
            return wall

        @override
        async def wait_for_ready_state(
            self,
            until: str = "interactive",
            timeout: int = 10,
        ) -> bool:
            del until, timeout
            if not self.loaded:
                self.loaded = True  # The first wait is the initial load.
                return True
            await asyncio.Event().wait()  # Parses forever.
            raise AssertionError("unreachable")

    async def go() -> float:
        started = time.monotonic()
        body = await _drive(_browser_with(_StalledParseTab()), timeout_sec=0.04)
        assert body == wall.encode()
        return time.monotonic() - started

    assert asyncio.run(asyncio.wait_for(go(), timeout=1.0)) < 0.5


def test_load_settle_ignores_child_frame_and_foreign_events() -> None:
    wall = "<html><title>Just a moment...</title></html>"

    class _NoisyTab(_ScriptedTab):
        @override
        async def get_content(self) -> str:
            for handler in self.handlers:
                handler(object())
                child = _main_frame_navigated()
                child.frame.parent_id = page.FrameId("parent")
                handler(child)
            return wall

        @override
        async def wait_for_ready_state(
            self,
            until: str = "interactive",
            timeout: int = 10,
        ) -> bool:
            del timeout
            self.ready_states.append(until)
            return True

        ready_states: list[str] = []  # noqa: RUF012 -- per-class record.

    tab = _NoisyTab()
    _NoisyTab.ready_states = []
    assert _load_page(_browser_with(tab), timeout_sec=0.04) == wall.encode()
    # Only the initial load: no main-frame navigation reached the settle wait.
    assert _NoisyTab.ready_states == ["complete"]


def test_load_waits_for_complete_after_a_navigation() -> None:
    class RecordingTab(_FakeTab):
        def __init__(self) -> None:
            super().__init__(
                content="",
                href="https://example.com/",
                documents=[
                    "<html><title>Just a moment...</title></html>",
                    "<html>clear</html>",
                ],
            )
            self.ready_states: list[str] = []

        @override
        async def wait_for_ready_state(
            self,
            until: str = "interactive",
            timeout: int = 10,
        ) -> bool:
            del timeout
            self.ready_states.append(until)
            return await super().wait_for_ready_state(until)

    tab = RecordingTab()
    assert _load_page(_browser_with(tab), timeout_sec=2.0) == b"<html>clear</html>"
    # The initial load, then the page the challenge navigated to.
    assert tab.ready_states == ["complete", "complete"]


def test_load_settle_marks_success_body(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def classify(body: str, *, on_success_body: bool = False) -> None:
        del body
        calls.append(on_success_body)

    monkeypatch.setattr(zendriver, "classify_challenge", classify)
    assert _load_page(_FakeBrowser()) == b"<html>ok</html>"
    assert calls == [True]


# -- _load tab close ----------------------------------------------------------


class _SlowCancelTab(_FakeTab):
    """A tab whose close hangs, then does real work while cancelling."""

    def __init__(self) -> None:
        super().__init__(content="<html>ok</html>", href="")
        self.finalized = False

    @override
    async def close(self) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # A real close releases the target and deregisters its handler;
            # both are awaits, so cancellation does not finish them instantly.
            await asyncio.sleep(0)
            self.finalized = True
            raise


def _shrink_close_budget(monkeypatch: pytest.MonkeyPatch) -> list[float | None]:
    """Shrink ``_load``'s 5s tab-close ceiling to 0.01s; return every budget seen."""
    real_timeout = asyncio.timeout
    budgets: list[float | None] = []

    def timeout(delay: float | None) -> asyncio.Timeout:
        budgets.append(delay)
        return real_timeout(0.01 if delay == 5.0 else delay)

    monkeypatch.setattr(asyncio, "timeout", timeout)
    return budgets


def test_an_abandoned_tab_close_is_reaped_before_load_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out close must not outlive the call that gave up on it.

    ``cancel()`` only REQUESTS cancellation; the pool's loop is persistent, so
    an unawaited task survives and raises into a loop nobody is watching.
    """
    tab = _SlowCancelTab()
    budgets = _shrink_close_budget(monkeypatch)

    async def go() -> int:
        before = asyncio.all_tasks()
        assert await _drive(_browser_with(tab), timeout_sec=2.0) == b"<html>ok</html>"
        return len([t for t in asyncio.all_tasks() - before if not t.done()])

    assert asyncio.run(asyncio.wait_for(go(), timeout=10.0)) == 0
    assert tab.finalized, "the abandoned close never ran its cleanup"
    assert budgets == [2.0, 5.0]


def test_tab_close_logs_timeout_and_failure_exactly(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class HangingTab(_FakeTab):
        @override
        async def close(self) -> None:
            await asyncio.Event().wait()

    _shrink_close_budget(monkeypatch)
    caplog.set_level("DEBUG", logger=zendriver.__name__)
    _load_page(
        _browser_with(HangingTab(content="<html>ok</html>", href="")),
        timeout_sec=2.0,
    )
    _load_page(_RaisingCloseBrowser(), timeout_sec=2.0)
    assert [record.getMessage() for record in caplog.records] == [
        "tab close timed out; abandoning the tab",
        "tab close failed; abandoning the tab",
    ]
    assert caplog.records[0].exc_info is None
    assert caplog.records[1].exc_info is not None


# -- BrowserPool --------------------------------------------------------------


@dataclass(slots=True, kw_only=True)
class _Connects:
    """Fakes for ``_shared_chrome`` and ``zendriver.start`` in ``BrowserPool.browser``."""

    shared: list[tuple[Path, bool]]
    started: list[int]
    browsers: list[_FakeBrowser]


@pytest.fixture
def connects(monkeypatch: pytest.MonkeyPatch) -> _Connects:
    """Stub the launch and attach steps; record each."""
    record = _Connects(shared=[], started=[], browsers=[])

    def shared(profile_dir: Path, *, headless: bool) -> int:
        record.shared.append((profile_dir, headless))
        return 7777

    async def start(**kwargs: object) -> _FakeBrowser:
        record.started.append(from_plain(kwargs["port"], int))
        assert kwargs["host"] == "127.0.0.1"
        # 3s to give up on a dead Chrome; a live one answers in 0.3s.
        assert kwargs["browser_connection_timeout"] == 0.5
        assert kwargs["browser_connection_max_tries"] == 6
        await asyncio.sleep(0)
        browser = _FakeBrowser()
        record.browsers.append(browser)
        return browser

    monkeypatch.setattr(zendriver, "_shared_chrome", shared)
    monkeypatch.setattr("zendriver.start", start)
    return record


def test_pool_attaches_to_the_port_the_shared_chrome_reports(
    connects: _Connects,
) -> None:
    pool = BrowserPool()
    try:
        browser = pool.run(pool.browser(_PROFILE, headless=True))
    finally:
        pool.shutdown()
    assert browser is connects.browsers[0]
    assert connects.shared == [(_PROFILE, True)]
    assert connects.started == [7777]


def test_pool_reuses_browser_per_profile(connects: _Connects) -> None:
    pool = BrowserPool()
    try:

        async def go() -> bool:
            a = await pool.browser(_PROFILE, headless=True)
            b = await pool.browser(_PROFILE, headless=True)
            return a is b

        assert pool.run(go())
    finally:
        pool.shutdown()
    assert len(connects.browsers) == 1


def test_pool_keys_separate_profiles(connects: _Connects) -> None:
    pool = BrowserPool()
    try:

        async def go() -> None:
            await pool.browser(_PROFILE / "one", headless=True)
            await pool.browser(_PROFILE / "two", headless=True)

        pool.run(go())
    finally:
        pool.shutdown()
    assert len(connects.browsers) == 2


def test_pool_rejects_mode_change_for_live_profile(connects: _Connects) -> None:
    pool = BrowserPool()
    try:

        async def go() -> None:
            await pool.browser(_PROFILE, headless=True)
            with pytest.raises(
                RuntimeError,
                match=(
                    r"^"
                    + re.escape(
                        "Cannot change Zendriver launch mode for a live profile.",
                    )
                    + r"$"
                ),
            ):
                await pool.browser(_PROFILE, headless=False)

        pool.run(go())
    finally:
        pool.shutdown()
    assert len(connects.browsers) == 1


def test_pool_serializes_concurrent_mode_changes(connects: _Connects) -> None:
    pool = BrowserPool()
    try:

        async def go() -> list[object]:
            return list(
                await asyncio.gather(
                    pool.browser(_PROFILE, headless=True),
                    pool.browser(_PROFILE, headless=False),
                    return_exceptions=True,
                ),
            )

        results = pool.run(go())
    finally:
        pool.shutdown()
    assert len(connects.browsers) == 1
    assert sum(isinstance(result, RuntimeError) for result in results) == 1


def test_pool_reconnects_a_dead_connection(connects: _Connects) -> None:
    pool = BrowserPool()
    try:

        async def go() -> None:
            first = await pool.browser(_PROFILE, headless=True)
            cast(_FakeBrowser, first).connection.closed = True  # Chrome exited.
            second = await pool.browser(_PROFILE, headless=True)
            assert second is not first

        pool.run(go())
    finally:
        pool.shutdown()
    assert len(connects.browsers) == 2


def test_pool_reports_an_unreachable_devtools_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def start(**kwargs: object) -> _FakeBrowser:
        del kwargs
        raise RuntimeError("Failed to connect to browser")

    def shared(profile_dir: Path, *, headless: bool) -> int:
        del profile_dir, headless
        return 9

    monkeypatch.setattr(zendriver, "_shared_chrome", shared)
    monkeypatch.setattr("zendriver.start", start)
    pool = BrowserPool()
    try:
        with pytest.raises(
            zendriver.BrowserUnavailableError,
            match=r"^Could not attach to Chrome on DevTools port 9: ",
        ):
            pool.run(pool.browser(_PROFILE, headless=True))
    finally:
        pool.shutdown()


@pytest.mark.parametrize("death", ["closed", "listener_stopped", "no_listener"])
def test_pool_reconnects_every_dead_connection_shape(
    connects: _Connects,
    death: str,
) -> None:
    pool = BrowserPool()
    try:

        async def go() -> None:
            first = cast(_FakeBrowser, await pool.browser(_PROFILE, headless=True))
            if death == "closed":
                first.connection.closed = True
            elif death == "listener_stopped":
                assert first.connection.listener is not None
                first.connection.listener.running = False
            else:
                first.connection.listener = None
            assert await pool.browser(_PROFILE, headless=True) is not first

        pool.run(go())
    finally:
        pool.shutdown()
    assert len(connects.browsers) == 2


def test_pool_reconnects_a_browser_without_a_connection(connects: _Connects) -> None:
    pool = BrowserPool()
    key = str(_PROFILE.resolve())
    try:

        async def go() -> None:
            first = await pool.browser(_PROFILE, headless=True)
            pool._browsers[key] = (True, cast(Browser, _ConnectionlessBrowser()))
            assert await pool.browser(_PROFILE, headless=True) is not first

        pool.run(go())
    finally:
        pool.shutdown()
    assert len(connects.browsers) == 2


class _ConnectionlessBrowser:
    """A browser whose connection never opened."""

    connection: None = None


def test_pool_run_gives_up_when_its_loop_stopped() -> None:
    """A stopped loop must surface a timeout, never an unbounded wait."""
    pool = BrowserPool()
    try:
        pool._loop.call_soon_threadsafe(pool._loop.stop)
        pool._thread.join(timeout=5)
        assert not pool._thread.is_alive()

        async def never_scheduled() -> str:
            raise AssertionError("A stopped loop must not run the coroutine.")

        with pytest.raises(TimeoutError):
            pool.run(never_scheduled(), timeout_sec=0.05)
    finally:
        pool.shutdown()


def test_pool_run_default_timeout_waits_without_a_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeouts: list[float | None] = []

    class Future:
        def result(self, timeout: float | None) -> int:
            timeouts.append(timeout)
            return 1

    def submit(
        coroutine: Coroutine[object, object, int],
        loop: asyncio.AbstractEventLoop,
    ) -> Future:
        del loop
        coroutine.close()
        return Future()

    pool = BrowserPool()
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)
    try:

        async def value() -> int:
            return 1

        assert pool.run(value()) == 1
        assert timeouts == [None]
    finally:
        pool.shutdown()


def test_pool_shutdown_joins_thread_and_closes_loop() -> None:
    pool = BrowserPool()
    assert pool._thread.name == "loop-web-browser"
    assert pool._thread.daemon is True
    pool.shutdown()
    assert not pool._thread.is_alive()
    assert pool._loop.is_closed()


@pytest.mark.asyncio
async def test_pool_shutdown_under_a_running_loop_runs_its_loop_s_exit_hooks() -> None:
    pool = BrowserPool()
    ran = threading.Event()
    pool.run(_on_exit(ran.set))
    pool.shutdown()
    assert ran.is_set()
    assert pool._loop.is_closed()


async def _on_exit(hook: Callable[[], None]) -> None:
    """Register ``hook`` to run when the running loop closes, as zendriver does."""
    asyncio_atexit.register(hook)


def test_pool_shutdown_twice_is_a_noop() -> None:
    pool = BrowserPool()
    pool.shutdown()
    pool.shutdown()
    assert pool._loop.is_closed()


def test_pool_shutdown_disconnects_without_stopping_the_shared_chrome() -> None:
    """Closing this process's pool must leave Chrome up for every other process.

    ``Browser.stop`` sends ``Browser.close``, which shuts Chrome down for every
    process attached to it.
    """
    pool = BrowserPool()
    browser = _FakeBrowser()
    pool._browsers["profile"] = (True, cast(Browser, browser))

    pool.shutdown()

    assert browser.connection.closed, "the websocket was left open"
    assert browser.stop_calls == 0, "shutdown stopped the shared Chrome"


def test_pool_shutdown_skips_a_browser_without_a_connection() -> None:
    pool = BrowserPool()
    pool._browsers["profile"] = (True, cast(Browser, _ConnectionlessBrowser()))
    pool.shutdown()
    assert pool._loop.is_closed()


def test_pool_shutdown_logs_a_failed_disconnect_and_still_stops(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _FailingConnection(_FakeConnection):
        @override
        async def aclose(self) -> None:
            raise RuntimeError("socket gone")

    browser = _FakeBrowser()
    browser.connection = _FailingConnection()
    pool = BrowserPool()
    pool._browsers["profile"] = (True, cast(Browser, browser))
    caplog.set_level("WARNING", logger=zendriver.__name__)

    pool.shutdown()

    assert [r.getMessage() for r in caplog.records] == [
        "browser disconnect failed during shutdown",
    ]
    assert pool._loop.is_closed()


def test_pool_shutdown_uses_its_literal_default_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = BrowserPool()
    pool._browsers["profile"] = (True, cast(Browser, _FakeBrowser()))
    timeouts: list[float] = []

    def run(
        coroutine: Coroutine[object, object, object],
        *,
        timeout_sec: float,
    ) -> object:
        timeouts.append(timeout_sec)
        coroutine.close()
        return object()

    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(pool, "run", run)
    pool.shutdown()

    assert timeouts == [5.0]


def test_pool_shutdown_spends_one_budget_across_every_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = BrowserPool()
    browsers = [_FakeBrowser() for _ in range(3)]
    for index, browser in enumerate(browsers):
        pool._browsers[f"profile-{index}"] = (True, cast(Browser, browser))
    # The loop thread reads the clock too, so this one never runs dry: the first
    # two reads (deadline, first browser) are 0, every later read is past it.
    reads = [0]

    def monotonic() -> float:
        reads[0] += 1
        return 0.0 if reads[0] <= 2 else 10.0

    monkeypatch.setattr(time, "monotonic", monotonic)

    pool.shutdown(budget_sec=5.0)

    assert [browser.connection.closed for browser in browsers] == [True, False, False]


class _Launched:
    """Stands in for a launched Chrome's ``Popen`` in ``_launched``."""

    def __init__(self, *, stuck: bool) -> None:
        self.stuck = stuck
        self.calls: list[str] = []

    def terminate(self) -> None:
        self.calls.append("terminate")

    def wait(self, timeout: float | None = None) -> int:
        self.calls.append(f"wait:{timeout}")
        if self.stuck and timeout is not None:
            raise subprocess.TimeoutExpired("chrome", timeout)
        return 0

    def kill(self) -> None:
        self.calls.append("kill")


def test_pool_shutdown_stop_launched_terminates_then_kills_a_stuck_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polite, stuck = _Launched(stuck=False), _Launched(stuck=True)
    monkeypatch.setattr(zendriver, "_launched", [polite, stuck])

    BrowserPool().shutdown(stop_launched=True)

    assert polite.calls == ["terminate", "wait:5"]
    assert stuck.calls == ["terminate", "wait:5", "kill", "wait:None"]
    assert zendriver._launched == []


def test_pool_shutdown_leaves_launched_chromes_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched = _Launched(stuck=False)
    monkeypatch.setattr(zendriver, "_launched", [launched])

    BrowserPool().shutdown()

    assert launched.calls == []


def test_pool_run_loop_installs_exact_warning_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.new_event_loop()
    pool = object.__new__(BrowserPool)
    pool._loop = loop
    calls: list[tuple[object, ...]] = []
    event_loops: list[object] = []

    def filterwarnings(*args: object, **kwargs: object) -> None:
        calls.append((*args, *kwargs.values()))

    def set_event_loop(value: object) -> None:
        event_loops.append(value)

    def run_forever() -> None:
        pass

    monkeypatch.setattr(asyncio, "set_event_loop", set_event_loop)
    monkeypatch.setattr(warnings, "filterwarnings", filterwarnings)
    monkeypatch.setattr(loop, "run_forever", run_forever)
    try:
        pool._run_loop()
    finally:
        loop.close()
    assert event_loops == [loop]
    assert calls == [
        ("ignore", DeprecationWarning, r"zendriver\..*"),
        ("ignore", ResourceWarning),
    ]


def test_attach_survives_a_reply_to_a_cancelled_cdp_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CDP reply arriving after its transaction was cancelled must be dropped.

    ``Transaction.__call__`` sets the result unconditionally, so a reply landing
    on a cancelled future raises ``InvalidStateError`` inside the listener and
    kills it for the whole connection.

    Driven through ``BrowserPool.browser`` rather than by calling the patcher
    directly: arming it on the attach path is the contract.
    """

    def shared(profile_dir: Path, *, headless: bool) -> int:
        del profile_dir, headless
        return 1

    async def start(**kwargs: object) -> _FakeBrowser:
        del kwargs
        return _FakeBrowser()

    monkeypatch.setattr(zendriver, "_shared_chrome", shared)
    monkeypatch.setattr("zendriver.start", start)
    # Restored to the VENDOR's own ``__call__``, undoing any arming an earlier
    # test in this process did, so the assertions below cannot pass vacuously.
    monkeypatch.setattr(Transaction, "__call__", _VENDOR_TRANSACTION_CALL)
    pool = BrowserPool()
    try:
        pool.run(pool.browser(_PROFILE, headless=True))
    finally:
        pool.shutdown()

    async def replies() -> None:
        # ``result`` alone: the vendor reads only ``error`` and ``result``.
        cancelled = Transaction(page.navigate("about:blank"))
        cancelled.cancel()
        cancelled(result={"frameId": "F", "loaderId": "L"})
        assert cancelled.cancelled()

        # A reply to a LIVE transaction still has to reach the caller awaiting it.
        live = Transaction(page.navigate("about:blank"))
        live(result={"frameId": "F", "loaderId": "L"})
        assert (live.result())[0] == (page.FrameId("F"))

    asyncio.run(replies())

    # Idempotent: arming twice must not wrap the guard in itself.
    armed = cast(object, Transaction[object].__call__)
    zendriver._tolerate_late_cdp_replies()
    assert cast(object, Transaction[object].__call__) is armed


# -- _shared_chrome, _launch_lock, _devtools_port ------------------------------


def _port_4242(profile_dir: Path) -> int:
    del profile_dir
    return 4242


def _answers_on_4242(port: int) -> bool:
    return port == 4242


def _answers_nowhere(port: int) -> bool:
    del port
    return False


def _answers_everywhere(port: int) -> bool:
    del port
    return True


def _no_port(profile_dir: Path) -> int | None:
    del profile_dir
    return None


def _no_sleep(seconds: float) -> None:
    del seconds


def _popen[T](process: T, *args: object, **kwargs: object) -> T:
    del args, kwargs
    return process


def _failed_ps(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
    del args, kwargs
    return subprocess.CompletedProcess([], 1, stdout="")


class _ChromeProcess:
    """Stands in for the launched Chrome's ``Popen``."""

    def __init__(self, *, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.killed = True

    def wait(self) -> int:
        return -9


def test_shared_chrome_attaches_to_a_running_chrome_instead_of_launching(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A second process must attach to the first one's Chrome, not replace it."""
    asked: list[Path] = []

    def port(profile_dir: Path) -> int:
        asked.append(profile_dir)
        return 4242

    def popen(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"launched despite a live Chrome: {args} {kwargs}")

    monkeypatch.setattr(zendriver, "_devtools_port", port)
    monkeypatch.setattr(zendriver, "_port_answers", _answers_on_4242)
    monkeypatch.setattr(subprocess, "Popen", popen)

    assert zendriver._shared_chrome(tmp_path / "profile", headless=True) == 4242
    assert asked == [tmp_path / "profile"]


@pytest.mark.parametrize("headless", [True, False])
def test_shared_chrome_launches_detached_when_no_chrome_answers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    headless: bool,
) -> None:
    """The shared Chrome must survive its launcher's exit and Ctrl-C."""
    profile = tmp_path / "profile"
    captured: dict[str, object] = {}
    (profile).mkdir()
    (profile / "DevToolsActivePort").write_text("1\n/stale\n")

    def popen(args: list[str], **kwargs: object) -> _ChromeProcess:
        captured["args"] = args
        captured.update(kwargs)
        (profile / "DevToolsActivePort").write_text("6789\n/devtools/browser/x\n")
        return _ChromeProcess()

    def answers(port: int) -> bool:
        return port == 6789

    monkeypatch.setattr(zendriver, "_devtools_port", _port_4242)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(zendriver, "_port_answers", answers)
    monkeypatch.setattr(zendriver, "_launched", [])
    monkeypatch.setattr(zendriver, "_chrome_for_testing", lambda: "")
    monkeypatch.setattr(os, "geteuid", lambda: 0)

    assert zendriver._shared_chrome(profile, headless=headless) == 6789
    assert captured["start_new_session"] is True
    args = cast(list[str], captured["args"])
    assert args[0] == "/fake/chrome"
    assert args[-2:] == ["--remote-debugging-port=0", "about:blank"]
    assert f"--user-data-dir={profile}" in args
    assert ("--headless=new" in args) is headless
    # Chrome's sandbox refuses to run as root.
    assert "--no-sandbox" in args
    assert "--remote-debugging-host=127.0.0.1" in args
    assert "--use-mock-keychain" not in args
    assert not any(
        a.startswith(("--proxy-server=", "--host-resolver-rules=")) for a in args
    )
    assert len(zendriver._launched) == 1


def test_shared_chrome_launches_chrome_for_testing_with_a_mock_keychain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Chrome must come from the build that does not claim ``https``.

    Without ``--use-mock-keychain`` Chrome for Testing raises a Safe Storage
    modal and blocks in startup; stock Chrome keeps its own keychain.
    """
    profile = tmp_path / "profile"
    captured: list[list[str]] = []

    def popen(args: list[str], **kwargs: object) -> _ChromeProcess:
        del kwargs
        captured.append(args)
        (profile / "DevToolsActivePort").write_text("6789\n/x\n")
        return _ChromeProcess()

    monkeypatch.setattr(zendriver, "_devtools_port", _port_4242)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(zendriver, "_port_answers", partial(int.__eq__, 6789))
    monkeypatch.setattr(zendriver, "_launched", [])
    monkeypatch.setattr(zendriver, "_chrome_for_testing", lambda: "/cache/CfT")

    zendriver._shared_chrome(profile, headless=True)
    assert captured[0][0] == "/cache/CfT"
    assert "--use-mock-keychain" in captured[0]


def test_shared_chrome_waits_for_a_reported_port_to_answer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"

    def popen(args: list[str], **kwargs: object) -> _ChromeProcess:
        del args, kwargs
        (profile / "DevToolsActivePort").write_text("4242\n/devtools/browser/x\n")
        return _ChromeProcess()

    answers = iter([False, True])
    sleeps: list[float] = []
    monkeypatch.setattr(zendriver, "_devtools_port", _no_port)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(zendriver, "_launched", [])

    def port_answers(port: int) -> bool:
        del port
        return next(answers)

    monkeypatch.setattr(zendriver, "_port_answers", port_answers)
    monkeypatch.setattr(zendriver, "_chrome_for_testing", lambda: "")
    monkeypatch.setattr(time, "sleep", sleeps.append)
    assert zendriver._shared_chrome(profile, headless=True) == 4242
    assert sleeps == [0.05]


def test_shared_chrome_reports_a_chrome_that_exits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(zendriver, "_devtools_port", _no_port)
    monkeypatch.setattr(
        subprocess,
        "Popen",
        partial(_popen, _ChromeProcess(returncode=21)),
    )
    monkeypatch.setattr(zendriver, "_launched", [])
    monkeypatch.setattr(zendriver, "_chrome_for_testing", lambda: "")
    with pytest.raises(zendriver.BrowserUnavailableError, match="exited with code 21"):
        zendriver._shared_chrome(tmp_path / "profile", headless=True)


def test_shared_chrome_kills_a_chrome_that_never_reports_a_port(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = _ChromeProcess()
    clock = iter([0.0, 0.0, 10.0])
    monkeypatch.setattr(zendriver, "_devtools_port", _no_port)
    monkeypatch.setattr(subprocess, "Popen", partial(_popen, process))
    monkeypatch.setattr(zendriver, "_launched", [])
    monkeypatch.setattr(zendriver, "_chrome_for_testing", lambda: "")
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(time, "sleep", _no_sleep)
    with pytest.raises(zendriver.BrowserUnavailableError, match="did not come up"):
        zendriver._shared_chrome(tmp_path / "profile", headless=True)
    assert process.killed


def test_shared_chrome_reports_a_missing_chrome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def popen(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise FileNotFoundError("no chrome")

    monkeypatch.setattr(zendriver, "_devtools_port", _no_port)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(zendriver, "_chrome_for_testing", lambda: "")
    with pytest.raises(zendriver.BrowserUnavailableError, match="no chrome"):
        zendriver._shared_chrome(tmp_path / "profile", headless=True)


@pytest.mark.cli_python_subprocess
def test_the_launch_lock_serializes_processes(tmp_path: Path) -> None:
    """Two processes arriving together must launch one Chrome, not two."""
    profile = tmp_path / "profile"
    marker = tmp_path / "held"
    script = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from wesearch.fetch.transport import zendriver\n"
        f"with zendriver._launch_lock(Path({str(profile)!r})):\n"
        f"    Path({str(marker)!r}).write_text('x')\n"
        "    time.sleep(1.0)\n"
    )
    holder = subprocess.Popen([sys.executable, "-c", script])  # noqa: S603 -- fixed argv.
    try:
        deadline = time.monotonic() + 20
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists(), "the holder never took the lock"
        started = time.monotonic()
        with zendriver._launch_lock(profile):
            waited = time.monotonic() - started
    finally:
        holder.wait(timeout=20)
    assert waited > 0.3, f"the lock was not exclusive across processes ({waited:.2f}s)"


def test_port_answers_reports_whether_the_port_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Connection:
        def close(self) -> None:
            pass

    def connect(address: tuple[str, int], timeout: float) -> Connection:
        assert address[0] == "127.0.0.1"
        assert timeout == 0.2
        if address[1] != 4242:
            raise ConnectionRefusedError
        return Connection()

    monkeypatch.setattr(socket, "create_connection", connect)
    assert zendriver._port_answers(4242)
    assert not zendriver._port_answers(4243)


def _owned_profile(tmp_path: Path, command: str) -> Path:
    """Make ``tmp_path/profile`` owned by pid 123 running ``command``."""
    profile = tmp_path / "profile"
    proc = tmp_path / "proc" / "123"
    profile.mkdir()
    proc.mkdir(parents=True)
    (profile / "SingletonLock").symlink_to("host-123")
    (proc / "cmdline").write_text(command.format(profile=profile))
    return profile


def test_devtools_port_reads_the_singleton_owner_command(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    process = tmp_path / "proc" / "123"
    profile.mkdir()
    process.mkdir(parents=True)
    (profile / "SingletonLock").symlink_to("host-123")
    (process / "cmdline").write_bytes(
        b"/opt/google/chrome/chrome\0"
        + f"--user-data-dir={profile}\0".encode()
        + b"--remote-debugging-port=4567\0"
        + b"about:blank\0",
    )
    assert zendriver._devtools_port(profile, proc_root=tmp_path / "proc") == 4567


def test_devtools_port_parses_owner_and_command_port_exactly(tmp_path: Path) -> None:
    profile = _owned_profile(
        tmp_path,
        "chrome --user-data-dir={profile} "
        "--remote-debugging-port=4567 extra-token another-token",
    )
    assert zendriver._devtools_port(profile, proc_root=tmp_path / "proc") == 4567


def test_devtools_port_uses_the_first_profile_flag(tmp_path: Path) -> None:
    profile = _owned_profile(
        tmp_path,
        "chrome --user-data-dir={profile} --user-data-dir=/second "
        "--remote-debugging-port=9",
    )
    assert zendriver._devtools_port(profile, proc_root=tmp_path / "proc") == 9


def test_devtools_port_keeps_spaces_and_dashes_in_the_profile_path(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "my profile-with-dash"
    proc = tmp_path / "proc" / "123"
    profile.mkdir()
    proc.mkdir(parents=True)
    (profile / "SingletonLock").symlink_to("host-123")
    (proc / "cmdline").write_text(
        f"chrome --user-data-dir={profile} --remote-debugging-port=7",
    )
    assert zendriver._devtools_port(profile, proc_root=tmp_path / "proc") == 7


def test_devtools_port_reads_the_macos_profile_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "SingletonLock").symlink_to("host-123")

    def fake_run(
        args: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert args == ["/bin/ps", "-p", "123", "-o", "command="]
        assert capture_output
        assert text
        assert not check
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=(
                "/opt/google/chrome/chrome "
                f"--user-data-dir={profile} "
                "--remote-debugging-port=4567 about:blank\n"
            ),
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert (
        zendriver._devtools_port(
            profile,
            proc_root=tmp_path / "proc",
            platform="darwin",
        )
        == 4567
    )


def test_devtools_port_reports_a_failed_macos_ps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "SingletonLock").symlink_to("host-123")
    monkeypatch.setattr(
        subprocess,
        "run",
        _failed_ps,
    )
    assert (
        zendriver._devtools_port(
            profile,
            proc_root=tmp_path / "proc",
            platform="darwin",
        )
        is None
    )


def test_devtools_port_never_runs_ps_on_linux(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "SingletonLock").symlink_to("host-123")

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        raise AssertionError("Linux proc lookup must not invoke ps")

    monkeypatch.setattr(subprocess, "run", run)
    assert (
        zendriver._devtools_port(profile, proc_root=tmp_path / "proc", platform="linux")
        is None
    )


def test_devtools_port_rejects_different_profile_with_shared_prefix(
    tmp_path: Path,
) -> None:
    profile = _owned_profile(
        tmp_path,
        "/opt/google/chrome/chrome --user-data-dir={profile}-other "
        "--remote-debugging-port=4567",
    )
    assert zendriver._devtools_port(profile, proc_root=tmp_path / "proc") is None


def test_devtools_port_rejects_stale_marker_without_profile_owner(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "DevToolsActivePort").write_text("4567\n/devtools/browser/id\n")
    assert zendriver._devtools_port(profile, proc_root=tmp_path / "proc") is None


def test_devtools_port_reads_active_port_when_command_has_no_port(
    tmp_path: Path,
) -> None:
    profile = _owned_profile(
        tmp_path,
        "chrome --user-data-dir={profile} --remote-debugging-port=0",
    )
    (profile / "DevToolsActivePort").write_text("4567\n/devtools/browser/id\n")
    assert zendriver._devtools_port(profile, proc_root=tmp_path / "proc") == 4567


def test_devtools_port_needs_a_debugging_port_flag(tmp_path: Path) -> None:
    profile = _owned_profile(tmp_path, "chrome --user-data-dir={profile}")
    assert zendriver._devtools_port(profile, proc_root=tmp_path / "proc") is None


@pytest.mark.parametrize("active", [None, "not-a-port\n"])
def test_devtools_port_without_a_readable_active_port_is_none(
    tmp_path: Path,
    active: str | None,
) -> None:
    profile = _owned_profile(
        tmp_path,
        "chrome --user-data-dir={profile} --remote-debugging-port=0",
    )
    if active is not None:
        (profile / "DevToolsActivePort").write_text(active)
    assert zendriver._devtools_port(profile, proc_root=tmp_path / "proc") is None


# -- _chrome_for_testing ------------------------------------------------------


def _fetch_browser_install(
    root: Path,
    build: str,
    *,
    name: str = "Google Chrome for Testing",
    arch: str = "chrome-mac-arm64",
) -> Path:
    """Write a stub browser where the download roots put a real one."""
    binary = root / build / arch / f"{name}.app" / "Contents" / "MacOS"
    binary.mkdir(parents=True, exist_ok=True)
    executable = binary / name
    _ = executable.write_text("#!/bin/sh\n")
    return executable


@pytest.fixture
def roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    """Redirect both download roots, as ``(puppeteer, playwright)``."""
    puppeteer = tmp_path / "home" / ".cache" / "puppeteer" / "chrome"
    playwright = tmp_path / "cache" / "ms-playwright"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(zendriver, "cache_dir", lambda: tmp_path / "cache")
    return puppeteer, playwright


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_non_macos_hosts_need_no_separate_browser(platform: str) -> None:
    """Linux and Windows keep the ordinary browser -- there is no capture."""
    assert zendriver._chrome_for_testing(platform=platform) == ""


def test_no_install_falls_back_to_zendriver(
    roots: tuple[Path, Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A miss must cost click capture, never the fetch: "" means "let zendriver
    # find Chrome".
    del roots
    with caplog.at_level("DEBUG", logger=zendriver.logger.name):
        assert zendriver._chrome_for_testing(platform="darwin") == ""
    assert caplog.records[-1].message == (
        "no Chrome for Testing found; falling back to zendriver's Chrome"
    )


def test_a_puppeteer_install_is_found(roots: tuple[Path, Path]) -> None:
    expected = _fetch_browser_install(roots[0], "mac_arm-147.0.7727.57")
    assert zendriver._chrome_for_testing(platform="darwin") == str(expected)


def test_a_playwright_install_is_found(roots: tuple[Path, Path]) -> None:
    expected = _fetch_browser_install(roots[1], "chromium-1217")
    assert zendriver._chrome_for_testing(platform="darwin") == str(expected)


def test_a_chromium_app_is_found(roots: tuple[Path, Path]) -> None:
    expected = _fetch_browser_install(
        roots[0],
        "build",
        name="Chromium",
        arch="chrome-mac-x64",
    )
    assert zendriver._chrome_for_testing(platform="darwin") == str(expected)


def test_the_newest_build_wins(roots: tuple[Path, Path]) -> None:
    # Builds accumulate across upgrades; a stale one eventually cannot read a
    # profile the current browser wrote.
    _ = _fetch_browser_install(roots[0], "mac_arm-131.0.6778.85")
    newest = _fetch_browser_install(roots[0], "mac_arm-147.0.7727.57")
    assert zendriver._chrome_for_testing(platform="darwin") == str(newest)


@pytest.mark.parametrize(
    ("older", "newer"),
    [
        ("mac_arm-99.0.4844.51", "mac_arm-100.0.4896.60"),
        ("chromium-999", "chromium-1000"),
        ("alpha", "beta"),
    ],
)
def test_build_order_is_numeric_then_by_name(
    roots: tuple[Path, Path],
    older: str,
    newer: str,
) -> None:
    # Text comparison ranks "99" above "100", which silently pins fetches to
    # the older browser at every version rollover past a digit boundary; names
    # without digits must still order deterministically.
    _ = _fetch_browser_install(roots[0], older)
    expected = _fetch_browser_install(roots[0], newer)
    assert zendriver._chrome_for_testing(platform="darwin") == str(expected)


def test_a_directory_without_the_binary_is_skipped(roots: tuple[Path, Path]) -> None:
    # Playwright's roots include ``ffmpeg-*`` and other non-browser payloads.
    (roots[1] / "ffmpeg-1011").mkdir(parents=True)
    expected = _fetch_browser_install(roots[1], "chromium-1217")
    assert zendriver._chrome_for_testing(platform="darwin") == str(expected)


def test_an_intel_download_is_found(roots: tuple[Path, Path]) -> None:
    # Runnable under Rosetta on Apple silicon, so keying discovery to the host
    # arch would skip a usable one.
    expected = _fetch_browser_install(
        roots[0],
        "mac-147.0.7727.57",
        arch="chrome-mac-x64",
    )
    assert zendriver._chrome_for_testing(platform="darwin") == str(expected)


# -- open_window, _launch, _watch_window, _close_chrome -----------------------


def test_launch_uses_a_bounded_connect_budget_and_profile_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A dead Chrome must fail fast; a healthy one answers DevTools in ~0.3s."""
    captured: dict[str, object] = {}

    async def start(config: Config) -> _FakeBrowser:
        captured.update(
            {
                "headless": config.headless,
                "profile": config.user_data_dir,
                "sandbox": config.sandbox,
                "executable": config.browser_executable_path,
                "args": config(),
                "timeout": config.browser_connection_timeout,
                "tries": config.browser_connection_max_tries,
            },
        )
        return _FakeBrowser()

    monkeypatch.setattr("zendriver.start", start)
    monkeypatch.setattr(zendriver, "_chrome_for_testing", lambda: "/cache/testing")
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    asyncio.run(zendriver._launch(tmp_path / "nested" / "profile"))

    assert captured["headless"] is False
    assert captured["profile"] == str(tmp_path / "nested" / "profile")
    assert (tmp_path / "nested" / "profile").is_dir()
    assert captured["sandbox"] is False
    assert captured["executable"] == "/cache/testing"
    budget = cast(float, captured["timeout"]) * cast(int, captured["tries"])
    assert cast(float, captured["timeout"]) >= 0.3
    assert budget <= 3.0
    assert "--use-mock-keychain" in cast(list[str], captured["args"])


def test_launch_falls_back_to_zendriver_chrome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    async def start(config: Config) -> _FakeBrowser:
        captured["executable"] = config.browser_executable_path
        captured["args"] = config()
        return _FakeBrowser()

    monkeypatch.setattr("zendriver.start", start)
    monkeypatch.setattr(zendriver, "_chrome_for_testing", lambda: "")
    asyncio.run(zendriver._launch(tmp_path))
    assert captured["executable"] == Config().browser_executable_path
    assert "--use-mock-keychain" not in cast(list[str], captured["args"])


def test_launch_wraps_start_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def start(config: Config) -> _FakeBrowser:
        del config
        raise RuntimeError("connection refused")

    monkeypatch.setattr("zendriver.start", start)
    with pytest.raises(
        zendriver.BrowserUnavailableError,
        match=r"^Could not launch Chrome: connection refused$",
    ):
        asyncio.run(zendriver._launch(tmp_path))


def test_close_chrome_starts_and_stops_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[_FakeBrowser] = []

    async def start(*, host: str, port: int) -> _FakeBrowser:
        assert host == "127.0.0.1"
        assert port == 9222
        browser = _FakeBrowser()
        created.append(browser)
        return browser

    monkeypatch.setattr("zendriver.start", start)
    monkeypatch.setattr(zendriver, "_port_answers", _answers_nowhere)
    asyncio.run(zendriver._close_chrome(9222))
    assert len(created) == 1
    assert created[0].stop_calls == 1


def test_close_chrome_reports_a_stuck_port(monkeypatch: pytest.MonkeyPatch) -> None:
    async def start(*, host: str, port: int) -> _FakeBrowser:
        del host, port
        return _FakeBrowser()

    clock = iter([0.0, 0.0, 10.0])
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("zendriver.start", start)
    monkeypatch.setattr(zendriver, "_port_answers", _answers_everywhere)
    # The module's clock only: the event loop reads ``time.monotonic`` too.
    monkeypatch.setattr(
        zendriver,
        "time",
        SimpleNamespace(monotonic=lambda: next(clock), sleep=time.sleep),
    )
    monkeypatch.setattr(asyncio, "sleep", sleep)
    with pytest.raises(
        zendriver.BrowserUnavailableError,
        match=r"^Chrome on DevTools port 9222 did not close\.$",
    ):
        asyncio.run(zendriver._close_chrome(9222))
    assert sleeps == [0.05]


@pytest.mark.parametrize(
    ("port", "answers"),
    [(None, False), (9222, False), (9222, True)],
)
def test_open_window_closes_the_shared_chrome_under_the_launch_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    port: int | None,
    answers: bool,
) -> None:
    """A fetch between close and launch would relaunch headless and take the lock."""
    events: list[str] = []
    browser = _FakeBrowser()

    @contextlib.contextmanager
    def lock(profile_dir: Path) -> Generator[None]:
        assert profile_dir == tmp_path
        events.append("lock")
        yield
        events.append("unlock")

    async def close(found: int) -> None:
        events.append(f"close:{found}")

    async def launch(profile_dir: Path) -> _FakeBrowser:
        assert profile_dir == tmp_path
        events.append("launch")
        return browser

    async def watch(url: str, browser_arg: Browser) -> set[str]:
        assert url == "https://gated.example/"
        assert browser_arg is browser
        events.append("window")
        return {"visited.example"}

    def clear(host: str) -> int:
        events.append(f"clear:{host}")
        return 1

    monkeypatch.setattr(zendriver, "_launch_lock", lock)

    def devtools_port(profile_dir: Path) -> int | None:
        del profile_dir
        return port

    def port_answers(found: int) -> bool:
        del found
        return answers

    monkeypatch.setattr(zendriver, "_devtools_port", devtools_port)
    monkeypatch.setattr(zendriver, "_port_answers", port_answers)
    monkeypatch.setattr(zendriver, "_close_chrome", close)
    monkeypatch.setattr(zendriver, "_launch", launch)
    monkeypatch.setattr(zendriver, "_watch_window", watch)
    monkeypatch.setattr(zendriver, "clear_host", clear)

    zendriver.open_window("https://gated.example/", profile_dir=tmp_path)

    closed = ["close:9222"] if answers else []
    assert events == [
        "lock",
        *closed,
        "launch",
        "unlock",
        "window",
        "clear:visited.example",
    ]


def _page(
    target_id: str,
    url: str,
    *,
    opener: str | None = None,
) -> target.TargetInfo:
    """Build the target info Chrome reports for one page."""
    return target.TargetInfo(
        target_id=target.TargetID(target_id),
        type_="page",
        title="",
        url=url,
        attached=False,
        can_access_opener=False,
        opener_id=None if opener is None else target.TargetID(opener),
    )


class _WatchedBrowser(_FakeBrowser):
    """A window that delivers scripted target events, then closes."""

    def __init__(self, events: Sequence[object]) -> None:
        super().__init__()
        self._events = list(events)

    @override
    async def get(self, url: str, new_tab: bool = False) -> _FakeTab:
        tab = await super().get(url, new_tab=new_tab)
        for event in self._events:
            await self.connection.emit(event)
        self._events = []
        self.stopped = True
        return tab


def _visits(
    monkeypatch: pytest.MonkeyPatch,
    events: Sequence[object],
) -> tuple[set[str], list[str]]:
    """Feed target events to a watched window; return its visits and clears."""
    cleared: list[str] = []

    def clear(host: str) -> int:
        cleared.append(host)
        return 1

    monkeypatch.setattr(zendriver, "clear_host", clear)
    monkeypatch.setattr(zendriver, "challenged_urls", list)
    browser = _WatchedBrowser(events)
    visited = asyncio.run(
        zendriver._watch_window("about:blank", cast(Browser, browser)),
    )
    return visited, cleared


def test_a_page_the_user_loads_lifts_its_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    visited, cleared = _visits(
        monkeypatch,
        [
            target.TargetCreated(_page("own", "about:blank")),
            target.TargetInfoChanged(_page("own", "https://www.google.com/search?q=x")),
            target.TargetInfoChanged(_page("own", "https://www.google.com/search?q=x")),
        ],
    )
    assert visited == {"www.google.com"}
    assert cleared == ["www.google.com"]


def test_an_agent_tab_and_its_popups_never_lift_a_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visited, cleared = _visits(
        monkeypatch,
        [
            target.TargetCreated(_page("agent", "about:blank#wesearch-agent")),
            target.TargetInfoChanged(_page("agent", "https://www.google.com/")),
            target.TargetCreated(
                _page("popup", "https://popup.example/", opener="agent"),
            ),
        ],
    )
    assert visited == set()
    assert cleared == []


def test_only_web_pages_count_as_visits(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = target.TargetInfo(
        target_id=target.TargetID("worker"),
        type_="service_worker",
        title="",
        url="https://worker.example/sw.js",
        attached=False,
        can_access_opener=False,
    )
    visited, _ = _visits(
        monkeypatch,
        [
            target.TargetCreated(worker),
            target.TargetInfoChanged(_page("own", "chrome://newtab/")),
            target.AttachedToTarget(
                session_id=target.SessionID("s"),
                target_info=_page("x", "https://other.example/"),
                waiting_for_debugger=False,
            ),
        ],
    )
    assert visited == set()


def test_a_misrouted_event_never_counts_as_a_visit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A foreign event type delivered to the handler is ignored (measured live).

    ``AttachedToTarget`` carries a ``target_info`` too, so without the type
    guard it would read as a visit and unlock its host.
    """
    cleared: list[str] = []

    def clear(host: str) -> int:
        cleared.append(host)
        return 1

    class _MisroutingBrowser(_FakeBrowser):
        @override
        async def get(self, url: str, new_tab: bool = False) -> _FakeTab:
            tab = await super().get(url, new_tab=new_tab)
            foreign = target.AttachedToTarget(
                session_id=target.SessionID("s"),
                target_info=_page("x", "https://other.example/"),
                waiting_for_debugger=False,
            )
            for handler in self.connection.handlers[target.TargetCreated]:
                result = handler(foreign, self.connection)
                if inspect.isawaitable(result):
                    await result
            self.stopped = True
            return tab

    monkeypatch.setattr(zendriver, "clear_host", clear)
    monkeypatch.setattr(zendriver, "challenged_urls", list)
    visited = asyncio.run(
        zendriver._watch_window("about:blank", cast(Browser, _MisroutingBrowser())),
    )
    assert visited == set()
    assert cleared == []


def test_a_watch_needs_a_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    browser = _FakeBrowser()
    monkeypatch.setattr(browser, "connection", None)
    with pytest.raises(zendriver.BrowserUnavailableError, match="no DevTools"):
        asyncio.run(zendriver._watch_window("about:blank", cast(Browser, browser)))


def test_the_window_opens_the_url_then_each_challenged_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = _FakeBrowser()
    browser.stopped = True
    tabs: list[_FakeTab] = []
    real_get = browser.get

    async def get(url: str, new_tab: bool = False) -> _FakeTab:
        tab = await real_get(url, new_tab=new_tab)
        tabs.append(tab)
        return tab

    monkeypatch.setattr(browser, "get", get)
    monkeypatch.setattr(
        zendriver,
        "challenged_urls",
        lambda: ["https://gated.example/", "https://locked.example/a"],
    )
    asyncio.run(
        zendriver._watch_window("https://gated.example/", cast(Browser, browser)),
    )
    assert browser.gets == ["about:blank", "about:blank"]
    assert [tab.navigations for tab in tabs] == [
        ["https://gated.example/"],
        ["https://locked.example/a"],
    ]


def test_a_site_that_will_not_open_spares_the_window(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _RefusingBrowser(_FakeBrowser):
        @override
        async def get(self, url: str, new_tab: bool = False) -> _FakeTab:
            raise RuntimeError("refused")

    browser = _RefusingBrowser()
    browser.stopped = True
    monkeypatch.setattr(zendriver, "challenged_urls", list)
    asyncio.run(zendriver._watch_window("https://bad.example/", cast(Browser, browser)))
    assert browser.stop_calls == 0
    assert "could not open https://bad.example/" in caplog.text


def test_the_window_polls_until_chrome_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    browser = _FakeBrowser()
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def sleep(delay: float) -> None:
        if not delay:
            await real_sleep(0)  # The fake tab's own yields.
            return
        sleeps.append(delay)
        browser.stopped = True

    monkeypatch.setattr(zendriver, "challenged_urls", list)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    asyncio.run(zendriver._watch_window("about:blank", cast(Browser, browser)))
    assert sleeps == [0.5]


class _HangingStopBrowser(_FakeBrowser):
    """A browser whose ``stop`` never returns, as a wedged connection's does."""

    @override
    async def stop(self) -> None:
        self.stop_calls += 1
        await asyncio.Event().wait()
        raise AssertionError("The browser stop was never bounded.")


class _RaisingStopBrowser(_FakeBrowser):
    """A browser whose ``stop`` fails outright rather than hanging."""

    @override
    async def stop(self) -> None:
        self.stop_calls += 1
        raise RuntimeError("stop blew up")


def _interrupt_watch(
    monkeypatch: pytest.MonkeyPatch,
    browser: _FakeBrowser,
) -> None:
    """Run the window until its poll is interrupted; expect that interruption."""
    real_sleep = asyncio.sleep

    async def interrupted(delay: float) -> None:
        if not delay:
            await real_sleep(0)  # The fake tab's own yields.
            return
        raise KeyboardInterrupt

    monkeypatch.setattr(zendriver, "challenged_urls", list)
    monkeypatch.setattr(asyncio, "sleep", interrupted)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(zendriver._watch_window("about:blank", cast(Browser, browser)))


def test_an_interrupted_window_stops_chrome_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = _FakeBrowser()
    _interrupt_watch(monkeypatch, browser)
    assert browser.stop_calls == 1


@pytest.mark.parametrize("browser_type", [_RaisingStopBrowser, _HangingStopBrowser])
def test_a_failed_stop_kills_chrome_without_replacing_the_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    browser_type: type[_FakeBrowser],
) -> None:
    """Cleanup must never outrank the failure that triggered it.

    A stop that raises or hangs leaks the whole Chrome tree, and the leaked
    root holds its profile's ``SingletonLock``, so it must be killed; and the
    interruption in flight, not the stop's trouble, must reach the caller.
    """
    browser = browser_type()
    process = _FakeProcess()
    browser._process = process
    real_timeout = asyncio.timeout
    budgets: list[float | None] = []

    def timeout(delay: float | None) -> asyncio.Timeout:
        budgets.append(delay)
        return real_timeout(0.01)

    monkeypatch.setattr(asyncio, "timeout", timeout)
    _interrupt_watch(monkeypatch, browser)
    assert browser.stop_calls == 1
    assert process.kills == 1
    assert budgets == [30.0]
    assert "browser stop failed; killing it" in caplog.text


def test_a_failed_stop_without_a_process_handle_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = _RaisingStopBrowser()
    _interrupt_watch(monkeypatch, browser)
    assert browser.stop_calls == 1


def test_a_kill_that_fails_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    class RaisingProcess(_FakeProcess):
        @override
        def kill(self) -> None:
            raise OSError("already gone")

    browser = _RaisingStopBrowser()
    browser._process = RaisingProcess()
    _interrupt_watch(monkeypatch, browser)


# -- main ---------------------------------------------------------------------


def test_main_prints_lifecycle_and_opens_url(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    opened: list[tuple[str, Path]] = []

    def open_window(url: str, *, profile_dir: Path) -> None:
        opened.append((url, profile_dir))

    profile = data_dir() / "rekursiv-ai" / "wesearch" / "fetch-zendriver"
    monkeypatch.setattr(zendriver, "open_window", open_window)
    monkeypatch.setattr(sys, "argv", ["zendriver", "https://example.com/"])
    assert zendriver.main() == 0
    assert opened == [("https://example.com/", profile)]
    assert capsys.readouterr().out == (
        f"Opening https://example.com/ in Chrome on {profile}"
        " -- close the window when done.\nWindow closed.\n"
    )


def test_main_defaults_to_a_blank_page(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []

    def open_window(url: str, *, profile_dir: Path) -> None:
        del profile_dir
        opened.append(url)

    monkeypatch.setattr(zendriver, "open_window", open_window)
    monkeypatch.setattr(sys, "argv", ["zendriver"])
    zendriver.main()
    assert opened == ["about:blank"]


@pytest.mark.parametrize("prog", ["zendriver", "different-name"])
def test_main_help_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    prog: str,
) -> None:
    monkeypatch.setattr(sys, "argv", [prog, "--help"])
    with pytest.raises(SystemExit) as error:
        zendriver.main()
    assert error.value.code == 0
    assert capsys.readouterr().out == (
        "usage: zendriver [-h] [url]\n\n"
        "Open a URL in a visible Chrome on the profile the zendriver transport fetches "
        "with, to see what a fetch saw or to log in. Cookies you set persist for later "
        "fetches. Close the window when done.\n\n"
        "positional arguments:\n"
        "  url         The URL to open; omit for a blank page.\n\n"
        "options:\n"
        "  -h, --help  show this help message and exit\n\n"
        "Examples:\n"
        "  zendriver https://the-site-that-failed.example/\n"
        "  zendriver https://login.example/  # seat a session cookie\n"
        "  zendriver # opens blank; navigate by hand\n"
    )


def test_no_prose_cites_a_line_number_in_this_package() -> None:
    """A ``file:NNN`` citation in prose rots the moment anything shifts.

    A line number is correct for exactly one commit and nothing re-checks it,
    so name the SYMBOL or the setting instead. Matched on the repo's OWN file
    kinds rather than any ``word.word:digits``, which would sweep in
    ``example.com:443``.
    """
    package = _CWD.parent.parent
    citation = re.compile(
        r"\b[\w./-]+\.(?:py|pyi|toml|ya?ml|cfg|ini|txt|md|sh|json|lock):\d+",
    )
    offenders = [
        f"{path.relative_to(package)}:{number}: {match.group(0)}"
        for path in sorted(package.rglob("*.py"))
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        )
        for match in [citation.search(line)]
        if match is not None
    ]

    assert offenders == [], (
        "prose cites a line number, which silently rots -- name the symbol "
        f"instead: {offenders}"
    )


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
