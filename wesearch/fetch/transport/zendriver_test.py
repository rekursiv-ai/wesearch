"""Tests for ``wesearch.fetch.transport.zendriver`` (zendriver headless fetch backend).

Hermetic: a fake async browser stands in for zendriver, so the transport logic
(cookie-domain filtering, challenge detection, redirect callback, pool reuse)
is exercised with no Chrome and no network.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import GeneratorType
from typing import TYPE_CHECKING, Final, cast, override

import argparse
import asyncio
import atexit
import importlib
import inspect
import os
import re
import selectors
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import warnings

from treekle import from_plain
from zendriver import Browser, Config, Tab
from zendriver.cdp import fetch, network, page
from zendriver.core.connection import Transaction

import pytest

from wesearch.fetch.transport import zendriver
from wesearch.fetch.transport.zendriver import (
    BrowserResult,
    _BrowserPool,
    _Flags,
    _navigate,
)
from wesearch.lib.userdirs import data_dir


if TYPE_CHECKING:
    from collections.abc import Iterator

    from wesearch.types.params import Trust


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
        self.seeded: list[object] = []

    async def get_all(self) -> list[_FakeCookie]:
        return self._cookies

    async def set_all(self, cookies: list[object]) -> None:
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
    ) -> None:
        self._documents = documents if documents is not None else [content]
        self._parsing = parsing
        self._href = href
        self.closed = False
        self.navigations: list[str] = []
        self.commands: list[object] = []
        self.handlers: list[Callable[[object], None]] = []
        self.paused_handlers: list[Callable[[object], None]] = []
        self.continued_requests: list[str] = []
        self.failed_requests: list[str] = []
        self._paused_events: list[object] = list(paused_events or [])
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
    ) -> None:
        self._content = content
        self._href = href
        self._documents = documents
        self._parsing = parsing
        self._paused_events = paused_events or []
        self.cookies = _FakeCookieJar(cookies or [])
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
            # A COPY: the tab drains its list once delivered, and the browser
            # builds a fresh tab per ``get`` (the blank one, then the real
            # navigation), so a shared list is empty by the time it matters.
            paused_events=list(self._paused_events),
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

    async def browser(
        self,
        egress: str,
        profile_dir: Path,
        *,
        headless: bool,
    ) -> _FakeBrowser:
        del egress, profile_dir, headless
        return self._browser


def _patch_pool(monkeypatch: pytest.MonkeyPatch, browser: _FakeBrowser) -> _StubPool:
    pool = _StubPool(browser)
    monkeypatch.setattr(zendriver, "_pool", lambda: pool)
    return pool


def test_launch_browser_caps_dead_browser_connect_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # ``zendriver`` retries the DevTools connection ``browser_connection_max_tries``
    # times, each bounded by ``browser_connection_timeout``. When Chrome cannot
    # connect, the launch blocks for their product before raising. A healthy
    # Chrome exposes DevTools in ~0.3s, so the budget must clear that with margin
    # yet stay small: an unbounded product turns a transient browser gap into a
    # multi-second hang that stacks past the live-test timeout instead of
    # surfacing as a fast skip.
    captured: dict[str, float] = {}

    async def fake_start(config: Config) -> _FakeBrowser:
        captured["timeout"] = config.browser_connection_timeout
        captured["max_tries"] = config.browser_connection_max_tries
        return _FakeBrowser()

    monkeypatch.setattr("zendriver.start", fake_start)
    asyncio.run(
        zendriver._launch_browser(
            tmp_path,
            headless=True,
        ),
    )

    budget = captured["timeout"] * captured["max_tries"]
    assert captured["timeout"] >= 0.3, "connect timeout must clear healthy startup"
    assert budget <= 3.0, f"dead-browser connect budget {budget}s too large"


def test_launch_browser_uses_vanilla_zendriver_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    browser = _FakeBrowser()
    browser_args: list[str] = []

    async def fake_start(config: Config) -> _FakeBrowser:
        browser_args.extend(config())
        return browser

    monkeypatch.setattr("zendriver.start", fake_start)
    result = asyncio.run(
        zendriver._launch_browser(
            tmp_path,
            headless=True,
        ),
    )

    assert result is browser
    assert not any(argument.startswith("--proxy-server=") for argument in browser_args)
    assert not any(
        argument.startswith("--proxy-bypass-list=") for argument in browser_args
    )
    assert not any(
        argument.startswith("--host-resolver-rules=") for argument in browser_args
    )


@pytest.mark.parametrize("headless", [True, False])
def test_launch_browser_uses_the_browser_that_claims_no_url_scheme(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    headless: bool,
) -> None:
    """Chrome must come from the build that does not claim ``https``.

    On zendriver's default the launch runs the installed Chrome under
    ``com.google.Chrome``, which then receives every ``open https://...`` the
    user's clicks produce and drops them. Headed too: the capture follows the
    bundle id, not the window count.
    """
    captured: dict[str, object] = {}

    async def fake_start(config: Config) -> _FakeBrowser:
        captured["executable"] = config.browser_executable_path
        captured["args"] = config()
        return _FakeBrowser()

    monkeypatch.setattr("zendriver.start", fake_start)
    monkeypatch.setattr(
        zendriver,
        "_fetch_browser",
        lambda: "/cache/ChromeForTesting",
    )
    asyncio.run(
        zendriver._launch_browser(
            tmp_path,
            headless=headless,
        ),
    )

    assert captured["executable"] == "/cache/ChromeForTesting"
    # Without it Chrome for Testing raises a Safe Storage modal and blocks in
    # startup: the process is alive, so this surfaces as a browser that
    # launched and never exposed DevTools.
    assert "--use-mock-keychain" in cast(list[str], captured["args"])


def test_launch_browser_falls_back_when_no_reidentified_chrome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Linux, or a macOS host whose clone failed: launch exactly as before.

    ``Config`` resolves an installed Chrome for a falsy path, so ``""`` is
    already "find Chrome yourself" and needs no ``or None`` at the call site.
    Asserted against a default ``Config`` rather than a literal: the point is
    that the empty path is indistinguishable from passing nothing.
    """
    captured: dict[str, object] = {}

    async def fake_start(config: Config) -> _FakeBrowser:
        captured["executable"] = config.browser_executable_path
        captured["args"] = config()
        return _FakeBrowser()

    monkeypatch.setattr("zendriver.start", fake_start)
    monkeypatch.setattr(
        zendriver,
        "_fetch_browser",
        lambda: "",
    )
    asyncio.run(
        zendriver._launch_browser(
            tmp_path,
            headless=True,
        ),
    )

    assert captured["executable"] == Config().browser_executable_path
    # Stock Chrome already holds its Keychain entry, so mocking it here would
    # cut the operator's own browser off from cookies it legitimately has.
    assert "--use-mock-keychain" not in cast(list[str], captured["args"])


def test_the_pool_leaves_zendriver_spawn_alone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Launching must NOT patch the vendor's private spawn helper.

    A patch injecting ``preexec_fn=die_with_parent`` lived here, and it armed a
    parent-death signal that Linux scopes to the LAUNCHING THREAD. Every pooled
    browser is launched from the pool's disposable loop thread, so the kernel
    SIGKILLs a live browser the moment that thread exits -- measured on real
    Chrome as `rc=-9`, against a control where the same launch from a
    long-lived thread survived.

    Teardown is `atexit` instead: process-scoped, and it fires on the ordinary
    exit that actually leaked.
    """
    util = importlib.import_module("zendriver.core.util")
    vendor = cast(Callable[..., object], util._start_process)

    async def fake_start(config: Config) -> _FakeBrowser:
        del config
        return _FakeBrowser()

    monkeypatch.setattr("zendriver.start", fake_start)
    asyncio.run(
        zendriver._launch_browser(
            tmp_path,
            headless=True,
        ),
    )

    assert cast(Callable[..., object], util._start_process) is vendor, (
        "the pool patched zendriver's spawn; a thread-scoped parent-death "
        "signal kills pooled browsers when the pool's loop thread exits"
    )


def test_creating_the_pool_registers_process_exit_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pool that can open browsers must also close them at process exit.

    Nothing else ever closes one: an ordinary exit left 378 Chrome processes
    holding 27.5 GiB, and 225 of those outlived the session that spawned them.
    Registered on pool CREATION, so a process that never fetches pays nothing.
    """
    registered: list[object] = []

    def stub_pool() -> object:
        """Stand in for the pool, so no loop thread or browser is created."""
        return object()

    monkeypatch.setattr(atexit, "register", registered.append)
    monkeypatch.setattr(
        zendriver,
        "_pool_singleton",
        None,
    )
    monkeypatch.setattr(
        zendriver,
        "_BrowserPool",
        stub_pool,
    )

    zendriver._pool()

    assert zendriver.shutdown_browsers in registered


def test_the_pool_never_stops_a_browser_a_caller_still_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only ``shutdown`` may close a pooled browser; nothing may evict one.

    A size cap was added here and reverted: ``browser()`` hands back a raw
    handle and the caller navigates with it OUTSIDE the pool's lock (see
    ``_navigate``), so the pool never learns that a browser went idle. Closing
    one to make room killed a browser mid-fetch -- measured, with the caller
    still holding the reference.

    Bounding the pool needs a checkout scope or a refcount, not an eviction
    policy. Until the contract changes, growth is bounded by the CALLER: the
    autouse ``isolate_user_dirs`` fixture varies ``data_dir()`` per test, and
    the module-scoped teardown in ``wesearch/conftest.py`` is what keeps a run
    from accumulating browsers.
    """
    launched: list[_FakeBrowser] = []

    async def launch(profile_dir: Path, *, headless: bool) -> Browser:
        del profile_dir, headless
        launched.append(_FakeBrowser())
        return cast(Browser, launched[-1])

    pool = zendriver._BrowserPool(serve_control=False)
    try:
        monkeypatch.setattr(pool, "_launch", launch)
        held = pool.run(pool.browser("ip", _PROFILE / "0", headless=True))
        for index in range(1, 4):
            pool.run(pool.browser("ip", _PROFILE / str(index), headless=True))

        closed = [index for index, fake in enumerate(launched) if fake.stopped]
        assert closed == [], (
            f"the pool closed browsers {closed} while a caller still held one; "
            f"a browser is idle only when its caller says so"
        )
        assert not (held).stopped
    finally:
        pool.shutdown()


# -- headed/backend navigation parity -----------------------------------------


def test_open_instance_uses_blank_tab_before_requested_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://gated.example/page?q=x"
    browser = _FakeBrowser()
    browser.stopped = True

    async def fake_launch(profile_dir: Path, *, headless: bool) -> _FakeBrowser:
        assert profile_dir == _PROFILE
        assert headless is False
        return browser

    monkeypatch.setattr(
        zendriver,
        "_launch_browser",
        fake_launch,
    )
    asyncio.run(zendriver._open_instance(url, _PROFILE))

    assert browser.gets == ["about:blank"]
    assert browser.last_tab is not None
    assert browser.last_tab.navigations == [url]


class _RefusingTab(_FakeTab):
    """A tab whose navigation fails, driving ``_open_instance``'s cleanup."""

    @override
    async def get(self, url: str) -> _FakeTab:
        del url
        raise RuntimeError("navigation refused")


class _HangingStopBrowser(_FakeBrowser):
    """A browser whose ``stop`` never returns, as a wedged connection's does.

    ``Browser.stop`` awaits ``connection.send(cdp.browser.close())`` with no
    ceiling of its own; its ``except Exception`` cannot help, because an await
    that never returns raises nothing to catch.
    """

    @override
    async def get(self, url: str, new_tab: bool = False) -> _FakeTab:
        del new_tab
        self.gets.append(url)
        self.last_tab = _RefusingTab(content="<html>ok</html>", href="")
        return self.last_tab

    @override
    async def stop(self) -> None:
        self.stop_calls += 1
        await asyncio.Event().wait()
        raise AssertionError("The browser stop was never bounded.")


def test_a_wedged_browser_stop_gives_up_at_its_budget() -> None:
    """``_stopped`` must return on a stop that never does.

    Driven directly on a SMALL budget: the real default matches the pool's 30s
    ceiling, and asserting against that would make this test cost 30 seconds to
    prove a bound that a fraction of a second proves just as well.
    """
    browser = _HangingStopBrowser()

    async def go() -> float:
        started = time.monotonic()
        await zendriver._stopped(
            cast(Browser, browser),
            budget_sec=0.01,
        )
        return time.monotonic() - started

    elapsed = asyncio.run(asyncio.wait_for(go(), timeout=1.0))
    assert browser.stop_calls == 1, "the browser was never asked to stop"
    assert elapsed < 0.5, f"stop outlived its 0.01s budget: {elapsed:.2f}s"


def test_open_instance_reports_the_setup_error_a_wedged_stop_would_bury(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed setup must report ITS error, not park in the cleanup.

    ``_open_instance`` stops the browser under ``except BaseException`` when
    navigation fails, and nothing above supplies a ceiling: ``open_instance``
    calls ``_pool().run`` with no ``timeout_sec``, which waits forever by
    contract. Unbounded, a wedged stop therefore turns a reportable navigation
    error into a CLI that hangs having printed nothing.

    The budget is shrunk rather than waited out -- what this asserts is that
    the cleanup routes through the bounded helper at all, which the 0.01s
    substitution shows in the same way 30s would.
    """
    browser = _HangingStopBrowser()

    async def fake_launch(profile_dir: Path, *, headless: bool) -> _FakeBrowser:
        del profile_dir, headless
        return browser

    monkeypatch.setattr(
        zendriver,
        "_launch_browser",
        fake_launch,
    )
    monkeypatch.setattr(
        zendriver,
        "_stopped",
        partial(zendriver._stopped, budget_sec=0.01),
    )

    async def go() -> float:
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="navigation refused"):
            await zendriver._open_instance(
                "https://gated.example/page",
                _PROFILE,
            )
        return time.monotonic() - started

    # The outer ceiling must fail instead of accepting a buried setup error.
    elapsed = asyncio.run(asyncio.wait_for(go(), timeout=1.0))
    assert browser.stop_calls == 1, "the browser was never asked to stop"
    assert elapsed < 0.5, f"cleanup outlived its budget: {elapsed:.2f}s"


class _RaisingStopBrowser(_FakeBrowser):
    """A browser whose ``stop`` fails outright rather than hanging."""

    @override
    async def get(self, url: str, new_tab: bool = False) -> _FakeTab:
        del new_tab
        self.gets.append(url)
        self.last_tab = _RefusingTab(content="<html>ok</html>", href="")
        return self.last_tab

    @override
    async def stop(self) -> None:
        self.stop_calls += 1
        raise RuntimeError("stop blew up")


def test_a_failing_browser_stop_does_not_replace_the_error_it_cleans_up_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup must never outrank the failure that triggered it.

    ``_stopped`` bounds a stop that HANGS, but an ordinary exception from it
    propagates out of the ``except BaseException`` block and replaces the
    navigation error in flight -- so the operator is told the browser would not
    close, and never told why the page failed. A cleanup that reports its own
    trouble instead of the caller's is strictly worse than one that says
    nothing.
    """
    browser = _RaisingStopBrowser()

    async def fake_launch(profile_dir: Path, *, headless: bool) -> _FakeBrowser:
        del profile_dir, headless
        return browser

    monkeypatch.setattr(
        zendriver,
        "_launch_browser",
        fake_launch,
    )

    with pytest.raises(RuntimeError, match="navigation refused"):
        asyncio.run(
            zendriver._open_instance(
                "https://gated.example/page",
                _PROFILE,
            ),
        )
    assert browser.stop_calls == 1, "the browser was never asked to stop"


def test_a_wedged_browser_stop_kills_the_process() -> None:
    """A stop that never returns must not leave the browser running.

    Abandoning it leaks the whole Chrome tree, and the leaked root holds its
    profile's ``SingletonLock``, so the next launch on that profile fails with
    no usable diagnosis.
    """
    browser = _HangingStopBrowser()
    process = _FakeProcess()
    browser._process = process

    asyncio.run(
        zendriver._stopped(
            cast(Browser, browser),
            budget_sec=0.01,
        ),
    )

    assert process.kills == 1, "a wedged browser was abandoned rather than killed"


def test_a_failing_browser_stop_kills_the_process() -> None:
    """A stop that raises leaks exactly as a wedged one does."""
    browser = _RaisingStopBrowser()
    process = _FakeProcess()
    browser._process = process

    asyncio.run(
        zendriver._stopped(
            cast(Browser, browser),
            budget_sec=1.0,
        ),
    )

    assert process.kills == 1, "a failed stop abandoned the browser"


def test_killing_a_browser_that_was_never_launched_is_a_no_op() -> None:
    """Cleanup must not raise when there is no process handle.

    It runs with the caller's real error in flight, so anything raised here
    would replace it.
    """
    zendriver._kill_browser_process(
        cast(Browser, _FakeBrowser()),
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


def test_a_failing_tab_close_does_not_mask_the_fetch_it_cleans_up_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same rule for the tab: teardown reports nothing over the caller.

    ``_closed`` runs from ``_navigate``'s ``finally``, so an exception it lets
    escape replaces whatever the fetch was about to return OR raise -- turning
    a harvested page into a teardown error the caller cannot act on.
    """
    browser = _RaisingCloseBrowser()
    _patch_pool(monkeypatch, browser)

    result = asyncio.run(
        _navigate(
            "https://example.com/",
            profile_dir=_PROFILE,
            egress="e",
            timeout_sec=5.0,
            headless=True,
            on_redirect=None,
        ),
    )
    assert result.body == b"<html><title>Plain</title>ok</html>"


class _SlowCancelTab(_FakeTab):
    """A tab whose close hangs, then does real work while cancelling."""

    def __init__(
        self,
        *,
        content: str = "<html>ok</html>",
        href: str = "",
        documents: list[str] | None = None,
        parsing: str | None = None,
        paused_events: list[object] | None = None,
    ) -> None:
        super().__init__(
            content=content,
            href=href,
            documents=documents,
            parsing=parsing,
            paused_events=paused_events,
        )
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


def test_an_abandoned_tab_close_is_reaped_before_the_helper_returns() -> None:
    """A timed-out close must not outlive the call that gave up on it.

    ``cancel()`` only REQUESTS cancellation; without awaiting the task, the
    helper returns while it is still pending. Under ``asyncio.run`` the loop
    dies immediately and hides this, but the pool's loop is persistent
    (``_BrowserPool._run_loop`` runs forever), so the task survives --
    finalizing late, or never, and raising into a loop nobody is watching.
    """
    tab = _SlowCancelTab(content="<html>ok</html>", href="")

    async def go() -> tuple[int, bool]:
        before = asyncio.all_tasks()
        await zendriver._closed(
            cast(Tab, tab),
            budget_sec=0.01,
        )
        leaked = [task for task in asyncio.all_tasks() - before if not task.done()]
        return len(leaked), tab.finalized

    pending, finalized = asyncio.run(asyncio.wait_for(go(), timeout=10.0))
    assert pending == 0, (
        f"{pending} close task(s) still pending after the helper returned"
    )
    assert finalized, "the abandoned close never ran its cleanup"


def test_open_instance_releases_profile_then_clears_domain_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakePool:
        def run(self, coroutine: Coroutine[object, object, object]) -> None:
            events.append("launch")
            coroutine.close()

    def release(profile: Path) -> None:
        del profile
        events.append("release")

    def clear(domain: str) -> int:
        events.append(f"clear:{domain}")
        return 1

    monkeypatch.setattr(
        zendriver,
        "_request_pool_release",
        release,
    )
    monkeypatch.setattr(
        zendriver,
        "clear_domain_cooldowns",
        clear,
    )
    monkeypatch.setattr(zendriver, "_pool", FakePool)

    zendriver.open_instance(
        "https://gated.example/page",
        profile_dir=_PROFILE,
    )

    assert events == ["release", "launch", "clear:gated.example"]


def test_pool_control_releases_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    releases: list[bool] = []
    checked: list[Path] = []
    monkeypatch.setattr(
        zendriver,
        "_close_orphan_browser",
        checked.append,
    )
    server = zendriver._PoolControlServer(
        _PROFILE,
        lambda: releases.append(True),
    )
    try:
        zendriver._request_pool_release(_PROFILE)
    finally:
        server.close()
    assert releases == [True]
    assert checked == [_PROFILE]


def test_control_address_uses_platform_socket_namespace() -> None:
    linux_address = zendriver._control_address(
        _PROFILE,
        platform="linux",
    )
    darwin_address = zendriver._control_address(
        _PROFILE,
        platform="darwin",
    )

    assert linux_address.startswith("\0loop-zendriver-")
    assert Path(darwin_address).parent == Path(tempfile.gettempdir())
    assert Path(darwin_address).name.startswith("loop-zd-")
    assert darwin_address.endswith(".sock")


def test_pool_release_closes_orphan_when_control_is_unreachable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    closed: list[Path] = []
    monkeypatch.setattr(
        zendriver,
        "_close_orphan_browser",
        closed.append,
    )

    zendriver._request_pool_release(tmp_path)

    assert closed == [tmp_path]


def test_devtools_port_falls_back_to_singleton_owner(tmp_path: Path) -> None:
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

    assert (
        zendriver._devtools_port(
            profile,
            proc_root=tmp_path / "proc",
        )
        == 4567
    )


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


def test_devtools_port_rejects_different_profile_with_shared_prefix(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"
    process = tmp_path / "proc" / "123"
    profile.mkdir()
    process.mkdir(parents=True)
    (profile / "SingletonLock").symlink_to("host-123")
    (process / "cmdline").write_text(
        "/opt/google/chrome/chrome "
        f"--user-data-dir={profile}-other "
        "--remote-debugging-port=4567",
    )

    assert (
        zendriver._devtools_port(
            profile,
            proc_root=tmp_path / "proc",
        )
        is None
    )


def test_devtools_port_rejects_stale_marker_without_profile_owner(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "DevToolsActivePort").write_text("4567\n/devtools/browser/id\n")

    assert (
        zendriver._devtools_port(
            profile,
            proc_root=tmp_path / "proc",
        )
        is None
    )


# -- _navigate: body + cookie harvest ----------------------------------------


def test_navigate_returns_body_and_domain_cookies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    browser = _FakeBrowser(
        content="<html>results</html>",
        cookies=[
            _FakeCookie(name="SID", value="abc", domain=".gated.example"),
            _FakeCookie(name="OTHER", value="zzz", domain="example.com"),
        ],
    )
    _patch_pool(monkeypatch, browser)
    result = asyncio.run(
        _navigate(
            "https://gated.example/page?q=x",
            profile_dir=_PROFILE,
            egress="1.2.3.4",
            timeout_sec=5.0,
            headless=True,
            on_redirect=None,
        ),
    )
    assert result.body == b"<html>results</html>"
    # Only the domain-matching cookie is harvested; the foreign one is dropped.
    assert result.cookies == {"SID": "abc"}
    # The per-fetch tab is closed after harvest -- the memory-teardown contract
    # (Chrome process stays warm; the scraped page's tab does not).
    assert browser.last_tab is not None
    assert browser.last_tab.closed is True


def test_navigate_reports_the_url_its_cookies_belong_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The result must name the origin its cookies were harvested for.

    Cookies are keyed to the FINAL url, because a cross-origin redirect seats
    the target's. Returning them without saying so left the caller filing
    ``b.example``'s session cookie under ``a.example`` -- and sending it back
    to ``a.example`` on the next fetch.
    """
    browser = _FakeBrowser(
        href="https://b.example/landing",
        cookies=[_FakeCookie(name="B_SESSION", value="secret", domain="b.example")],
    )
    _patch_pool(monkeypatch, browser)
    result = asyncio.run(
        _navigate(
            "https://a.example/start",
            profile_dir=_PROFILE,
            egress="1.2.3.4",
            timeout_sec=5.0,
            headless=True,
            on_redirect=None,
        ),
    )
    assert result.cookies == {"B_SESSION": "secret"}
    assert result.final_url == "https://b.example/landing"


def test_navigate_unwraps_chromes_json_viewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A JSON response must come back as JSON, not as Chrome's viewer shell.

    ``get_content`` serializes the DOM, and for a non-HTML body Chrome
    SYNTHESIZES a document to display it: the payload is re-wrapped in
    ``<html><head>...</head><body><pre>``. A caller that asked a JSON endpoint
    for JSON then gets markup around valid data and fails to parse it. The
    original bytes are still there, inside the ``<pre>``.
    """
    payload = '{"query": "x", "results": []}'
    browser = _FakeBrowser(
        content=(
            '<html><head><meta name="color-scheme" content="light dark">'
            '<meta charset="utf-8"></head><body>'
            f"<pre>{payload}</pre></body></html>"
        ),
    )
    _patch_pool(monkeypatch, browser)
    result = asyncio.run(
        _navigate(
            "https://search.example/search?format=json",
            profile_dir=_PROFILE,
            egress="1.2.3.4",
            timeout_sec=5.0,
            headless=True,
            on_redirect=None,
        ),
    )
    assert result.body == payload.encode()


def test_navigate_unwraps_the_real_chrome_viewer_markup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verbatim shell captured from Chrome, not a hand-written approximation.

    Chrome mounts its JSON formatter in a ``<div>`` AFTER the ``</pre>``, so
    the payload is not the last node in the body. A pattern requiring
    ``</pre></body>`` adjacency matched an invented fixture and missed every
    real response -- the fixture agreed with the code because the same
    assumption wrote both.
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
    _patch_pool(monkeypatch, browser)
    result = asyncio.run(
        _navigate(
            "https://search.example/search?format=json",
            profile_dir=_PROFILE,
            egress="1.2.3.4",
            timeout_sec=5.0,
            headless=True,
            on_redirect=None,
        ),
    )
    assert result.body == payload.encode()


def test_navigate_leaves_real_html_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine page containing a ``<pre>`` must not be reduced to it."""
    content = "<html><body><h1>Title</h1><pre>code sample</pre></body></html>"
    browser = _FakeBrowser(content=content)
    _patch_pool(monkeypatch, browser)
    result = asyncio.run(
        _navigate(
            "https://example.com/article",
            profile_dir=_PROFILE,
            egress="1.2.3.4",
            timeout_sec=5.0,
            headless=True,
            on_redirect=None,
        ),
    )
    assert result.body == content.encode()


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


# Distinct from :func:`_continued_headers`, which reports what the override said: the
# question here is whether an override was sent AT ALL. That is the property under test,
# because an override is unreliable whatever it carries -- ``Cookie`` applies
# intermittently and no override survives a redirect hop (see
# :func:`~wesearch.fetch.transport.zendriver._continue`) -- so sending one at all
# is the hazard, not any value inside it.
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


# The genuine CDP dataclass, not a look-alike: the guard filters on ``isinstance``, so a
# stand-in would satisfy the fake and be ignored in production -- the direction a test
# must never fail in.
#
# ``headers`` are the ones Chrome reports ALREADY on the paused request, which include
# whatever ``set_extra_http_headers`` installed on the tab -- the replay this guard
# exists to trim.
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


class TestBrowserHonorsTrustPerHop:
    """Chrome follows redirects itself, so each hop must be checked before it runs.

    The header transports re-validate every hop (both call ``pinned_host`` per
    hop) because ``common.py`` states the rule: "A redirect target is a URL like
    any other and must be re-checked; skipping that is the classic SSRF
    bypass." The browser leg reached the same rule through a different
    door -- it validated the URL the CALLER passed and then handed navigation to
    Chrome, which fetched every subsequent hop with nothing watching.
    """

    def test_fetch_zendriver_accepts_trust(self) -> None:
        # The signature IS the enforcement: a transport that cannot express the
        # policy cannot be held to it, and no reviewer of the call site sees the
        # gap because that line does call ``pinned_host``.
        assert (
            "trust"
            in inspect.signature(
                zendriver.fetch_zendriver,
            ).parameters
        )

    @classmethod
    def _run(
        cls,
        browser: _FakeBrowser,
        *,
        url: str = "https://public.example/start",
        trust: Trust = "untrusted",
        on_redirect: Callable[[str], None] | None = None,
        headers: dict[str, str] | None = None,
    ) -> _FakeTab:
        """Drive one navigation and return the tab that served it."""
        asyncio.run(
            _navigate(
                url,
                profile_dir=_PROFILE,
                egress="1.2.3.4",
                timeout_sec=5.0,
                headless=True,
                trust=trust,
                headers=headers,
                on_redirect=on_redirect,
            ),
        )
        assert browser.last_tab is not None
        return browser.last_tab

    def test_a_hop_adding_no_header_keeps_chromes_own_header_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A guard that adds nothing must not rewrite the request's headers.

        An override on ``Fetch.continueRequest`` is unreliable by documented
        behavior, so echoing headers back can only subtract:

        - ``Cookie`` is overridden INTERMITTENTLY (crbug 40762053: "only 3 of
          21 requests ... have the cookie override"), and a clearance cookie
          that rides only sometimes reads as an unsolved challenge.
        - Overrides "do not extend to subsequent redirect hops" (CDP ``Fetch``
          docs), and the clear IS a redirect chain -- measured GET, POST, GET.

        Measured against one live Cloudflare-fronted URL, interleaved with the
        no-override control on a fresh egress, same browser and profile, the
        only difference being the continue verb::

            continue_request(id, headers=[echo of request.headers])
                ->  5619 / 5696 bytes, "Just a moment..."
            continue_request(id)
                -> 405954 / 405978 bytes, the real page

        Interleaving is load-bearing: Cloudflare scores the EGRESS, so a
        sequential A-then-B run degrades under its own probing and the control
        stops clearing -- which reads as an arm effect and is not one.
        """
        browser = _FakeBrowser(
            paused_events=[
                _request_paused("https://public.example/next", {"accept": "text/html"}),
            ],
        )
        _patch_pool(monkeypatch, browser)
        tab = self._run(browser)
        assert tab.continued_requests == ["req-1"]
        assert _continue_override(tab) is None

    def test_a_cross_origin_hop_drops_the_credential_without_rewriting(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Withholding a header is not a reason to replace Chrome's set.

        The credential is dropped by never installing it tab-wide (see
        ``test_origin_bound_headers_are_never_installed_tab_wide``), so Chrome
        never had it on this hop to begin with. Overriding here would add the
        bot-detection tell above while removing nothing.
        """
        browser = _FakeBrowser(
            paused_events=[
                _request_paused("https://evil.example/steal", {"accept": "text/html"}),
            ],
        )
        _patch_pool(monkeypatch, browser)
        tab = self._run(
            browser,
            headers={"Authorization": "Bearer secret", "Accept": "text/html"},
        )
        assert _continue_override(tab) is None

    def test_caller_headers_do_not_cross_an_origin_boundary(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A caller credential must not follow a redirect to another origin.

        ``set_extra_http_headers`` is TAB-scoped, so Chrome re-sends whatever
        was installed on every hop that tab makes. An ``Authorization`` seeded
        for the requested origin therefore reached a redirect target the caller
        never chose. The header transports already refuse this: ``common.py``
        rewrites ``Origin`` cross-origin for the same reason.
        """
        # Chrome reports only what was installed tab-wide, which is the
        # origin-free subset; the credential is attached by the guard, per hop.
        browser = _FakeBrowser(
            paused_events=[
                _request_paused("https://evil.example/steal", {"accept": "text/html"}),
            ],
        )
        _patch_pool(monkeypatch, browser)
        tab = self._run(
            browser,
            headers={"Authorization": "Bearer secret", "Accept": "text/html"},
        )
        # Asserted on what the hop CARRIES, not on how it was assembled: the
        # request continues without an override, so it carries exactly Chrome's
        # own set, and the credential is absent from that set because it was
        # never installed tab-wide.
        assert _continue_override(tab) is None
        assert "authorization" not in _extra_http_headers(tab)
        # A non-credential header is not the hazard and must still travel.
        assert _extra_http_headers(tab).get("accept") == "text/html"

    def test_caller_headers_survive_a_same_origin_hop(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Same origin is what the header was seeded FOR; stripping there would
        # break every authenticated fetch that redirects internally.
        browser = _FakeBrowser(
            paused_events=[_request_paused("https://public.example/next")],
        )
        _patch_pool(monkeypatch, browser)
        tab = self._run(browser, headers={"Authorization": "Bearer secret"})
        assert "authorization" in _continued_headers(tab)

    def test_an_entitled_header_replaces_chromes_row_rather_than_adding_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One header name must yield ONE row, whatever case the caller used.

        HTTP field names are case-insensitive, but a dict merge is not: Chrome
        reports the paused request's headers Title-Cased (measured:
        ``['Accept', 'Cookie', 'Upgrade-Insecure-Requests', 'User-Agent']``),
        so a caller spelling one lower-case produced BOTH keys and the override
        emitted two rows for it. ``fetch.py`` collapses exactly this on the curl
        leg and names the cost: "Two dict keys ... would emit two Cookie lines
        on the wire -- a bot tell."

        ``Cookie`` is the case that can actually collide, because unlike
        ``Authorization`` it is not withheld from the tab-wide install, so
        Chrome holds one of its own.
        """
        browser = _FakeBrowser(
            paused_events=[
                _request_paused(
                    "https://public.example/next",
                    {"Cookie": "chrome=own"},
                ),
            ],
        )
        _patch_pool(monkeypatch, browser)
        tab = self._run(browser, headers={"cookie": "caller=seeded"})
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
        assert names.count("cookie") == 1, f"duplicate Cookie row: {names}"
        # The caller's value is the one that must survive: a per-call cookie is
        # an explicit override, matching ``set_session_cookies`` on the curl leg.
        assert _continued_headers(tab)["cookie"] == "caller=seeded"

    def test_origin_bound_headers_are_never_installed_tab_wide(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An origin-bound header must be attached per hop, never tab-wide.

        ``set_extra_http_headers`` applies to EVERY request the tab makes --
        subresources included, and those are not intercepted (interception is
        document-scoped, because pausing each subresource for a DNS resolution
        stalled the page). Trimming at the guard therefore protected redirect
        hops and nothing else: a cross-origin image or XHR still carried the
        caller's ``Authorization``. Installing only the origin-free headers
        removes the leak at its source; the guard re-attaches the rest to the
        document hops entitled to them.
        """
        browser = _FakeBrowser()
        _patch_pool(monkeypatch, browser)
        tab = self._run(
            browser,
            headers={"Authorization": "Bearer secret", "Accept": "text/html"},
        )
        installed = _extra_http_headers(tab)
        assert "authorization" not in installed
        assert installed.get("accept") == "text/html"

    def test_origin_bound_headers_use_the_shared_redirect_contract(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The browser leg must drop what the header transports drop.

        ``common.apply_redirect`` drops every origin-bound header cross-origin,
        and that set includes the extended client hints -- not just credentials.
        A browser leg with its own shorter list leaks the source origin's
        fingerprint to a redirect target that the curl leg would never tell.
        """
        seeded = {"Sec-CH-UA-Model": "Pixel", "Accept": "text/html"}
        # Asserted on the SAME-origin hop, which is where the two candidate
        # sets differ observably: a hint treated as origin-free is installed
        # tab-wide (so it reaches every origin and never appears here), while
        # one treated as origin-bound is withheld and re-attached exactly here.
        browser = _FakeBrowser(
            paused_events=[
                _request_paused("https://public.example/next", {"accept": "text/html"}),
            ],
        )
        _patch_pool(monkeypatch, browser)
        tab = self._run(browser, headers=seeded)
        assert _extra_http_headers(tab).get("sec-ch-ua-model") is None
        assert _continued_headers(tab).get("sec-ch-ua-model") == "Pixel"

    def test_private_redirect_target_is_refused_before_it_is_fetched(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        browser = _FakeBrowser(
            paused_events=[_request_paused("http://127.0.0.1:1/secret")],
        )
        _patch_pool(monkeypatch, browser)
        tab = self._run(browser)
        # Failed, not continued: the loopback hop must never reach the socket.
        assert tab.failed_requests == ["req-1"]
        assert tab.continued_requests == []

    def test_public_redirect_target_is_allowed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        browser = _FakeBrowser(
            paused_events=[_request_paused("https://example.com/next")],
        )
        _patch_pool(monkeypatch, browser)
        tab = self._run(browser)
        assert tab.continued_requests == ["req-1"]
        assert tab.failed_requests == []

    def test_internal_trust_permits_a_private_target(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # ``internal`` is the caller's statement that it authored the URL, so a
        # loopback SearXNG instance must still be reachable through the browser.
        browser = _FakeBrowser(
            paused_events=[_request_paused("http://127.0.0.1:8888/next")],
        )
        _patch_pool(monkeypatch, browser)
        tab = self._run(browser, url="http://127.0.0.1:8888/search", trust="internal")
        assert tab.continued_requests == ["req-1"]
        assert tab.failed_requests == []

    def test_on_redirect_fires_before_the_hop_is_followed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # ObserveParams documents on_redirect as "called with the redirect target
        # URL before following; raise to abort". Firing it after the load makes
        # the abort unreachable.
        browser = _FakeBrowser(
            paused_events=[_request_paused("https://example.com/next")],
        )
        _patch_pool(monkeypatch, browser)
        seen: list[str] = []
        tab = self._run(browser, on_redirect=seen.append)
        assert seen == ["https://example.com/next"]
        assert tab.continued_requests == ["req-1"]

    def test_on_redirect_raising_aborts_the_hop(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def refuse(url: str) -> None:
            del url
            raise RuntimeError("caller refused the hop")

        browser = _FakeBrowser(
            paused_events=[_request_paused("https://example.com/next")],
        )
        _patch_pool(monkeypatch, browser)
        tab = self._run(browser, on_redirect=refuse)
        assert tab.failed_requests == ["req-1"]
        assert tab.continued_requests == []

    def test_a_canonicalized_initial_request_is_not_a_hop(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Chrome normalizes the URL it was given; that is not a redirect.

        A bare host acquires the empty path before it reaches the wire, so
        comparing the paused target against the caller's SPELLING reports the
        very first request as a hop. ``on_redirect`` is raise-to-abort and
        Google's raises on ``/sorry``, so a false hop can abort an ordinary
        fetch -- the same hazard the fragment case already guards.
        """
        browser = _FakeBrowser(
            href="https://example.com/",
            paused_events=[_request_paused("https://example.com/")],
        )
        _patch_pool(monkeypatch, browser)
        seen: list[str] = []
        self._run(browser, url="https://example.com", on_redirect=seen.append)
        assert seen == []

    def test_a_hop_back_to_the_starting_url_is_still_a_hop(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``A -> B -> A`` must report both hops, including the return.

        Comparing each target against the ORIGIN url rather than the previous
        one makes the second real navigation invisible: a redirect chain that
        lands back where it started is exactly how a login bounce behaves, and
        the caller's guard never sees it.
        """
        browser = _FakeBrowser(
            href="https://a.example/start",
            paused_events=[
                _request_paused("https://b.example/next"),
                _request_paused("https://a.example/start"),
            ],
        )
        _patch_pool(monkeypatch, browser)
        seen: list[str] = []
        self._run(browser, url="https://a.example/start", on_redirect=seen.append)
        assert seen == ["https://b.example/next", "https://a.example/start"]

    def test_a_redirect_budget_of_zero_refuses_the_hop(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``max_redirects=0`` must stop the browser leg too.

        ``RetryParams`` documents the knob as "Maximum redirects to follow; 0
        disables" and both header transports honor it (each threads it into its
        own redirect loop). The browser leg cannot even express it, so a caller
        that disabled redirects still had Chrome follow them -- a silently
        weaker contract on the one transport that follows hops itself.
        """
        browser = _FakeBrowser(
            paused_events=[_request_paused("https://public.example/next")],
        )
        _patch_pool(monkeypatch, browser)
        asyncio.run(
            _navigate(
                "https://public.example/start",
                profile_dir=_PROFILE,
                egress="1.2.3.4",
                timeout_sec=5.0,
                headless=True,
                max_redirects=0,
                on_redirect=None,
            ),
        )
        assert browser.last_tab is not None
        assert browser.last_tab.failed_requests == ["req-1"]
        assert browser.last_tab.continued_requests == []

    def test_a_redirect_budget_bounds_the_chain(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One hop is allowed at ``max_redirects=1``; the second is refused."""
        browser = _FakeBrowser(
            paused_events=[
                _request_paused("https://public.example/one", request_id="hop-1"),
                _request_paused("https://public.example/two", request_id="hop-2"),
            ],
        )
        _patch_pool(monkeypatch, browser)
        asyncio.run(
            _navigate(
                "https://public.example/start",
                profile_dir=_PROFILE,
                egress="1.2.3.4",
                timeout_sec=5.0,
                headless=True,
                max_redirects=1,
                on_redirect=None,
            ),
        )
        assert browser.last_tab is not None
        assert browser.last_tab.continued_requests == ["hop-1"]
        assert browser.last_tab.failed_requests == ["hop-2"]

    def test_interception_is_scoped_to_documents(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Only navigations are paused, and that scoping is load-bearing.

        Intercepting every request pauses each subresource until the handler
        answers, and answering costs a DNS resolution on zendriver's connection
        thread; measured against live Google, the page never reached
        ``readyState == "complete"`` and the fetch timed out. Documents are also
        the entire SSRF surface -- a redirect chain is documents, and a
        subresource cannot redirect the navigation anywhere.
        """
        browser = _FakeBrowser()
        _patch_pool(monkeypatch, browser)
        tab = self._run(browser)
        enable = next(
            (
                c
                for c in tab.wire_commands
                if from_plain(c.get("method"), str, default="") == "Fetch.enable"
            ),
            None,
        )
        assert enable is not None
        patterns = from_plain(
            from_plain(enable.get("params"), dict[str, object], default={}).get(
                "patterns",
            ),
            list[object],
            default=[],
        )
        shapes = [from_plain(p, dict[str, object]) for p in patterns]
        assert [from_plain(s.get("resourceType"), str, default="") for s in shapes] == [
            "Document",
        ]
        assert [from_plain(s.get("requestStage"), str, default="") for s in shapes] == [
            "Request",
        ]


def test_navigate_seeds_request_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = _FakeBrowser()
    _patch_pool(monkeypatch, browser)
    asyncio.run(
        _navigate(
            "https://google.com/search?q=x",
            profile_dir=_PROFILE,
            egress="1.2.3.4",
            timeout_sec=5.0,
            headless=True,
            headers={"X-Test": "yes"},
            cookies={"CONSENT": "YES+"},
        ),
    )
    assert len(browser.cookies.seeded) == 1
    seeded = browser.cookies.seeded[0]
    assert isinstance(seeded, network.CookieParam)
    assert seeded.name == "CONSENT"
    assert seeded.value == "YES+"
    assert browser.last_tab is not None
    # Three: the request guard's ``Fetch.enable`` precedes the two header
    # commands, because every tab is guarded whether or not headers are set.
    assert len(browser.last_tab.commands) == 3


def test_navigate_timeout_includes_browser_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SlowPool:
        async def browser(
            self,
            egress: str,
            profile_dir: Path,
            *,
            headless: bool,
        ) -> _FakeBrowser:
            del egress, profile_dir, headless
            await asyncio.Event().wait()
            raise AssertionError("Browser acquisition escaped the timeout.")

    slow_pool = _SlowPool()
    monkeypatch.setattr(
        zendriver,
        "_pool",
        lambda: slow_pool,
    )
    with pytest.raises(TimeoutError):
        asyncio.run(
            _navigate(
                "https://example.com/",
                profile_dir=_PROFILE,
                egress="e",
                timeout_sec=0.001,
                headless=True,
                on_redirect=None,
            ),
        )


class _WedgedTab(_FakeTab):
    """A tab whose navigation AND close both never return.

    Models one wedged CDP connection: ``Tab.get`` waits on a load event that
    never arrives, and ``Tab.close`` then sends ``Target.closeTarget`` and
    awaits a reply with no ceiling of its own (only the ``TargetDestroyed``
    wait AFTER it is bounded, at 10s).
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


def test_navigate_reports_its_own_timeout_when_teardown_also_wedges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled navigation must still finish, so the caller sees the budget.

    An ``asyncio.timeout`` scope delivers exactly ONE cancellation. When it
    lands on the navigation, ``_navigate_tab``'s ``except BaseException: await
    tab.close()`` runs with that cancellation already spent -- so an unbounded
    close parks there forever and the coroutine NEVER completes. Its
    ``TimeoutError`` is therefore never raised, and the caller instead waits out
    ``fetch_zendriver``'s ``timeout_sec + 30`` before ``future.result`` raises a
    bare, wall-less ``TimeoutError``.

    That is the CI shape: the traceback ended at ``concurrent.futures``'
    ``Future.result`` raising a bare ``TimeoutError()`` rather than at
    ``asyncio.timeouts``' ``__aexit__``, which is what a coroutine reporting
    its own deadline produces.
    """
    browser = _WedgedBrowser()
    _patch_pool(monkeypatch, browser)
    monkeypatch.setattr(
        zendriver,
        "_closed",
        partial(zendriver._closed, budget_sec=0.01),
    )

    async def go() -> float:
        before = asyncio.all_tasks()
        started = time.monotonic()
        with pytest.raises(TimeoutError) as error:
            await _navigate(
                "https://example.com/",
                profile_dir=_PROFILE,
                egress="e",
                timeout_sec=0.01,
                headless=True,
                on_redirect=None,
            )
        assert isinstance(error.value.__cause__, asyncio.CancelledError)
        assert asyncio.all_tasks() - before == set(), "the close task was not reaped"
        return time.monotonic() - started

    # Outside ``raises``: an unbounded close must fail on the outer deadline.
    elapsed = asyncio.run(asyncio.wait_for(go(), timeout=1.0))
    assert elapsed < 0.5, f"teardown outlived its budget: {elapsed:.2f}s"
    assert browser.last_tab is not None
    assert browser.last_tab.navigations == ["https://example.com/"]
    assert browser.last_tab.closed


def test_navigate_uses_one_overall_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every step wait must shrink against ONE deadline, never restart it.

    A CHALLENGE page, not a clear one: ``_settled_content`` returns on its
    first read when the document is already clear, so a clear page never
    reaches the two step waits and the assertion below has nothing to observe.
    Measured on the previous fixture: zero ``wait_for`` calls, i.e. the test
    passed without exercising the invariant it names.
    """
    browser = _FakeBrowser(
        documents=["<html><title>Just a moment...</title></html>", "<html>ok</html>"],
    )
    _patch_pool(monkeypatch, browser)

    budgets: list[float] = []
    real_wait_for = asyncio.wait_for

    async def record_budget(
        awaitable: Awaitable[object],
        timeout: float | None = None,  # noqa: ASYNC109 -- The browser test timeout bounds polling across an awaited operation.
    ) -> object:
        # A per-step timeout that RESET the budget would hand out a constant;
        # one overall deadline yields a strictly shrinking remainder.
        budgets.append(-1.0 if timeout is None else timeout)
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr(asyncio, "wait_for", record_budget)
    asyncio.run(
        _navigate(
            "https://example.com/",
            profile_dir=_PROFILE,
            egress="e",
            timeout_sec=5.0,
            headless=True,
            on_redirect=None,
        ),
    )

    assert len(budgets) >= 2, f"the settle path never reached its step waits: {budgets}"
    assert all(timeout >= 0.0 for timeout in budgets), (
        f"a step wait was left unbounded: {budgets}"
    )
    assert budgets == sorted(budgets, reverse=True), (
        f"step waits did not shrink against one deadline: {budgets}"
    )
    assert max(budgets) <= 5.0 / 2, f"a step wait exceeded the settle budget: {budgets}"


def test_navigate_opens_blank_tab_before_requested_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://gated.example/page?q=x"
    browser = _FakeBrowser()
    _patch_pool(monkeypatch, browser)

    asyncio.run(
        _navigate(
            url,
            profile_dir=_PROFILE,
            egress="e",
            timeout_sec=5.0,
            headless=True,
            on_redirect=None,
        ),
    )

    assert browser.gets == ["about:blank"]
    assert browser.last_tab is not None
    assert browser.last_tab.navigations == [url]


def test_navigate_returns_rendered_page_without_semantic_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = '<html><div id="cf_chl_widget"></div></html>'
    browser = _FakeBrowser(content=body)
    _patch_pool(monkeypatch, browser)

    result = asyncio.run(
        _navigate(
            "https://example.com/",
            profile_dir=_PROFILE,
            egress="e",
            # Small: this body IS challenge markup, so the settle poll waits out
            # its whole budget (half the timeout) before giving up. The exact
            # budget is irrelevant to what this asserts -- that the transport
            # returns the page rather than classifying it -- so keep it cheap.
            timeout_sec=0.02,
            headless=True,
            on_redirect=None,
        ),
    )

    assert result.body == body.encode()


def test_navigate_waits_out_a_cloudflare_interstitial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The whole point of the browser transport is clearing a JS challenge, and
    # it captured the challenge instead. The interstitial reaches readyState
    # "complete" on its own -- it IS a loaded document -- and only then does its
    # JS navigate to the real page. Harvesting at the first "complete" returns
    # the 5KB "Just a moment..." wall every time, so every Cloudflare-walled
    # site failed through the one transport meant to clear it (measured live:
    # 5516 bytes at complete, 380404 bytes after the handoff).
    real = "<html><title>Real Page</title>body</html>"
    browser = _FakeBrowser(
        documents=["<html><title>Just a moment...</title></html>", real],
    )
    _patch_pool(monkeypatch, browser)

    result = asyncio.run(
        _navigate(
            "https://walled.example/",
            profile_dir=_PROFILE,
            egress="e",
            timeout_sec=30.0,
            headless=True,
            on_redirect=None,
        ),
    )

    assert result.body == real.encode(), (
        "browser transport returned the Cloudflare interstitial, not the page "
        "it exists to unwrap"
    )


def test_navigate_waits_for_the_new_document_to_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The second trap, and the subtler one. A main-frame navigation COMMITS the
    # new document before it parses, so a read taken right after the event sees
    # a bare <head>: challenge markup gone, correct <title>, no body. Live that
    # was 386 bytes between the 5516-byte wall and the 380404-byte page -- and
    # it looks like success, which is exactly why it needs its own test.
    real = "<html><title>Real Page</title>the whole body</html>"
    browser = _FakeBrowser(
        documents=["<html><title>Just a moment...</title></html>", real],
        parsing="<html><title>Real Page</title></html>",
    )
    _patch_pool(monkeypatch, browser)

    result = asyncio.run(
        _navigate(
            "https://walled.example/",
            profile_dir=_PROFILE,
            egress="e",
            timeout_sec=30.0,
            headless=True,
            on_redirect=None,
        ),
    )

    assert result.body == real.encode(), (
        "harvested the document mid-parse: right title, empty body"
    )


def test_navigate_returns_promptly_when_no_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The settle wait must cost an unchallenged page nothing: a plain document
    # is harvested on the FIRST read, with no re-poll. Otherwise every fetch
    # pays the challenge budget.
    browser = _FakeBrowser(content="<html><title>Plain</title>ok</html>")
    _patch_pool(monkeypatch, browser)

    result = asyncio.run(
        _navigate(
            "https://example.com/",
            profile_dir=_PROFILE,
            egress="e",
            timeout_sec=30.0,
            headless=True,
            on_redirect=None,
        ),
    )

    assert result.body == b"<html><title>Plain</title>ok</html>"
    assert browser.last_tab is not None
    assert browser.last_tab.content_reads == 1


def test_navigate_gives_up_on_an_unclearable_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A challenge that never clears (a real block, not an interstitial) must
    # return what Chrome rendered so the caller's classifier raises its specific
    # BotDetectionError -- never hang until the fetch timeout.
    walled = "<html><title>Just a moment...</title></html>"
    browser = _FakeBrowser(content=walled)
    _patch_pool(monkeypatch, browser)

    result = asyncio.run(
        _navigate(
            "https://walled.example/",
            profile_dir=_PROFILE,
            egress="e",
            # Deliberately small: giving up is what this asserts, and the budget
            # is real time. A production-sized 30s would sleep 15s per run.
            timeout_sec=0.02,
            headless=True,
            on_redirect=None,
        ),
    )

    assert result.body == walled.encode()


def test_navigate_allows_embedded_captcha_on_rendered_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = _FakeBrowser(content='<html><div class="g-recaptcha"></div></html>')
    _patch_pool(monkeypatch, browser)

    result = asyncio.run(
        _navigate(
            "https://example.com/login",
            profile_dir=_PROFILE,
            egress="e",
            timeout_sec=5.0,
            headless=True,
            on_redirect=None,
        ),
    )

    assert result.body == b'<html><div class="g-recaptcha"></div></html>'


def test_navigate_closes_tab_after_returning_rendered_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = _FakeBrowser(content="<html><title>Just a moment...</title></html>")
    _patch_pool(monkeypatch, browser)
    asyncio.run(
        _navigate(
            "https://walled.example/",
            profile_dir=_PROFILE,
            egress="e",
            timeout_sec=0.02,  # An unclearable wall; see the note above.
            headless=True,
            on_redirect=None,
        ),
    )
    assert browser.last_tab is not None
    assert browser.last_tab.closed is True


def test_navigate_matches_exact_host_cookie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    browser = _FakeBrowser(
        cookies=[_FakeCookie(name="H", value="1", domain="example.com")],
    )
    _patch_pool(monkeypatch, browser)
    result = asyncio.run(
        _navigate(
            "https://example.com/",
            profile_dir=_PROFILE,
            egress="e",
            timeout_sec=5.0,
            headless=True,
            on_redirect=None,
        ),
    )
    assert result.cookies == {"H": "1"}


def test_settled_content_returns_promptly_when_the_wall_clears_off_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A challenge that clears must be observed then, not at the deadline.

    zendriver dispatches sync handlers off-loop, and a cross-thread
    ``Event.set()`` does not wake the selector. Asserted on elapsed time: the
    body is identical either way.
    """
    wall = "<html><title>Just a moment...</title></html>"
    handlers: list[Callable[..., None]] = []
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

    class _ClearsOffLoopTab:
        """Serves the wall until an off-loop navigation event says otherwise.

        Deliberately not ``_FakeTab``: that one advances its document from
        inside ``get_content``, on the loop thread, which is the arrangement
        that hides a wakeup delivered from anywhere else.
        """

        def add_handler(
            self,
            event_type: type[object],
            handler: Callable[..., None],
        ) -> None:
            del event_type
            handlers.append(handler)

        async def get_content(self) -> str:
            return "<html>ok</html>" if cleared.is_set() else wall

        async def wait_for_ready_state(self, until: str = "complete") -> bool:
            del until
            return True

    def clear_from_another_thread() -> None:
        assert waiting.wait(timeout=1.0), "the settle wait never blocked"
        cleared.set()
        handlers[0](_main_frame_navigated())

    async def go() -> float:
        started = time.monotonic()
        thread = threading.Thread(target=clear_from_another_thread)
        thread.start()
        try:
            body = await zendriver._settled_content(
                cast(Tab, _ClearsOffLoopTab()),
                budget_sec=1.0,
            )
            assert body == "<html>ok</html>"
            return time.monotonic() - started
        finally:
            thread.join(timeout=1.0)
            assert not thread.is_alive()

    with asyncio.Runner(
        loop_factory=partial(asyncio.SelectorEventLoop, selector=selector),
    ) as runner:
        assert runner.run(go()) < 0.5


def test_settled_content_bounds_a_stalled_document_parse() -> None:
    """``budget_sec`` must bound the whole settle, parse included.

    A document that commits near the deadline and then stalls would otherwise
    spend the caller's entire request timeout in ``wait_for_ready_state``.
    """
    wall = "<html><title>Just a moment...</title></html>"

    class _StalledParseTab:
        def add_handler(
            self,
            event_type: object,
            handler: Callable[[object], None],
        ) -> None:
            del event_type
            # Commit a navigation immediately, so the settle loop always
            # advances to the ready-state wait that has no ceiling.
            handler(_main_frame_navigated())

        async def get_content(self) -> str:
            return wall

        async def wait_for_ready_state(self, until: str = "complete") -> bool:
            del until
            await asyncio.Event().wait()  # Parses forever.
            raise AssertionError("unreachable")

    async def go() -> float:
        started = time.monotonic()
        body = await zendriver._settled_content(
            cast(Tab, _StalledParseTab()),
            budget_sec=0.01,
        )
        assert body == wall
        return time.monotonic() - started

    # Generous multiple of the budget: this must fail on an UNBOUNDED wait, not
    # on scheduler jitter around a bound that is working.
    assert asyncio.run(asyncio.wait_for(go(), timeout=1.0)) < 0.5


# -- _navigate: redirect callback --------------------------------------------


def test_navigate_fires_on_redirect_per_hop_not_on_the_landing_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The callback reports each hop BEFORE it is followed, not the final URL.

    Reading ``document.location.href`` after the load could only ever report
    where the page ENDED UP -- one notification, after every hop had already
    been fetched, which is unusable for the abort ``ObserveParams`` promises.
    The hops themselves are now observed, so a fetch that merely lands
    elsewhere without an intercepted document request reports nothing.
    """
    browser = _FakeBrowser(
        href="https://example.com/landing",
        paused_events=[_request_paused("https://example.com/landing")],
    )
    _patch_pool(monkeypatch, browser)
    seen: list[str] = []
    asyncio.run(
        _navigate(
            "https://example.com/start",
            profile_dir=_PROFILE,
            egress="e",
            timeout_sec=5.0,
            headless=True,
            on_redirect=seen.append,
        ),
    )
    assert seen == ["https://example.com/landing"]


def test_navigate_treats_fragment_only_difference_as_no_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fragment is never sent, so its absence on the wire is not a hop.

    ``on_redirect`` is documented raise-to-abort, and Google's callback raises
    on ``/sorry`` -- so a false hop on the initial navigation aborts an ordinary
    fetch.
    """
    url = "https://example.com/page#section"
    browser = _FakeBrowser(
        href=url,
        paused_events=[_request_paused("https://example.com/page")],
    )
    _patch_pool(monkeypatch, browser)
    seen: list[str] = []
    asyncio.run(
        _navigate(
            url,
            profile_dir=_PROFILE,
            egress="e",
            timeout_sec=5.0,
            headless=True,
            on_redirect=seen.append,
        ),
    )
    assert seen == []


def test_navigate_no_redirect_when_url_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    url = "https://example.com/x"
    browser = _FakeBrowser(href=url)
    _patch_pool(monkeypatch, browser)
    seen: list[str] = []
    asyncio.run(
        _navigate(
            url,
            profile_dir=_PROFILE,
            egress="e",
            timeout_sec=5.0,
            headless=True,
            on_redirect=seen.append,
        ),
    )
    assert seen == []


# -- _BrowserPool: reuse + replacement ---------------------------------------


def test_pool_reuses_browser_per_key(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[_FakeBrowser] = []

    async def fake_launch(
        self: _BrowserPool,
        profile_dir: Path,
        *,
        headless: bool,
    ) -> _FakeBrowser:
        del self, profile_dir, headless
        b = _FakeBrowser()
        launched.append(b)
        return b

    monkeypatch.setattr(_BrowserPool, "_launch", fake_launch)
    pool = _BrowserPool(serve_control=False)
    try:

        async def go() -> bool:
            a = await pool.browser("e", _PROFILE, headless=True)
            b = await pool.browser("e", _PROFILE, headless=True)
            return a is b

        assert pool.run(go())
        assert len(launched) == 1
    finally:
        pool.shutdown()


def test_pool_rejects_mode_change_for_live_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[_FakeBrowser] = []

    async def fake_launch(
        self: _BrowserPool,
        profile_dir: Path,
        *,
        headless: bool,
    ) -> _FakeBrowser:
        del self, profile_dir, headless
        browser = _FakeBrowser()
        launched.append(browser)
        return browser

    monkeypatch.setattr(_BrowserPool, "_launch", fake_launch)
    pool = _BrowserPool(serve_control=False)
    try:

        async def go() -> None:
            await pool.browser("e", _PROFILE, headless=True)
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
                await pool.browser("e", _PROFILE, headless=False)

        pool.run(go())
        assert len(launched) == 1
    finally:
        pool.shutdown()


def test_pool_serializes_concurrent_mode_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[_FakeBrowser] = []

    async def fake_launch(
        self: _BrowserPool,
        profile_dir: Path,
        *,
        headless: bool,
    ) -> _FakeBrowser:
        del self, profile_dir, headless
        await asyncio.sleep(0)
        browser = _FakeBrowser()
        launched.append(browser)
        return browser

    monkeypatch.setattr(_BrowserPool, "_launch", fake_launch)
    pool = _BrowserPool(serve_control=False)
    try:

        async def go() -> tuple[object, object]:
            return await asyncio.gather(
                pool.browser("e", _PROFILE, headless=True),
                pool.browser("e", _PROFILE, headless=False),
                return_exceptions=True,
            )

        results = pool.run(go())
        assert len(launched) == 1
        assert sum(isinstance(result, RuntimeError) for result in results) == 1
    finally:
        pool.shutdown()


def test_pool_relaunches_stopped_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[_FakeBrowser] = []

    async def fake_launch(
        self: _BrowserPool,
        profile_dir: Path,
        *,
        headless: bool,
    ) -> _FakeBrowser:
        del self, profile_dir, headless
        b = _FakeBrowser()
        launched.append(b)
        return b

    monkeypatch.setattr(_BrowserPool, "_launch", fake_launch)
    pool = _BrowserPool(serve_control=False)
    try:

        async def go() -> None:
            first = await pool.browser("e", _PROFILE, headless=True)
            cast(_FakeBrowser, first).stopped = True  # Simulate Chrome exit.
            second = await pool.browser("e", _PROFILE, headless=True)
            assert second is not first

        pool.run(go())
        assert len(launched) == 2
    finally:
        pool.shutdown()


def test_pool_run_gives_up_when_its_loop_stopped() -> None:
    """A stopped loop must surface a timeout, never an unbounded wait.

    The coroutine's own ``asyncio.timeout`` bounds the fetch only while the loop
    is running it; a stopped loop never schedules it, so nothing arms.
    """
    pool = _BrowserPool(serve_control=False)
    try:
        pool._loop.call_soon_threadsafe(pool._loop.stop)
        # Joining the thread is what proves the loop is DONE running, rather
        # than sampling ``is_running`` in a spin that can observe the gap
        # between the callback firing and the loop actually stopping.
        pool._thread.join(timeout=5)
        assert not pool._thread.is_alive()

        async def never_scheduled() -> str:
            raise AssertionError("A stopped loop must not run the coroutine.")

        with pytest.raises(TimeoutError):
            pool.run(never_scheduled(), timeout_sec=0.05)
    finally:
        pool.shutdown()


def test_fetch_zendriver_bounds_its_wait_above_the_navigate_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The outer ceiling must exceed the coroutine's own budget.

    At or below it the outer wait fires first, cancelling a ``_navigate`` that
    was about to report a wall and turning it into an opaque timeout.
    """
    waits: list[float] = []

    class _RecordingPool:
        def run(
            self,
            coro: Coroutine[object, object, object],
            *,
            timeout_sec: float = 0,
        ) -> BrowserResult:
            coro.close()
            waits.append(timeout_sec)
            return BrowserResult(body=b"", cookies={}, final_url="")

    pool = _RecordingPool()
    monkeypatch.setattr(zendriver, "_pool", lambda: pool)
    zendriver.fetch_zendriver(
        "https://example.com/",
        profile_dir=_PROFILE,
        egress="e",
        timeout_sec=30.0,
    )

    assert waits == [pytest.approx(60.0)]
    close_budget = cast(
        object,
        inspect.signature(zendriver._closed).parameters["budget_sec"].default,
    )
    assert isinstance(close_budget, float)
    assert 0 < close_budget < waits[0] - 30.0


def test_launch_survives_a_reply_to_a_cancelled_cdp_transaction(tmp_path: Path) -> None:
    """A CDP reply arriving after its transaction was cancelled must be dropped.

    ``Transaction.__call__`` sets the result unconditionally, so a reply landing
    on a cancelled future raises ``InvalidStateError`` inside
    ``Listener.listener_loop`` and kills the listener for the whole connection.
    Every fetch here is cancellable (``_navigate`` wraps the navigation in
    ``asyncio.timeout``) and Chrome answers the in-flight ``Page.navigate``
    afterwards, so a timed-out fetch leaves the pooled browser deaf to every
    later one.

    Driven through ``_launch_browser`` rather than by calling the patcher
    directly: arming it on the launch path is the contract -- a fetch must never
    reach live CDP traffic with the vendor's unguarded ``__call__`` in place.
    """

    async def fake_start(config: object) -> _FakeBrowser:
        del config
        return _FakeBrowser()

    async def go() -> None:
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr("zendriver.start", fake_start)
            # Restored to the VENDOR's own ``__call__``, undoing any arming an
            # earlier test in this process did: the guard is installed
            # class-wide, so without this the assertions below pass vacuously.
            patcher.setattr(Transaction, "__call__", _VENDOR_TRANSACTION_CALL)
            await zendriver._launch_browser(
                tmp_path,
                headless=True,
            )

            # ``result`` alone, though the listener splats the whole message:
            # the vendor reads only ``error`` and ``result``, and its
            # ``**response: dict[str, object]`` annotation rejects the integer
            # ``id`` a real reply also carries.
            cancelled = Transaction(page.navigate("about:blank"))
            cancelled.cancel()
            cancelled(result={"frameId": "F", "loaderId": "L"})
            assert cancelled.cancelled()

            # The guard must drop only what is already settled: a reply to a
            # LIVE transaction still has to reach the caller awaiting it.
            live = Transaction(page.navigate("about:blank"))
            live(result={"frameId": "F", "loaderId": "L"})
            assert (live.result())[0] == (page.FrameId("F"))

    asyncio.run(go())


def test_pool_shutdown_joins_thread_and_closes_loop() -> None:
    pool = _BrowserPool(serve_control=False)
    pool.shutdown()

    assert not pool._thread.is_alive()
    assert pool._loop.is_closed()


def test_pool_shutdown_bounds_teardown_across_all_browsers() -> None:
    """The teardown budget is TOTAL, not per browser.

    Per browser, N wedged browsers would take N times the ceiling, and a
    supervisor following SIGTERM with SIGKILL would cut teardown short and
    leak exactly what this exists to close. Whatever the budget cannot close
    politely is killed instead.
    """
    pool = _BrowserPool(serve_control=False)
    wedged = [_HangingStopBrowser() for _ in range(3)]
    processes = [_FakeProcess() for _ in wedged]
    for index, browser in enumerate(wedged):
        browser._process = processes[index]
        pool._browsers[("egress", f"/profile/{index}")] = (
            True,
            cast(Browser, browser),
        )

    pool.shutdown(budget_sec=0.01)

    # Attempts, not elapsed time: three browsers on a per-browser budget also
    # finish promptly, so a timing bound passes on the bug this pins. The
    # first browser is skipped because whether its ``stop()`` starts before
    # the budget expires is a scheduling race; the rest can never be asked.
    assert [browser.stop_calls for browser in wedged[1:]] == [0, 0], (
        "the budget was spent per browser rather than across all of them"
    )
    assert [process.kills for process in processes] == [1, 1, 1], (
        "a browser left open when the budget expired was not killed"
    )


def test_pool_keys_separate_egress(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[_FakeBrowser] = []

    async def fake_launch(
        self: _BrowserPool,
        profile_dir: Path,
        *,
        headless: bool,
    ) -> _FakeBrowser:
        del self, profile_dir, headless
        b = _FakeBrowser()
        launched.append(b)
        return b

    monkeypatch.setattr(_BrowserPool, "_launch", fake_launch)
    pool = _BrowserPool(serve_control=False)
    try:

        async def go() -> None:
            await pool.browser("egress-a", _PROFILE, headless=True)
            await pool.browser("egress-b", _PROFILE, headless=True)

        pool.run(go())
        assert len(launched) == 2  # Distinct egress -> distinct browser.
    finally:
        pool.shutdown()


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
    monkeypatch.setattr(
        zendriver,
        "cache_dir",
        lambda: tmp_path / "cache",
    )
    return puppeteer, playwright


def test_non_macos_hosts_need_no_separate_browser() -> None:
    """Linux and Windows keep the ordinary browser -- there is no capture.

    A URL there reaches a browser through ``xdg-open`` and the desktop file
    rather than a running process, and a headless server has only the stock
    Chrome installed anyway.
    """
    assert zendriver._fetch_browser(platform="linux") == ""
    assert zendriver._fetch_browser(platform="win32") == ""


def test_non_macos_hosts_launch_with_no_extra_flags() -> None:
    # Chrome on a headless server has no keychain to mock, so a flag leaking
    # onto that path would break the only browser it has.
    assert (
        zendriver._fetch_browser_args(
            zendriver._fetch_browser(platform="linux"),
        )
        == []
    )


def test_no_install_falls_back_to_zendriver(roots: tuple[Path, Path]) -> None:
    # A miss must cost click capture, never the fetch: "" means "let zendriver
    # find Chrome".
    del roots

    assert zendriver._fetch_browser(platform="darwin") == ""


def test_a_puppeteer_install_is_found(roots: tuple[Path, Path]) -> None:
    expected = _fetch_browser_install(roots[0], "mac_arm-147.0.7727.57")

    assert zendriver._fetch_browser(
        platform="darwin",
    ) == str(expected)


def test_a_playwright_install_is_found(roots: tuple[Path, Path]) -> None:
    expected = _fetch_browser_install(roots[1], "chromium-1217")

    assert zendriver._fetch_browser(
        platform="darwin",
    ) == str(expected)


def test_the_newest_build_wins(roots: tuple[Path, Path]) -> None:
    # Builds accumulate across upgrades; a stale one eventually cannot read a
    # profile the current browser wrote.
    _ = _fetch_browser_install(roots[0], "mac_arm-131.0.6778.85")
    newest = _fetch_browser_install(roots[0], "mac_arm-147.0.7727.57")

    assert zendriver._fetch_browser(
        platform="darwin",
    ) == str(newest)


@pytest.mark.parametrize(
    ("older", "newer"),
    [
        ("mac_arm-99.0.4844.51", "mac_arm-100.0.4896.60"),
        ("chromium-999", "chromium-1000"),
    ],
)
def test_build_order_is_numeric_not_lexical(
    roots: tuple[Path, Path],
    older: str,
    newer: str,
) -> None:
    # Text comparison ranks "99" above "100", which silently pins fetches to
    # the older browser at every version rollover past a digit boundary.
    _ = _fetch_browser_install(roots[0], older)
    expected = _fetch_browser_install(roots[0], newer)

    assert zendriver._fetch_browser(
        platform="darwin",
    ) == str(expected)


def test_unversioned_build_directories_still_order_deterministically() -> None:
    # Filesystem iteration order is arbitrary, so names carrying no digits
    # must not leave the choice to it. Ranked directly rather than through a
    # fixture: a fixture would have to observe the very order in question.
    unversioned = [Path("beta"), Path("alpha"), Path("dev")]

    ranked = sorted(
        unversioned,
        key=zendriver._build_order,
        reverse=True,
    )

    assert [path.name for path in ranked] == ["dev", "beta", "alpha"]


def test_a_directory_without_the_binary_is_skipped(roots: tuple[Path, Path]) -> None:
    # Playwright's roots include ``ffmpeg-*`` and other non-browser payloads,
    # and an interrupted download leaves a build directory with no binary.
    (roots[1] / "ffmpeg-1011").mkdir(parents=True)
    expected = _fetch_browser_install(roots[1], "chromium-1217")

    assert zendriver._fetch_browser(
        platform="darwin",
    ) == str(expected)


def test_an_intel_download_is_found(roots: tuple[Path, Path]) -> None:
    # The only build on an Intel Mac, and runnable under Rosetta on Apple
    # silicon, so keying discovery to the host arch would skip a usable one.
    expected = _fetch_browser_install(
        roots[0],
        "mac-147.0.7727.57",
        arch="chrome-mac-x64",
    )

    assert zendriver._fetch_browser(
        platform="darwin",
    ) == str(expected)


def test_chrome_for_testing_gets_the_mock_keychain_flag() -> None:
    """Pinned verbatim: a typo yields a silent hang, not an error.

    Chrome ignores an unknown flag rather than rejecting it, so a misspelling
    restores the keychain prompt and the launch blocks behind it.
    """
    assert zendriver._fetch_browser_args(
        "/cache/ChromeForTesting",
    ) == ["--use-mock-keychain"]


def test_stock_chrome_keeps_its_own_keychain() -> None:
    # It owns a real keychain entry, and mocking that cuts it off from cookies
    # it legitimately has.
    assert zendriver._fetch_browser_args("") == []


def test_no_prose_cites_a_line_number_in_this_package() -> None:
    """A ``file:NNN`` citation in prose rots the moment anything shifts.

    Measured twice. First, four files cited one line of this module for the
    pool's ``timeout_sec + 30``; inserting the browser-discovery helpers above
    it moved that code down ~100 lines and left the cited line BLANK. Then a
    narrower version of THIS guard -- matching only Python files -- passed while
    two ``pyproject`` citations stayed wrong in the files it had just swept: one
    claimed ``addopts`` and landed on a GitHub marker, the other claimed
    ``--dist=worksteal`` and landed on the ``real_llm`` marker.

    A line number is correct for exactly one commit and nothing re-checks it,
    so name the SYMBOL or the setting instead -- those survive every edit that
    does not rename them.

    Matched on the repo's OWN file kinds rather than any ``word.word:digits``,
    which would sweep in ``example.com:443`` and every ``127.0.0.1:8888``: a
    host and port is not a citation and does not rot. The extension list is the
    thing that must stay honest -- it is what let the two ``.toml`` citations
    through -- so it names every kind this repo actually cites.

    Scoped to wesearch, where the drift happened and the convention is being
    established; the repo-wide sweep is a separate change.
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


def test_add_arguments_registers_optional_url() -> None:
    parser = argparse.ArgumentParser()
    zendriver._add_arguments(parser)
    default_flags = cast(_Flags, parser.parse_args([]))
    explicit_flags = cast(
        _Flags,
        parser.parse_args(["https://example.com/"]),
    )
    assert default_flags.url == "about:blank"
    assert explicit_flags.url == "https://example.com/"


def test_main_prints_lifecycle_and_opens_url(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(zendriver, "open_instance", opened.append)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch-zendriver", "https://example.com/"],
    )
    assert zendriver.main() == 0
    assert opened == ["https://example.com/"]
    assert capsys.readouterr().out == (
        "Opening https://example.com/ in Chrome on "
        f"{data_dir() / 'rekursiv-ai' / 'wesearch' / 'fetch-zendriver'}"
        " -- close the window when done.\nWindow closed.\n"
    )


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_fetch_browser_candidates_skip_non_macos(platform: str) -> None:
    assert zendriver._fetch_browser(platform=platform) == ""


def test_close_browser_on_port_starts_and_stops_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[_FakeBrowser] = []

    async def start(
        *,
        host: str,
        port: int,
    ) -> _FakeBrowser:
        assert host == "127.0.0.1"
        assert port == 9222
        browser = _FakeBrowser()
        created.append(browser)
        return browser

    monkeypatch.setattr("zendriver.start", start)
    asyncio.run(zendriver._close_browser_on_port(9222))
    assert len(created) == 1
    assert created[0].stop_calls == 1


def test_command_flag_uses_first_marker_and_stops_at_next_flag() -> None:
    command = "--user-data-dir=/first --user-data-dir=/second --remote-debugging-port=9"
    assert zendriver._command_flag(command, "--user-data-dir=") == "/first"
    assert zendriver._command_flag(command, "--missing=") == ""


def test_command_flag_preserves_values_containing_single_dashes() -> None:
    assert (
        zendriver._command_flag(
            "--flag=value-with-dash --other=x",
            "--flag=",
        )
        == "value-with-dash"
    )


def test_control_address_digest_is_profile_specific_and_fixed_width() -> None:
    first = zendriver._control_address(Path("profile-a"), platform="linux")
    second = zendriver._control_address(Path("profile-b"), platform="linux")
    prefix = chr(0) + "loop-zendriver-"
    assert first.startswith(prefix)
    assert len(first.removeprefix(prefix)) == 24
    assert first != second


def test_process_command_reads_proc_bytes_and_flattens_nuls(tmp_path: Path) -> None:
    command = tmp_path / "proc" / "12" / "cmdline"
    command.parent.mkdir(parents=True)
    command.write_bytes(b"chrome\x00--flag=x\x00")
    assert (
        zendriver._process_command(
            12,
            proc_root=tmp_path / "proc",
            platform="linux",
        )
        == "chrome --flag=x "
    )


def test_process_command_reports_failed_macos_ps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def run(
        args: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del args, capture_output, text, check
        return subprocess.CompletedProcess([], 1, stdout="")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(OSError, match="Could not inspect process 12"):
        zendriver._process_command(
            12,
            proc_root=tmp_path / "proc",
            platform="darwin",
        )


def test_devtools_port_parses_owner_and_command_port_exactly(tmp_path: Path) -> None:
    profile = tmp_path / "my-profile"
    proc = tmp_path / "proc" / "123"
    profile.mkdir()
    proc.mkdir(parents=True)
    (profile / "SingletonLock").symlink_to("host-owner-123")
    (proc / "cmdline").write_text(
        f"chrome --user-data-dir={profile} "
        "--remote-debugging-port=4567 extra-token another-token",
    )
    assert zendriver._devtools_port(profile, proc_root=tmp_path / "proc") == 4567


def test_devtools_port_reads_active_port_when_command_has_no_port(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"
    proc = tmp_path / "proc" / "123"
    profile.mkdir()
    proc.mkdir(parents=True)
    (profile / "SingletonLock").symlink_to("host-123")
    (proc / "cmdline").write_text(
        f"chrome --user-data-dir={profile} --remote-debugging-port=0",
    )
    (profile / "DevToolsActivePort").write_text("4567\n/devtools/browser/id\n")
    assert zendriver._devtools_port(profile, proc_root=tmp_path / "proc") == 4567


def test_domain_cookies_handles_empty_host_domain_and_value() -> None:
    browser = _FakeBrowser(
        cookies=[
            _FakeCookie(name="EMPTY", value="", domain="example.com"),
            _FakeCookie(name="NONE", value="v", domain=""),
            _FakeCookie(name="BOUNDARY", value="bad", domain="ample.com"),
        ],
    )
    assert asyncio.run(
        zendriver._domain_cookies(cast(Browser, browser), "https://example.com/"),
    ) == {
        "EMPTY": "",
    }
    assert (
        asyncio.run(zendriver._domain_cookies(cast(Browser, browser), "about:blank"))
        == {}
    )


def test_fetch_browser_logs_when_no_candidate(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(zendriver, "_fetch_browser_candidates", lambda: iter(()))
    with caplog.at_level("DEBUG", logger=zendriver.logger.name):
        assert zendriver._fetch_browser(platform="darwin") == ""
    assert caplog.records[-1].message == (
        "no Chrome for Testing found; falling back to zendriver's Chrome"
    )


def test_fetch_browser_candidates_yields_all_supported_app_shapes(
    roots: tuple[Path, Path],
) -> None:
    _fetch_browser_install(
        roots[0],
        "build",
        name="Chromium",
        arch="chrome-mac-x64",
    )
    candidates = list(zendriver._fetch_browser_candidates())
    assert (
        roots[0]
        / "build"
        / "chrome-mac-x64"
        / "Chromium.app"
        / "Contents"
        / "MacOS"
        / "Chromium"
        in candidates
    )


def test_kill_browser_process_handles_browser_without_process_attribute() -> None:
    class BrowserWithoutProcess:
        pass

    zendriver._kill_browser_process(cast(Browser, BrowserWithoutProcess()))


def test_kill_browser_process_swallows_process_oserror() -> None:
    class RaisingProcess(_FakeProcess):
        @override
        def kill(self) -> None:
            raise OSError("already gone")

    browser = _FakeBrowser()
    browser._process = RaisingProcess()
    zendriver._kill_browser_process(cast(Browser, browser))


def test_main_frame_navigations_ignores_child_and_foreign_events() -> None:
    class FakeNavigationTab:
        def __init__(self) -> None:
            self.registered: Callable[[object], None] | None = None

        def add_handler(
            self,
            event_type: object,
            handler: Callable[[object], None],
        ) -> None:
            assert event_type is page.FrameNavigated
            self.registered = handler

    async def go() -> bool:
        tab = FakeNavigationTab()
        event = zendriver._main_frame_navigations(cast(Tab, tab))
        assert tab.registered is not None
        tab.registered(object())
        child_frame = _main_frame_navigated()
        child_frame.frame.parent_id = page.FrameId("parent")
        tab.registered(child_frame)
        assert not event.is_set()
        tab.registered(_main_frame_navigated())
        await asyncio.sleep(0)
        return event.is_set()

    assert asyncio.run(go())


def test_wire_url_removes_fragment_and_supplies_root_path() -> None:
    assert zendriver._wire_url("https://example.com#frag") == "https://example.com/"
    assert zendriver._wire_url("https://example.com/a#frag") == "https://example.com/a"


def test_close_orphan_browser_returns_when_no_devtools_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_port(profile: Path) -> int | None:
        del profile
        return None

    monkeypatch.setattr(zendriver, "_devtools_port", no_port)
    zendriver._close_orphan_browser(_PROFILE)


def test_close_orphan_browser_closes_a_reachable_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Connection:
        def close(self) -> None:
            pass

    calls = iter([Connection(), OSError("closed")])

    def port(profile: Path) -> int:
        del profile
        return 9222

    def create_connection(
        address: tuple[str, int],
        timeout: float,
    ) -> Connection | OSError:
        del address, timeout
        connection = next(calls)
        if isinstance(connection, OSError):
            raise connection
        return connection

    async def close_browser(port: int) -> None:
        closed.append(port)

    monkeypatch.setattr(zendriver, "_devtools_port", port)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    closed: list[int] = []
    monkeypatch.setattr(zendriver, "_close_browser_on_port", close_browser)
    zendriver._close_orphan_browser(_PROFILE)
    assert closed == [9222]


def test_shutdown_browsers_is_noop_without_a_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(zendriver, "_pool_singleton", None)
    zendriver.shutdown_browsers()
    assert zendriver._pool_singleton is None


def test_shutdown_browsers_clears_and_stops_existing_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Pool:
        def __init__(self) -> None:
            self.calls = 0

        def shutdown(self) -> None:
            self.calls += 1

    pool = Pool()
    monkeypatch.setattr(zendriver, "_pool_singleton", pool)
    zendriver.shutdown_browsers()
    assert pool.calls == 1
    assert zendriver._pool_singleton is None


def test_pool_launch_forwards_profile_and_headless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _BrowserPool(serve_control=False)
    browser = _FakeBrowser()
    captured: list[tuple[Path, bool]] = []

    async def launch(profile: Path, *, headless: bool) -> _FakeBrowser:
        captured.append((profile, headless))
        return browser

    monkeypatch.setattr(zendriver, "_launch_browser", launch)
    try:
        result = pool.run(pool._launch(_PROFILE, headless=False))
    finally:
        pool.shutdown()
    assert result is browser
    assert captured == [(_PROFILE, False)]


def test_pool_browser_releases_unowned_profile_before_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _BrowserPool()
    browser = _FakeBrowser()
    released: list[Path] = []
    ensured: list[Path] = []

    async def launch(profile: Path, *, headless: bool) -> _FakeBrowser:
        assert profile == _PROFILE
        assert headless
        return browser

    monkeypatch.setattr(zendriver, "_request_pool_release", released.append)
    monkeypatch.setattr(pool, "_ensure_control", ensured.append)
    monkeypatch.setattr(pool, "_launch", launch)
    try:
        result = pool.run(pool.browser("egress", _PROFILE, headless=True))
    finally:
        pool.shutdown()
    assert result is browser
    assert released == [_PROFILE]
    assert ensured == [_PROFILE]


def test_pool_browser_does_not_release_owned_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _BrowserPool()
    server = zendriver._PoolControlServer(_PROFILE, lambda: None)
    browser = _FakeBrowser()
    released: list[Path] = []
    pool._controls[zendriver._control_address(_PROFILE)] = server

    async def launch(profile: Path, *, headless: bool) -> _FakeBrowser:
        assert profile == _PROFILE
        assert headless
        return browser

    monkeypatch.setattr(zendriver, "_request_pool_release", released.append)
    monkeypatch.setattr(pool, "_launch", launch)
    try:
        pool.run(pool.browser("egress", _PROFILE, headless=True))
    finally:
        pool.shutdown()
    assert released == []


def test_pool_keys_separate_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[_FakeBrowser] = []

    async def fake_launch(
        self: _BrowserPool,
        profile_dir: Path,
        *,
        headless: bool,
    ) -> _FakeBrowser:
        del self, profile_dir, headless
        browser = _FakeBrowser()
        launched.append(browser)
        return browser

    monkeypatch.setattr(_BrowserPool, "_launch", fake_launch)
    pool = _BrowserPool(serve_control=False)
    try:

        async def go() -> None:
            await pool.browser("e", _PROFILE / "one", headless=True)
            await pool.browser("e", _PROFILE / "two", headless=True)

        pool.run(go())
    finally:
        pool.shutdown()
    assert len(launched) == 2


def test_command_flag_preserves_spaces_before_next_flag() -> None:
    assert (
        zendriver._command_flag(
            "--flag=value with spaces --other=x",
            "--flag=",
        )
        == "value with spaces"
    )


def test_domain_cookies_does_not_invent_a_host_for_invalid_urls() -> None:
    browser = _FakeBrowser(
        cookies=[
            _FakeCookie(name="INVENTED", value="bad", domain="XXXX"),
            _FakeCookie(name="PREFIX", value="bad", domain="xexample.com"),
        ],
    )
    assert (
        asyncio.run(
            zendriver._domain_cookies(cast(Browser, browser), "about:blank"),
        )
        == {}
    )


def test_domain_cookies_strips_only_leading_cookie_dots() -> None:
    browser = _FakeBrowser(
        cookies=[_FakeCookie(name="COOKIE", value="ok", domain=".example.com")],
    )
    assert asyncio.run(
        zendriver._domain_cookies(cast(Browser, browser), "https://example.com/"),
    ) == {"COOKIE": "ok"}


def test_launch_browser_passes_every_profile_config_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
    monkeypatch.setattr(zendriver, "_fetch_browser", lambda: "/cache/testing")
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    asyncio.run(
        zendriver._launch_browser(tmp_path / "nested" / "profile", headless=True),
    )

    assert captured["headless"] is True
    assert captured["profile"] == str(tmp_path / "nested" / "profile")
    assert captured["sandbox"] is False
    assert captured["executable"] == "/cache/testing"
    assert captured["timeout"] == 0.5
    assert captured["tries"] == 6
    assert "--use-mock-keychain" in cast(list[str], captured["args"])


def test_launch_browser_wraps_start_failure(
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
        asyncio.run(zendriver._launch_browser(tmp_path, headless=True))


def test_fetch_zendriver_forwards_all_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def navigate(
        url: str,
        *,
        profile_dir: Path,
        egress: str,
        timeout_sec: float,
        headless: bool,
        headers: dict[str, str] | None,
        cookies: dict[str, str] | None,
        trust: Trust,
        max_redirects: int,
        on_redirect: Callable[[str], None] | None,
    ) -> BrowserResult:
        captured.update(
            {
                "url": url,
                "profile_dir": profile_dir,
                "egress": egress,
                "timeout_sec": timeout_sec,
                "headless": headless,
                "headers": headers,
                "cookies": cookies,
                "trust": trust,
                "max_redirects": max_redirects,
                "on_redirect": on_redirect,
            },
        )
        return BrowserResult(body=b"body", cookies={}, final_url=url)

    class Pool:
        def run(
            self,
            coro: Coroutine[object, object, BrowserResult],
            *,
            timeout_sec: float = 0,
        ) -> BrowserResult:
            captured["outer_timeout"] = timeout_sec
            return asyncio.run(coro)

    def callback(target: str) -> None:
        del target

    def pool() -> Pool:
        return Pool()

    monkeypatch.setattr(zendriver, "_navigate", navigate)
    monkeypatch.setattr(zendriver, "_pool", pool)
    result = zendriver.fetch_zendriver(
        "https://example.com/start",
        profile_dir=_PROFILE,
        egress="egress",
        timeout_sec=7.0,
        headless=False,
        headers={"X-Test": "yes"},
        cookies={"SID": "cookie"},
        trust="internal",
        max_redirects=2,
        on_redirect=callback,
    )

    assert result == BrowserResult(
        body=b"body",
        cookies={},
        final_url="https://example.com/start",
    )
    assert captured == {
        "url": "https://example.com/start",
        "profile_dir": _PROFILE,
        "egress": "egress",
        "timeout_sec": 7.0,
        "headless": False,
        "headers": {"X-Test": "yes"},
        "cookies": {"SID": "cookie"},
        "trust": "internal",
        "max_redirects": 2,
        "on_redirect": callback,
        "outer_timeout": 37.0,
    }


def test_navigate_tab_opens_new_tab_and_waits_for_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TabDouble(_FakeTab):
        def __init__(self) -> None:
            super().__init__(content="<html>ok</html>", href="")
            self.ready_states: list[str] = []

        @override
        async def wait_for_ready_state(
            self,
            until: str = "interactive",
            timeout: int = 10,
        ) -> bool:
            del timeout
            self.ready_states.append(until)
            return True

    class BrowserDouble:
        def __init__(self) -> None:
            self.tab = TabDouble()
            self.calls: list[tuple[str, bool]] = []

        async def get(self, url: str, new_tab: bool = False) -> TabDouble:
            self.calls.append((url, new_tab))
            return self.tab

    browser = BrowserDouble()

    async def guard(*args: object, **kwargs: object) -> None:
        del args, kwargs

    monkeypatch.setattr(zendriver, "_guard_requests", guard)
    tab = asyncio.run(
        zendriver._navigate_tab(
            cast(Browser, browser),
            "https://example.com/",
        ),
    )
    assert tab is browser.tab
    assert browser.calls == [("about:blank", True)]
    assert browser.tab.ready_states == ["complete"]


def test_closed_reports_timeout_and_failure_exactly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class HangingTab(_FakeTab):
        @override
        async def close(self) -> None:
            await asyncio.Event().wait()

    class FailingTab(_FakeTab):
        @override
        async def close(self) -> None:
            raise RuntimeError("close failed")

    caplog.set_level("DEBUG", logger=zendriver.__name__)
    asyncio.run(
        zendriver._closed(cast(Tab, HangingTab(content="", href="")), budget_sec=0.001),
    )
    asyncio.run(
        zendriver._closed(cast(Tab, FailingTab(content="", href="")), budget_sec=1.0),
    )
    assert [record.getMessage() for record in caplog.records] == [
        "tab close timed out; abandoning the tab",
        "tab close failed; abandoning the tab",
    ]
    default = cast(
        object,
        inspect.signature(zendriver._closed).parameters["budget_sec"].default,
    )
    assert default == 5.0


def test_close_orphan_browser_uses_bounded_probe_and_reports_stuck_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Connection:
        def close(self) -> None:
            pass

    connections = [Connection(), Connection()]
    calls: list[tuple[tuple[str, int], float | None]] = []
    clock = iter([0.0, 0.0, 10.0])
    sleeps: list[float] = []

    def create_connection(
        address: tuple[str, int],
        timeout: float | None,
    ) -> Connection:
        calls.append((address, timeout))
        return connections.pop()

    def port(profile: Path) -> int:
        del profile
        return 9222

    async def close_browser(port: int) -> None:
        del port

    def run(coro: Coroutine[object, object, object]) -> None:
        coro.close()

    monkeypatch.setattr(zendriver, "_devtools_port", port)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(time, "sleep", sleeps.append)
    monkeypatch.setattr(zendriver, "_close_browser_on_port", close_browser)
    monkeypatch.setattr(asyncio, "run", run)
    with pytest.raises(
        zendriver.BrowserUnavailableError,
        match=r"^Chrome on DevTools port 9222 did not close\.$",
    ):
        zendriver._close_orphan_browser(_PROFILE)
    assert calls == [
        (("127.0.0.1", 9222), 0.2),
        (("127.0.0.1", 9222), 0.1),
    ]
    assert sleeps == [0.05]


def test_open_instance_polls_until_browser_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = _FakeBrowser()
    sleeps: list[float] = []

    async def launch(profile_dir: Path, *, headless: bool) -> _FakeBrowser:
        assert profile_dir == _PROFILE
        assert headless is False
        return browser

    async def navigate(browser_arg: Browser, url: str) -> Tab:
        assert browser_arg is browser
        assert url == "https://example.com/"
        return cast(Tab, browser.last_tab)

    async def sleep(delay: float) -> None:
        sleeps.append(delay)
        browser.stopped = True

    monkeypatch.setattr(zendriver, "_launch_browser", launch)
    monkeypatch.setattr(zendriver, "_navigate_tab", navigate)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    asyncio.run(zendriver._open_instance("https://example.com/", _PROFILE))
    assert sleeps == [0.5]


def test_stopped_logs_and_kills_on_timeout_and_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class HangingBrowser(_FakeBrowser):
        @override
        async def stop(self) -> None:
            await asyncio.Event().wait()

    class FailingBrowser(_FakeBrowser):
        @override
        async def stop(self) -> None:
            raise RuntimeError("stop failed")

    timed_out = HangingBrowser()
    failed = FailingBrowser()
    timed_out_process = _FakeProcess()
    failed_process = _FakeProcess()
    timed_out._process = timed_out_process
    failed._process = failed_process
    caplog.set_level("WARNING", logger=zendriver.__name__)
    asyncio.run(
        zendriver._stopped(cast(Browser, timed_out), budget_sec=0.001),
    )
    asyncio.run(zendriver._stopped(cast(Browser, failed), budget_sec=1.0))
    assert [record.getMessage() for record in caplog.records] == [
        "browser stop timed out; killing the browser process",
        "browser stop failed; killing the browser process",
    ]
    assert caplog.records[0].exc_info is None
    assert caplog.records[1].exc_info is not None
    assert timed_out_process.kills == 1
    assert failed_process.kills == 1


def test_settled_content_waits_for_complete_after_navigation() -> None:
    class RecordingTab(_FakeTab):
        def __init__(self) -> None:
            super().__init__(
                content="<html><title>Just a moment...</title></html>",
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
    assert asyncio.run(zendriver._settled_content(cast(Tab, tab), budget_sec=1.0)) == (
        "<html>clear</html>"
    )
    assert tab.ready_states == ["complete"]


def test_pool_returns_the_same_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    class Pool:
        pass

    created: list[Pool] = []

    def make_pool() -> Pool:
        pool = Pool()
        created.append(pool)
        return pool

    monkeypatch.setattr(zendriver, "_pool_singleton", None)
    monkeypatch.setattr(zendriver, "_BrowserPool", make_pool)
    first = zendriver._pool()
    second = zendriver._pool()
    assert first is second
    assert created == [first]


def test_main_help_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["fetch-zendriver", "--help"])
    with pytest.raises(SystemExit) as error:
        zendriver.main()
    assert error.value.code == 0
    assert capsys.readouterr().out == (
        "usage: fetch-zendriver [-h] [url]\n\n"
        "Open a URL in a headed Chrome on the zendriver backend's dedicated profile -- "
        "the same profile the headless RequestParams(policy=PolicyParams(transport="
        '"zendriver")) fetch uses. Use it to debug a fetch that errored: you see '
        "exactly what Chrome renders (a challenge, a login wall, a broken page), and "
        "any cookies you seat while there (e.g. by logging in) persist for later "
        "headless fetches. Close the window when done.\n\n"
        "positional arguments:\n"
        "  url         The URL to open (typically the one whose headless fetch failed).\n"
        "              Omit to open a blank page and navigate by hand.\n\n"
        "options:\n"
        "  -h, --help  show this help message and exit\n\n"
        "Examples:\n"
        "  fetch-zendriver https://the-site-that-failed.example/\n"
        "  fetch-zendriver https://login.example/  # seat a session cookie\n"
        "  fetch-zendriver # opens blank; navigate by hand\n"
    )


def test_fetch_zendriver_defaults_are_stable() -> None:
    parameters = inspect.signature(zendriver.fetch_zendriver).parameters
    assert cast(object, parameters["timeout_sec"].default) == 30.0
    assert cast(object, parameters["headless"].default) is True
    assert cast(object, parameters["trust"].default) == "untrusted"
    assert cast(object, parameters["max_redirects"].default) == 10


def test_closed_logs_traceback_for_failed_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingTab(_FakeTab):
        @override
        async def close(self) -> None:
            raise RuntimeError("close failed")

    caplog.set_level("DEBUG", logger=zendriver.__name__)
    asyncio.run(
        zendriver._closed(cast(Tab, FailingTab(content="", href=""))),
    )
    assert len(caplog.records) == 1
    assert caplog.records[0].getMessage() == "tab close failed; abandoning the tab"
    assert caplog.records[0].exc_info is not None


def test_domain_cookies_strips_one_cookie_domain_prefix() -> None:
    browser = _FakeBrowser(
        cookies=[_FakeCookie(name="COOKIE", value="ok", domain=".example.com")],
    )
    assert asyncio.run(
        zendriver._domain_cookies(cast(Browser, browser), "https://example.com/"),
    ) == {"COOKIE": "ok"}


def test_guard_logs_exact_redirect_budget(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    browser = _FakeBrowser(
        paused_events=[_request_paused("https://public.example/next")],
    )
    _patch_pool(monkeypatch, browser)
    caplog.set_level("DEBUG", logger=zendriver.__name__)
    asyncio.run(
        _navigate(
            "https://public.example/start",
            profile_dir=_PROFILE,
            egress="e",
            timeout_sec=5.0,
            headless=True,
            max_redirects=0,
            on_redirect=None,
        ),
    )
    assert [record.getMessage() for record in caplog.records] == [
        "redirect budget of 0 exhausted at 'https://public.example/next'",
    ]


def test_request_pool_release_uses_unix_socket_and_rejects_non_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Socket:
        def __init__(self, family: int, kind: int) -> None:
            self.arguments = (family, kind)
            self.timeout: float | None = None
            self.connected: str | None = None
            self.sent: bytes | None = None
            self.closed = False

        def settimeout(self, timeout: float) -> None:
            self.timeout = timeout

        def connect(self, address: str) -> None:
            self.connected = address

        def sendall(self, data: bytes) -> None:
            self.sent = data

        def recv(self, size: int) -> bytes:
            assert size == 64
            return b"unexpected"

        def close(self) -> None:
            self.closed = True

    sockets: list[Socket] = []

    def make_socket(family: int, kind: int) -> Socket:
        socket = Socket(family, kind)
        sockets.append(socket)
        return socket

    monkeypatch.setattr(socket, "socket", make_socket)
    with pytest.raises(
        RuntimeError,
        match=r"^Zendriver browser pool returned an invalid response\.$",
    ):
        zendriver._request_pool_release(_PROFILE)
    assert len(sockets) == 1
    assert sockets[0].arguments == (socket.AF_UNIX, socket.SOCK_STREAM)
    assert sockets[0].timeout == 10
    assert sockets[0].connected == zendriver._control_address(_PROFILE)
    assert sockets[0].sent == b"release\n"
    assert sockets[0].closed


def test_domain_cookies_rejects_nonmatching_prefix_domains() -> None:
    browser = _FakeBrowser(
        cookies=[
            _FakeCookie(name="PREFIX", value="bad", domain="XX.example.com"),
            _FakeCookie(name="MATCH", value="ok", domain="example.com"),
        ],
    )
    assert asyncio.run(
        zendriver._domain_cookies(cast(Browser, browser), "https://example.com/"),
    ) == {"MATCH": "ok"}


def test_closed_preserves_failure_traceback_and_default_budget(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingTab(_FakeTab):
        @override
        async def close(self) -> None:
            raise RuntimeError("close failed")

    caplog.set_level("DEBUG", logger=zendriver.__name__)
    asyncio.run(
        zendriver._closed(cast(Tab, FailingTab(content="", href="")), budget_sec=1.0),
    )
    record = caplog.records[-1]
    assert record.getMessage() == "tab close failed; abandoning the tab"
    assert record.exc_info is not None


def test_settled_content_marks_success_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def classify(body: str, *, on_success_body: bool = False) -> None:
        del body
        calls.append(on_success_body)

    monkeypatch.setattr(zendriver, "classify_challenge", classify)
    tab = _FakeTab(content="<html>ok</html>", href="https://example.com/")
    assert asyncio.run(zendriver._settled_content(cast(Tab, tab), budget_sec=0.0)) == (
        "<html>ok</html>"
    )
    assert calls == [True]


def test_navigate_signature_defaults_and_settle_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = _FakeBrowser(href="https://example.com/final")
    pool = _StubPool(browser)
    budgets: list[float] = []

    async def settle(tab: Tab, *, budget_sec: float) -> str:
        del tab
        budgets.append(budget_sec)
        return "<html>ok</html>"

    monkeypatch.setattr(zendriver, "_pool", lambda: pool)
    monkeypatch.setattr(zendriver, "_settled_content", settle)
    result = asyncio.run(
        zendriver._navigate(
            "https://example.com/",
            profile_dir=_PROFILE,
            egress="egress",
            timeout_sec=8.0,
            headless=True,
        ),
    )
    assert result == BrowserResult(
        body=b"<html>ok</html>",
        cookies={},
        final_url="https://example.com/final",
    )
    assert budgets == [4.0]


def test_navigate_tab_uses_exact_defaults_and_enables_network_for_ambient_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = _FakeBrowser()

    async def guard(*args: object, **kwargs: object) -> None:
        del args, kwargs

    monkeypatch.setattr(zendriver, "_guard_requests", guard)
    asyncio.run(
        zendriver._navigate_tab(
            cast(Browser, browser),
            "https://example.com/",
            headers={"X-Test": "yes"},
        ),
    )
    assert browser.last_tab is not None
    assert [command["method"] for command in browser.last_tab.wire_commands] == [
        "Network.enable",
        "Network.setExtraHTTPHeaders",
    ]


def test_guard_uses_exact_document_pattern() -> None:
    tab = _FakeTab(content="", href="")
    asyncio.run(
        zendriver._guard_requests(
            cast(Tab, tab),
            "https://example.com/start#fragment",
            trust="internal",
            on_redirect=None,
            max_redirects=0,
        ),
    )
    assert tab.wire_commands == [
        {
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
        },
    ]


def test_main_uses_fixed_program_name(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["different-name", "--help"])
    with pytest.raises(SystemExit):
        zendriver.main()
    assert capsys.readouterr().out.startswith("usage: fetch-zendriver [-h] [url]\n")


def test_process_command_requires_false_check_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[bool] = []

    def run(
        args: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del args, capture_output, text
        seen.append(check)
        return subprocess.CompletedProcess([], 0, stdout="command")

    monkeypatch.setattr(subprocess, "run", run)
    assert (
        zendriver._process_command(
            1,
            proc_root=tmp_path / "missing",
            platform="darwin",
        )
        == "command"
    )
    assert seen == [False]


def test_process_command_does_not_switch_linux_to_ps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        raise AssertionError("Linux proc lookup must not invoke ps")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(FileNotFoundError):
        zendriver._process_command(
            1,
            proc_root=tmp_path / "missing",
            platform="linux",
        )


def test_domain_cookies_does_not_match_empty_domain_to_placeholder_host() -> None:
    browser = _FakeBrowser(
        cookies=[_FakeCookie(name="EMPTY", value="bad", domain="")],
    )
    assert (
        asyncio.run(
            zendriver._domain_cookies(cast(Browser, browser), "https://XXXX/"),
        )
        == {}
    )


def test_request_pool_release_removes_stale_files_with_missing_ok(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stale = tmp_path / "control.sock"
    unlinked: list[tuple[Path, bool | None]] = []

    class Socket:
        def settimeout(self, timeout: float) -> None:
            del timeout

        def connect(self, address: str) -> None:
            del address
            raise ConnectionRefusedError

        def close(self) -> None:
            pass

    def make_socket(family: int, kind: int) -> Socket:
        assert (family, kind) == (socket.AF_UNIX, socket.SOCK_STREAM)
        return Socket()

    def unlink(self: Path, *, missing_ok: bool = False) -> None:
        unlinked.append((self, missing_ok))

    def control_address(profile: Path) -> str:
        assert profile == _PROFILE
        return str(stale)

    def close_orphan_browser(profile: Path) -> None:
        assert profile == _PROFILE

    monkeypatch.setattr(zendriver, "_control_address", control_address)
    monkeypatch.setattr(socket, "socket", make_socket)
    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(zendriver, "_close_orphan_browser", close_orphan_browser)
    zendriver._request_pool_release(_PROFILE)
    assert unlinked == [(stale, True)]


def test_close_orphan_browser_passes_the_profile_to_port_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Path] = []

    def no_port(profile: Path) -> int | None:
        seen.append(profile)
        return None

    monkeypatch.setattr(zendriver, "_devtools_port", no_port)
    zendriver._close_orphan_browser(_PROFILE)
    assert seen == [_PROFILE]


def test_cleanup_defaults_are_passed_to_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budgets: list[float] = []
    real_timeout = asyncio.timeout

    def timeout(budget: float):
        budgets.append(budget)
        return real_timeout(0.0)

    monkeypatch.setattr(asyncio, "timeout", timeout)
    asyncio.run(zendriver._closed(cast(Tab, _FakeTab(content="", href=""))))
    asyncio.run(zendriver._stopped(cast(Browser, _FakeBrowser())))
    assert budgets == [5.0, 30.0]


def test_browser_pool_thread_is_named_and_daemonized() -> None:
    pool = _BrowserPool(serve_control=False)
    try:
        assert pool._thread.name == "loop-web-browser"
        assert pool._thread.daemon is True
    finally:
        pool.shutdown()


def test_browser_pool_without_control_does_not_create_a_server() -> None:
    pool = _BrowserPool(serve_control=False)
    try:
        pool._ensure_control(_PROFILE)
        assert pool._controls == {}
    finally:
        pool.shutdown()


def test_browser_pool_control_server_receives_profile_and_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[Path, Callable[[], None]]] = []

    class Control:
        def __init__(self, profile: Path, release: Callable[[], None]) -> None:
            created.append((profile, release))

        def close(self) -> None:
            pass

    pool = _BrowserPool(serve_control=True)
    monkeypatch.setattr(zendriver, "_PoolControlServer", Control)
    try:
        pool._ensure_control(_PROFILE)
    finally:
        pool.shutdown()
    assert len(created) == 1
    assert created[0][0] == _PROFILE
    assert created[0][1] is zendriver.shutdown_browsers


@pytest.fixture
def socket_dir() -> Iterator[Path]:
    """Yield a directory shallow enough to bind a Unix socket in.

    AF_UNIX caps a socket path at 108 bytes, and ``tmp_path`` nests several levels
    under ``TMPDIR``: past the cap wherever TMPDIR is itself long. Production binds
    directly under ``tempfile.gettempdir()`` for the same reason.
    """
    with tempfile.TemporaryDirectory(prefix="zd-") as root:
        yield Path(root)


def test_pool_control_server_tracks_filesystem_socket_path(
    monkeypatch: pytest.MonkeyPatch,
    socket_dir: Path,
) -> None:
    address = socket_dir / "control.sock"

    serve_calls: list[float] = []

    def control_address(profile: Path) -> str:
        assert profile == _PROFILE
        return str(address)

    def serve_forever(
        server: socketserver.ThreadingUnixStreamServer,
        *,
        poll_interval: float = 0.5,
    ) -> None:
        del server
        serve_calls.append(poll_interval)

    monkeypatch.setattr(zendriver, "_control_address", control_address)
    monkeypatch.setattr(
        socketserver.ThreadingUnixStreamServer,
        "serve_forever",
        serve_forever,
    )
    server = zendriver._PoolControlServer(_PROFILE, lambda: None)
    monkeypatch.setattr(server, "shutdown", lambda: None)
    try:
        assert server._control_path == address
        assert address.exists()
        assert server._thread.name == "loop-web-browser-control"
        assert server._thread.daemon is True
        assert serve_calls == [0.01]
    finally:
        server.close()
    assert not address.exists()


def test_pool_control_close_unlinks_missing_socket_without_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "control.sock"
    unlinked: list[tuple[Path, bool | None]] = []

    class Thread:
        def join(self) -> None:
            pass

    def unlink(self: Path, *, missing_ok: bool = False) -> None:
        unlinked.append((self, missing_ok))

    def shutdown(server: zendriver._PoolControlServer) -> None:
        del server

    def server_close(server: zendriver._PoolControlServer) -> None:
        del server

    monkeypatch.setattr(zendriver._PoolControlServer, "shutdown", shutdown)
    monkeypatch.setattr(
        zendriver._PoolControlServer,
        "server_close",
        server_close,
    )
    monkeypatch.setattr(Path, "unlink", unlink)
    server = object.__new__(zendriver._PoolControlServer)
    server._control_path = path
    server._thread = cast(threading.Thread, Thread())
    server.close()

    assert unlinked == [(path, True)]


def test_browser_pool_run_defaults_to_no_timeout() -> None:
    default = cast(
        object,
        inspect.signature(_BrowserPool.run).parameters["timeout_sec"].default,
    )
    assert default == 0


def test_browser_pool_run_loop_installs_exact_warning_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.new_event_loop()
    pool = object.__new__(_BrowserPool)
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


def test_browser_pool_shutdown_kills_at_zero_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _BrowserPool(serve_control=False)
    browser = _FakeBrowser()
    pool._browsers[("egress", "/profile")] = (True, cast(Browser, browser))
    killed: list[Browser] = []

    def monotonic() -> float:
        return 100.0

    def kill(browser_to_kill: Browser) -> None:
        killed.append(browser_to_kill)

    monkeypatch.setattr(time, "monotonic", monotonic)
    monkeypatch.setattr(zendriver, "_kill_browser_process", kill)
    pool.shutdown(budget_sec=0.0)

    assert killed == [cast(Browser, browser)]
    assert browser.stop_calls == 0


def test_browser_pool_shutdown_stops_a_live_browser(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pool = _BrowserPool(serve_control=False)
    browser = _FakeBrowser()
    pool._browsers[("egress", "/profile")] = (True, cast(Browser, browser))
    try:
        pool.shutdown(budget_sec=1.0)
    finally:
        if not pool._loop.is_closed():
            pool.shutdown()
    assert browser.stop_calls == 1
    assert caplog.records == []


def test_browser_pool_shutdown_logs_and_kills_failed_stop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    browser = _RaisingStopBrowser()
    process = _FakeProcess()
    browser._process = process
    pool = _BrowserPool(serve_control=False)
    pool._browsers[("egress", "/profile")] = (True, cast(Browser, browser))
    caplog.set_level("WARNING", logger=zendriver.__name__)
    pool.shutdown()
    assert [record.getMessage() for record in caplog.records] == [
        "browser stop failed during shutdown",
    ]
    record = caplog.records[0]
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError
    assert record.exc_text is not None
    assert "stop blew up" in record.exc_text
    assert process.kills == 1


def test_browser_pool_shutdown_default_budget_is_five_seconds() -> None:
    default = cast(
        object,
        inspect.signature(_BrowserPool.shutdown).parameters["budget_sec"].default,
    )
    assert default == 5.0


def test_navigate_tab_default_policy_values_reach_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = _FakeBrowser()
    captured: dict[str, object] = {}

    async def guard(*args: object, **kwargs: object) -> None:
        del args
        captured.update(kwargs)

    monkeypatch.setattr(zendriver, "_guard_requests", guard)
    asyncio.run(
        zendriver._navigate_tab(
            cast(Browser, browser),
            "https://example.com/",
        ),
    )
    assert captured["trust"] == "untrusted"
    assert captured["max_redirects"] == 10


def test_guard_strips_fragment_from_initial_wire_url() -> None:
    seen: list[str] = []
    browser = _FakeBrowser(
        paused_events=[_request_paused("https://example.com/")],
    )
    asyncio.run(
        zendriver._navigate_tab(
            cast(Browser, browser),
            "https://example.com/#fragment",
            trust="internal",
            on_redirect=seen.append,
        ),
    )
    assert seen == []


def test_settled_content_returns_at_exact_deadline_without_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoWait:
        def clear(self) -> None:
            pass

        async def wait(self) -> None:
            raise AssertionError("wait crossed an exact deadline")

    def navigation_watcher(tab: Tab) -> NoWait:
        del tab
        return NoWait()

    monkeypatch.setattr(zendriver, "_main_frame_navigations", navigation_watcher)
    tab = _FakeTab(
        content="<html><title>Just a moment...</title></html>",
        href="https://example.com/",
    )

    async def go() -> str:
        loop = asyncio.get_running_loop()
        calls = 0

        def time() -> float:
            nonlocal calls
            calls += 1
            return 100.0

        monkeypatch.setattr(loop, "time", time)
        return await zendriver._settled_content(cast(Tab, tab), budget_sec=0.0)

    assert asyncio.run(go()) == "<html><title>Just a moment...</title></html>"


def test_domain_cookies_never_invents_a_domain_for_empty_cookie_domain() -> None:
    browser = _FakeBrowser(
        cookies=[_FakeCookie(name="EMPTY", value="bad", domain="")],
    )
    assert (
        asyncio.run(
            zendriver._domain_cookies(cast(Browser, browser), "https://XXXX/"),
        )
        == {}
    )


def test_settled_content_does_not_wait_at_exact_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_wait(*args: object, **kwargs: object) -> str:
        del args, kwargs
        raise AssertionError("wait_for crossed an exact deadline")

    monkeypatch.setattr(asyncio, "wait_for", fail_wait)
    tab = _FakeTab(
        content="<html><title>Just a moment...</title></html>",
        href="https://example.com/",
    )

    async def go() -> str:
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "time", lambda: 100.0)
        return await zendriver._settled_content(cast(Tab, tab), budget_sec=0.0)

    assert asyncio.run(go()) == "<html><title>Just a moment...</title></html>"


def test_navigate_forwards_every_argument_and_harvests_final_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = _FakeBrowser(href="https://example.com/final")

    class RecordingTab(_FakeTab):
        def __init__(self) -> None:
            super().__init__(content="", href="https://example.com/final")
            self.evaluations: list[str] = []

        @override
        async def evaluate(self, expr: str) -> str:
            self.evaluations.append(expr)
            return await super().evaluate(expr)

    tab = RecordingTab()
    captured: dict[str, object] = {}
    default_calls: list[dict[str, object]] = []

    def callback(target: str) -> None:
        del target

    class Pool:
        async def browser(
            self,
            egress: str,
            profile_dir: Path,
            *,
            headless: bool,
        ) -> _FakeBrowser:
            captured["browser"] = (egress, profile_dir, headless)
            return browser

    async def navigate_tab(
        browser_arg: Browser,
        url: str,
        **kwargs: object,
    ) -> Tab:
        captured["tab"] = (browser_arg, url, kwargs)
        if kwargs.get("trust") == "untrusted":
            default_calls.append(kwargs)
        return cast(Tab, tab)

    async def settle(tab_arg: Tab, *, budget_sec: float) -> str:
        captured["settle"] = (tab_arg, budget_sec)
        return "<html>body</html>"

    async def cookies(browser_arg: Browser, url: str) -> dict[str, str]:
        captured["cookies"] = (browser_arg, url)
        return {"SID": "value"}

    async def close(tab_arg: Tab) -> None:
        captured["close"] = tab_arg

    def pool() -> Pool:
        return Pool()

    monkeypatch.setattr(zendriver, "_pool", pool)
    monkeypatch.setattr(zendriver, "_navigate_tab", navigate_tab)
    monkeypatch.setattr(zendriver, "_settled_content", settle)
    monkeypatch.setattr(zendriver, "_domain_cookies", cookies)
    monkeypatch.setattr(zendriver, "_closed", close)
    result = asyncio.run(
        zendriver._navigate(
            "https://example.com/start",
            profile_dir=_PROFILE,
            egress="egress",
            timeout_sec=8.0,
            headless=False,
            headers={"X-Test": "yes"},
            cookies={"OLD": "cookie"},
            trust="internal",
            max_redirects=2,
            on_redirect=callback,
        ),
    )
    assert result == BrowserResult(
        body=b"<html>body</html>",
        cookies={"SID": "value"},
        final_url="https://example.com/final",
    )
    assert len(browser.cookies.seeded) == 1
    seeded = browser.cookies.seeded[0]
    assert isinstance(seeded, network.CookieParam)
    assert seeded.name == "OLD"
    assert seeded.value == "cookie"
    assert seeded.url == "https://example.com/start"
    assert tab.evaluations == ["document.location.href"]
    assert captured == {
        "browser": ("egress", _PROFILE, False),
        "tab": (
            browser,
            "https://example.com/start",
            {
                "headers": {"X-Test": "yes"},
                "trust": "internal",
                "max_redirects": 2,
                "on_redirect": callback,
            },
        ),
        "settle": (tab, 4.0),
        "cookies": (browser, "https://example.com/final"),
        "close": tab,
    }
    asyncio.run(
        zendriver._navigate(
            "https://example.com/start",
            profile_dir=_PROFILE,
            egress="egress",
            timeout_sec=8.0,
            headless=False,
        ),
    )
    assert default_calls == [
        {
            "headers": None,
            "trust": "untrusted",
            "max_redirects": 10,
            "on_redirect": None,
        },
    ]


def test_open_instance_forwards_exact_target_and_clears_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_target = Path("data/rekursiv-ai/wesearch/fetch-zendriver")
    explicit_target = Path("explicit-profile")
    releases: list[Path] = []
    runs: list[tuple[str, Path]] = []
    cleared: list[str] = []

    class Pool:
        def run(self, coroutine: Coroutine[object, object, None]) -> None:
            async def capture() -> None:
                await coroutine

            asyncio.run(capture())

    async def open_instance(url: str, profile_dir: Path) -> None:
        runs.append((url, profile_dir))

    def release(profile_dir: Path) -> None:
        releases.append(profile_dir)

    def pool() -> Pool:
        return Pool()

    monkeypatch.setattr(zendriver, "data_dir", lambda: Path("data"))
    monkeypatch.setattr(zendriver, "_request_pool_release", release)
    monkeypatch.setattr(zendriver, "_open_instance", open_instance)
    monkeypatch.setattr(zendriver, "_pool", pool)
    monkeypatch.setattr(zendriver, "clear_domain_cooldowns", cleared.append)

    zendriver.open_instance("https://example.com/page")
    zendriver.open_instance("https://example.com/other", profile_dir=explicit_target)

    assert releases == [default_target, explicit_target]
    assert runs == [
        ("https://example.com/page", default_target),
        ("https://example.com/other", explicit_target),
    ]
    assert cleared == ["example.com", "example.com"]


def test_open_instance_does_not_clear_cooldowns_without_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleared: list[str] = []

    class Pool:
        def run(self, coroutine: Coroutine[object, object, None]) -> None:
            coroutine.close()

    def data_path() -> Path:
        return Path("data")

    def release(profile_dir: Path) -> None:
        del profile_dir

    async def open_instance(url: str, profile_dir: Path) -> None:
        del url, profile_dir

    def pool() -> Pool:
        return Pool()

    monkeypatch.setattr(zendriver, "data_dir", data_path)
    monkeypatch.setattr(zendriver, "_request_pool_release", release)
    monkeypatch.setattr(zendriver, "_open_instance", open_instance)
    monkeypatch.setattr(zendriver, "_pool", pool)
    monkeypatch.setattr(zendriver, "clear_domain_cooldowns", cleared.append)

    zendriver.open_instance("about:blank")

    assert cleared == []


def test_fetch_zendriver_forwards_all_literal_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def navigate(
        url: str,
        *,
        profile_dir: Path,
        egress: str,
        timeout_sec: float,
        headless: bool,
        headers: dict[str, str] | None,
        cookies: dict[str, str] | None,
        trust: Trust,
        max_redirects: int,
        on_redirect: Callable[[str], None] | None,
    ) -> BrowserResult:
        captured.update(
            {
                "url": url,
                "profile_dir": profile_dir,
                "egress": egress,
                "timeout_sec": timeout_sec,
                "headless": headless,
                "headers": headers,
                "cookies": cookies,
                "trust": trust,
                "max_redirects": max_redirects,
                "on_redirect": on_redirect,
            },
        )
        return BrowserResult(body=b"body", cookies={}, final_url=url)

    class Pool:
        def run(
            self,
            coroutine: Coroutine[object, object, BrowserResult],
            *,
            timeout_sec: float,
        ) -> BrowserResult:
            captured["run_timeout_sec"] = timeout_sec
            return asyncio.run(coroutine)

    def pool() -> Pool:
        return Pool()

    monkeypatch.setattr(zendriver, "_navigate", navigate)
    monkeypatch.setattr(zendriver, "_pool", pool)
    result = zendriver.fetch_zendriver(
        "https://example.com/",
        profile_dir=_PROFILE,
        egress="egress",
    )

    assert result == BrowserResult(
        body=b"body",
        cookies={},
        final_url="https://example.com/",
    )
    assert captured == {
        "url": "https://example.com/",
        "profile_dir": _PROFILE,
        "egress": "egress",
        "timeout_sec": 30.0,
        "headless": True,
        "headers": None,
        "cookies": None,
        "trust": "untrusted",
        "max_redirects": 10,
        "on_redirect": None,
        "run_timeout_sec": 60.0,
    }


def test_pool_run_default_timeout_waits_without_a_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeouts: list[float | None] = []

    class Future:
        def result(self, timeout: float | None) -> int:
            timeouts.append(timeout)
            return 1

        def cancel(self) -> bool:
            return False

    def submit(
        coroutine: Coroutine[object, object, int],
        loop: asyncio.AbstractEventLoop,
    ) -> Future:
        del loop
        coroutine.close()
        return Future()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)
    pool = _BrowserPool(serve_control=False)
    try:

        async def value() -> int:
            return 1

        assert pool.run(value()) == 1
        assert timeouts == [None]
    finally:
        pool.shutdown()


def test_pool_shutdown_uses_its_literal_default_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _BrowserPool(serve_control=False)
    browser = _FakeBrowser()
    pool._browsers[("egress", "profile")] = (True, cast(Browser, browser))
    timeouts: list[float] = []

    def monotonic() -> float:
        return 100.0

    def run(
        coroutine: Coroutine[object, object, object],
        *,
        timeout_sec: float,
    ) -> object:
        timeouts.append(timeout_sec)
        coroutine.close()
        return object()

    monkeypatch.setattr(time, "monotonic", monotonic)
    monkeypatch.setattr(pool, "run", run)
    pool.shutdown()

    assert timeouts == [5.0]


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
