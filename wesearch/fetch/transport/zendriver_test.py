"""Tests for ``wesearch.fetch.transport.zendriver`` (zendriver headless fetch backend).

Hermetic: a fake async browser stands in for zendriver, so the transport logic
(cookie-domain filtering, challenge detection, redirect callback, pool reuse)
is exercised with no Chrome and no network.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import GeneratorType
from typing import Any, cast, override

import asyncio
import atexit
import importlib
import inspect
import subprocess
import tempfile
import threading
import time

from zendriver.core.connection import Transaction

import pytest
import zendriver

from wesearch.fetch.transport.zendriver import (
    BrowserResult,
    _BrowserPool,
    _navigate,
)
from wesearch.lib.custom_json import DictCodec, ListCodec, StrCodec
from wesearch.types.params import Trust

import wesearch.fetch.transport.zendriver as fz_mod


# A fake profile dir; the browser is mocked in every test, so it is never
# touched on disk.
_PROFILE = Path("test-profile")

# Captured at IMPORT, which is the only moment it is still reachable: arming is
# class-wide and permanent, and earlier tests in this module launch browsers, so
# a test that read ``Transaction.__call__`` at call time would restore the guard
# it means to remove and assert nothing.
_VENDOR_TRANSACTION_CALL = Transaction.__call__


@pytest.mark.cli_python_subprocess
def test_direct_executable_reexecutes_as_module() -> None:
    script = Path(__file__).with_name("zendriver.py")
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
        self.seeded: list[Any] = []

    async def get_all(self) -> list[_FakeCookie]:
        return self._cookies

    async def set_all(self, cookies: list[Any]) -> None:
        self.seeded = cookies


def _main_frame_navigated() -> zendriver.cdp.page.FrameNavigated:
    """A real ``FrameNavigated`` for the MAIN frame (``parent_id is None``).

    The genuine CDP dataclass, not a look-alike: the transport guards on
    ``isinstance`` (a live run delivered a foreign event type to the handler),
    so a stand-in would satisfy the fake and be rejected in production -- the
    exact direction a test must never fail in. Only the two fields the transport
    reads carry meaning; the rest are the shape the class requires.
    """
    frame = zendriver.cdp.page.Frame(
        id_=zendriver.cdp.page.FrameId("main"),
        loader_id=zendriver.cdp.network.LoaderId("loader"),
        url="https://walled.example/",
        domain_and_registry="walled.example",
        security_origin="https://walled.example",
        mime_type="text/html",
        secure_context_type=zendriver.cdp.page.SecureContextType.SECURE,
        cross_origin_isolated_context_type=(
            zendriver.cdp.page.CrossOriginIsolatedContextType.NOT_ISOLATED
        ),
        gated_api_features=[],
        parent_id=None,
    )
    return zendriver.cdp.page.FrameNavigated(
        frame=frame, type_=zendriver.cdp.page.NavigationType.NAVIGATION
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
        paused_events: list[Any] | None = None,
    ) -> None:
        self._documents = documents if documents is not None else [content]
        self._parsing = parsing
        self._href = href
        self.closed = False
        self.navigations: list[str] = []
        self.commands: list[Any] = []
        self.handlers: list[Any] = []
        self.paused_handlers: list[Any] = []
        self.continued_requests: list[str] = []
        self.failed_requests: list[str] = []
        self._paused_events: list[Any] = list(paused_events or [])
        self.wire_commands: list[dict[str, object]] = []
        self.content_reads = 0
        self._index = 0
        # True between a navigation and its ready-state wait -- the window in
        # which the new document exists but has not parsed.
        self._is_parsing = False

    def add_handler(self, event_type: Any, handler: Any) -> None:
        # Routed by event type, mirroring zendriver's own dispatch: the
        # navigation watcher and the per-request guard must not receive each
        # other's events, which a single handler list cannot express.
        if event_type is zendriver.cdp.fetch.RequestPaused:
            self.paused_handlers.append(handler)
            return
        self.handlers.append(handler)

    def pause_request(self, event: Any) -> None:
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

    async def send(self, command: Any) -> None:
        self.commands.append(command)
        # A CDP verb is a generator yielding its wire dict; reading that dict is
        # how the fake records the REAL decision (method + request id) rather
        # than trusting the transport's own account of what it sent.
        raw = next(command, None) if isinstance(command, GeneratorType) else None
        if not isinstance(raw, dict):
            return
        # ``DictCodec.coerce`` rather than a bare ``.get`` ladder: the CDP verbs are
        # unstubbed, so their wire dict arrives as ``dict[Unknown, Unknown]``
        # and every read off it is partially unknown.
        payload = DictCodec.coerce(cast("object", raw))
        # Kept: a generator is single-use, so a test that re-reads ``commands``
        # would find every one exhausted by this very inspection.
        self.wire_commands.append(payload)
        request_id = StrCodec.coerce(
            DictCodec.coerce(payload.get("params")).get("requestId")
        )
        if not request_id:
            return
        method = StrCodec.coerce(payload.get("method"))
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
        paused_events: list[Any] | None = None,
    ) -> None:
        self._content = content
        self._href = href
        self._documents = documents
        self._parsing = parsing
        self._paused_events = paused_events or []
        self.cookies = _FakeCookieJar(cookies or [])
        self.stopped = False
        self.gets: list[str] = []
        self.stop_calls = 0
        self.last_tab: _FakeTab | None = None

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
    monkeypatch.setattr(fz_mod, "_pool", lambda: pool)
    return pool


def test_launch_browser_caps_dead_browser_connect_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # zendriver retries the DevTools connection ``browser_connection_max_tries``
    # times, each bounded by ``browser_connection_timeout``. When Chrome cannot
    # connect, the launch blocks for their product before raising. A healthy
    # Chrome exposes DevTools in ~0.3s, so the budget must clear that with margin
    # yet stay small: an unbounded product turns a transient browser gap into a
    # multi-second hang that stacks past the live-test timeout instead of
    # surfacing as a fast skip.
    captured: dict[str, float] = {}

    async def fake_start(config: zendriver.Config) -> _FakeBrowser:
        captured["timeout"] = config.browser_connection_timeout
        captured["max_tries"] = config.browser_connection_max_tries
        return _FakeBrowser()

    monkeypatch.setattr(zendriver, "start", fake_start)
    asyncio.run(fz_mod._launch_browser(tmp_path, headless=True))

    budget = captured["timeout"] * captured["max_tries"]
    assert captured["timeout"] >= 0.3, "connect timeout must clear healthy startup"
    assert budget <= 3.0, f"dead-browser connect budget {budget}s too large"


def test_launch_browser_uses_vanilla_zendriver_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    browser = _FakeBrowser()
    browser_args: list[str] = []

    async def fake_start(config: zendriver.Config) -> _FakeBrowser:
        browser_args.extend(config())
        return browser

    monkeypatch.setattr(zendriver, "start", fake_start)
    result = asyncio.run(fz_mod._launch_browser(tmp_path, headless=True))

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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, headless: bool
) -> None:
    """Chrome must come from the build that does not claim ``https``.

    On zendriver's default the launch runs the installed Chrome under
    ``com.google.Chrome``, which then receives every ``open https://...`` the
    user's clicks produce and drops them. Headed too: the capture follows the
    bundle id, not the window count.
    """
    captured: dict[str, object] = {}

    async def fake_start(config: zendriver.Config) -> _FakeBrowser:
        captured["executable"] = config.browser_executable_path
        captured["args"] = config()
        return _FakeBrowser()

    monkeypatch.setattr(zendriver, "start", fake_start)
    monkeypatch.setattr(fz_mod, "_fetch_browser", lambda: "/cache/ChromeForTesting")
    asyncio.run(fz_mod._launch_browser(tmp_path, headless=headless))

    assert captured["executable"] == "/cache/ChromeForTesting"
    # Without it Chrome for Testing raises a Safe Storage modal and blocks in
    # startup: the process is alive, so this surfaces as a browser that
    # launched and never exposed DevTools.
    assert "--use-mock-keychain" in cast(list[str], captured["args"])


def test_launch_browser_falls_back_when_no_reidentified_chrome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Linux, or a macOS host whose clone failed: launch exactly as before.

    ``Config`` resolves an installed Chrome for a falsy path, so ``""`` is
    already "find Chrome yourself" and needs no ``or None`` at the call site.
    Asserted against a default ``Config`` rather than a literal: the point is
    that the empty path is indistinguishable from passing nothing.
    """
    captured: dict[str, object] = {}

    async def fake_start(config: zendriver.Config) -> _FakeBrowser:
        captured["executable"] = config.browser_executable_path
        captured["args"] = config()
        return _FakeBrowser()

    monkeypatch.setattr(zendriver, "start", fake_start)
    monkeypatch.setattr(fz_mod, "_fetch_browser", lambda: "")
    asyncio.run(fz_mod._launch_browser(tmp_path, headless=True))

    assert captured["executable"] == zendriver.Config().browser_executable_path
    # Stock Chrome already holds its Keychain entry, so mocking it here would
    # cut the operator's own browser off from cookies it legitimately has.
    assert "--use-mock-keychain" not in cast(list[str], captured["args"])


def test_the_pool_leaves_zendriver_spawn_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
    vendor = util._start_process

    async def fake_start(config: zendriver.Config) -> _FakeBrowser:
        del config
        return _FakeBrowser()

    monkeypatch.setattr(zendriver, "start", fake_start)
    asyncio.run(fz_mod._launch_browser(tmp_path, headless=True))

    assert util._start_process is vendor, (
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
    monkeypatch.setattr(fz_mod, "_pool_singleton", None)
    monkeypatch.setattr(fz_mod, "_BrowserPool", stub_pool)

    fz_mod._pool()

    assert fz_mod.shutdown_browsers in registered


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

    async def launch(profile_dir: Path, *, headless: bool) -> zendriver.Browser:
        del profile_dir, headless
        launched.append(_FakeBrowser())
        return cast(zendriver.Browser, launched[-1])

    pool = fz_mod._BrowserPool(serve_control=False)
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
        assert not cast(_FakeBrowser, held).stopped
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

    monkeypatch.setattr(fz_mod, "_launch_browser", fake_launch)
    asyncio.run(fz_mod._open_instance(url, _PROFILE))

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
        await fz_mod._stopped(cast(Any, browser), budget_sec=0.2)
        return time.monotonic() - started

    elapsed = asyncio.run(asyncio.wait_for(go(), timeout=10.0))
    assert browser.stop_calls == 1, "the browser was never asked to stop"
    assert elapsed < 3.0, f"stop outlived its 0.2s budget: {elapsed:.2f}s"


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
    the cleanup routes through the bounded helper at all, which the 0.2s
    substitution shows in the same way 30s would.
    """
    browser = _HangingStopBrowser()

    async def fake_launch(profile_dir: Path, *, headless: bool) -> _FakeBrowser:
        del profile_dir, headless
        return browser

    # Bound BEFORE the patch: reading ``fz_mod._stopped`` inside the
    # replacement would resolve to the replacement itself and recurse.
    real_stopped = fz_mod._stopped

    async def briefly(browser: Any, *, budget_sec: float = 0.2) -> None:
        await real_stopped(browser, budget_sec=budget_sec)

    monkeypatch.setattr(fz_mod, "_launch_browser", fake_launch)
    monkeypatch.setattr(fz_mod, "_stopped", briefly)

    async def go() -> float:
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="navigation refused"):
            await fz_mod._open_instance("https://gated.example/page", _PROFILE)
        return time.monotonic() - started

    # Ten seconds stands in for the ceiling the real chain lacks: without a
    # bound the cleanup never returns and THIS fires, surfacing as TimeoutError
    # instead of the navigation's own error.
    elapsed = asyncio.run(asyncio.wait_for(go(), timeout=10.0))
    assert browser.stop_calls == 1, "the browser was never asked to stop"
    assert elapsed < 3.0, f"cleanup outlived its budget: {elapsed:.2f}s"


def test_open_instance_releases_profile_then_clears_domain_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakePool:
        def run(self, coroutine: Any) -> None:
            events.append("launch")
            coroutine.close()

    def release(_profile: Path) -> None:
        events.append("release")

    def clear(domain: str) -> int:
        events.append(f"clear:{domain}")
        return 1

    monkeypatch.setattr(fz_mod, "_request_pool_release", release)
    monkeypatch.setattr(fz_mod, "clear_domain_cooldowns", clear)
    monkeypatch.setattr(fz_mod, "_pool", FakePool)

    fz_mod.open_instance("https://gated.example/page", profile_dir=_PROFILE)

    assert events == ["release", "launch", "clear:gated.example"]


def test_pool_control_releases_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    releases: list[bool] = []
    checked: list[Path] = []
    monkeypatch.setattr(fz_mod, "_close_orphan_browser", checked.append)
    server = fz_mod._PoolControlServer(_PROFILE, lambda: releases.append(True))
    try:
        fz_mod._request_pool_release(_PROFILE)
    finally:
        server.close()
    assert releases == [True]
    assert checked == [_PROFILE]


def test_control_address_uses_platform_socket_namespace() -> None:
    linux_address = fz_mod._control_address(_PROFILE, platform="linux")
    darwin_address = fz_mod._control_address(_PROFILE, platform="darwin")

    assert linux_address.startswith("\0loop-zendriver-")
    assert Path(darwin_address).parent == Path(tempfile.gettempdir())
    assert Path(darwin_address).name.startswith("loop-zd-")
    assert darwin_address.endswith(".sock")


def test_pool_release_closes_orphan_when_control_is_unreachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    closed: list[Path] = []
    monkeypatch.setattr(fz_mod, "_close_orphan_browser", closed.append)

    fz_mod._request_pool_release(tmp_path)

    assert closed == [tmp_path]


def test_devtools_port_falls_back_to_singleton_owner(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    process = tmp_path / "proc" / "123"
    profile.mkdir()
    process.mkdir(parents=True)
    (profile / "SingletonLock").symlink_to("tron-123")
    (process / "cmdline").write_bytes(
        b"/opt/google/chrome/chrome\0"
        + f"--user-data-dir={profile}\0".encode()
        + b"--remote-debugging-port=4567\0"
        + b"about:blank\0"
    )

    assert fz_mod._devtools_port(profile, proc_root=tmp_path / "proc") == 4567


def test_devtools_port_rejects_different_profile_with_shared_prefix(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"
    process = tmp_path / "proc" / "123"
    profile.mkdir()
    process.mkdir(parents=True)
    (profile / "SingletonLock").symlink_to("tron-123")
    (process / "cmdline").write_text(
        "/opt/google/chrome/chrome "
        f"--user-data-dir={profile}-other "
        "--remote-debugging-port=4567"
    )

    assert fz_mod._devtools_port(profile, proc_root=tmp_path / "proc") is None


def test_devtools_port_rejects_stale_marker_without_profile_owner(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "DevToolsActivePort").write_text("4567\n/devtools/browser/id\n")

    assert fz_mod._devtools_port(profile, proc_root=tmp_path / "proc") is None


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
        )
    )
    assert isinstance(result, BrowserResult)
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
        )
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
        )
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
        )
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
        )
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
        )
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
        )
    )
    assert result.body == content.encode()


def _continued_headers(tab: _FakeTab) -> dict[str, str]:
    """Return the headers the guard released a continued request with."""
    payload = next(
        (
            c
            for c in tab.wire_commands
            if StrCodec.coerce(c.get("method")) == "Fetch.continueRequest"
        ),
        None,
    )
    assert payload is not None, "the request was never continued"
    entries = ListCodec.coerce(DictCodec.coerce(payload.get("params")).get("headers"))
    return {
        StrCodec.coerce(DictCodec.coerce(entry).get("name")).lower(): StrCodec.coerce(
            DictCodec.coerce(entry).get("value")
        )
        for entry in entries
    }


def _continue_override(tab: _FakeTab) -> dict[str, str] | None:
    """Return the header OVERRIDE the guard sent, or ``None`` if it sent none.

    Distinct from :func:`_continued_headers`, which reports what the override
    said: the question here is whether an override was sent AT ALL. That is the
    property under test, because an override is unreliable whatever it carries
    -- ``Cookie`` applies intermittently and no override survives a redirect hop
    (see :func:`~wesearch.fetch.transport.zendriver._continue`) -- so
    sending one at all is the hazard, not any value inside it.
    """
    payload = next(
        (
            c
            for c in tab.wire_commands
            if StrCodec.coerce(c.get("method")) == "Fetch.continueRequest"
        ),
        None,
    )
    assert payload is not None, "the request was never continued"
    params = DictCodec.coerce(payload.get("params"))
    if "headers" not in params:
        return None
    return _continued_headers(tab)


def _extra_http_headers(tab: _FakeTab) -> dict[str, str]:
    """Return the headers installed tab-wide via ``setExtraHTTPHeaders``."""
    payload = next(
        (
            c
            for c in tab.wire_commands
            if StrCodec.coerce(c.get("method")) == "Network.setExtraHTTPHeaders"
        ),
        None,
    )
    if payload is None:
        return {}
    installed = DictCodec.coerce(DictCodec.coerce(payload.get("params")).get("headers"))
    return {name.lower(): StrCodec.coerce(value) for name, value in installed.items()}


def _request_paused(
    url: str, headers: dict[str, str] | None = None
) -> zendriver.cdp.fetch.RequestPaused:
    """A real ``RequestPaused`` for a main-frame document request.

    The genuine CDP dataclass, not a look-alike: the guard filters on
    ``isinstance``, so a stand-in would satisfy the fake and be ignored in
    production -- the direction a test must never fail in.

    ``headers`` are the ones Chrome reports ALREADY on the paused request,
    which include whatever ``set_extra_http_headers`` installed on the tab --
    the replay this guard exists to trim.
    """
    request = zendriver.cdp.network.Request(
        url=url,
        method="GET",
        headers=zendriver.cdp.network.Headers(headers or {}),
        initial_priority=zendriver.cdp.network.ResourcePriority.HIGH,
        referrer_policy="no-referrer",
    )
    return zendriver.cdp.fetch.RequestPaused(
        request_id=zendriver.cdp.fetch.RequestId("req-1"),
        request=request,
        frame_id=zendriver.cdp.page.FrameId("main"),
        resource_type=zendriver.cdp.network.ResourceType.DOCUMENT,
        response_error_reason=None,
        response_status_code=None,
        response_status_text=None,
        response_headers=None,
        network_id=None,
        redirected_request_id=None,
    )


class TestBrowserHonorsTrustPerHop:
    """Chrome follows redirects itself, so each hop must be checked before it runs.

    The header transports re-validate every hop (``curl.py:364``,
    ``stdlib.py:213``) because ``common.py`` states the rule: "A redirect target
    is a URL like any other and must be re-checked; skipping that is the classic
    SSRF bypass." The browser leg reached the same rule through a different
    door -- it validated the URL the CALLER passed and then handed navigation to
    Chrome, which fetched every subsequent hop with nothing watching.
    """

    def test_fetch_zendriver_accepts_trust(self) -> None:
        # The signature IS the enforcement: a transport that cannot express the
        # policy cannot be held to it, and no reviewer of the call site sees the
        # gap because that line does call ``pinned_host``.
        assert "trust" in inspect.signature(fz_mod.fetch_zendriver).parameters

    @staticmethod
    def _run(
        browser: _FakeBrowser,
        *,
        url: str = "https://public.example/start",
        trust: Trust = "untrusted",
        on_redirect: Any = None,
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
            )
        )
        assert browser.last_tab is not None
        return browser.last_tab

    def test_a_hop_adding_no_header_keeps_chromes_own_header_set(
        self, monkeypatch: pytest.MonkeyPatch
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
                _request_paused("https://public.example/next", {"accept": "text/html"})
            ]
        )
        _patch_pool(monkeypatch, browser)
        tab = self._run(browser)
        assert tab.continued_requests == ["req-1"]
        assert _continue_override(tab) is None

    def test_a_cross_origin_hop_drops_the_credential_without_rewriting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Withholding a header is not a reason to replace Chrome's set.

        The credential is dropped by never installing it tab-wide (see
        ``test_origin_bound_headers_are_never_installed_tab_wide``), so Chrome
        never had it on this hop to begin with. Overriding here would add the
        bot-detection tell above while removing nothing.
        """
        browser = _FakeBrowser(
            paused_events=[
                _request_paused("https://evil.example/steal", {"accept": "text/html"})
            ]
        )
        _patch_pool(monkeypatch, browser)
        tab = self._run(
            browser,
            headers={"Authorization": "Bearer secret", "Accept": "text/html"},
        )
        assert _continue_override(tab) is None

    def test_caller_headers_do_not_cross_an_origin_boundary(
        self, monkeypatch: pytest.MonkeyPatch
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
                _request_paused("https://evil.example/steal", {"accept": "text/html"})
            ]
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
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Same origin is what the header was seeded FOR; stripping there would
        # break every authenticated fetch that redirects internally.
        browser = _FakeBrowser(
            paused_events=[_request_paused("https://public.example/next")]
        )
        _patch_pool(monkeypatch, browser)
        tab = self._run(browser, headers={"Authorization": "Bearer secret"})
        assert "authorization" in _continued_headers(tab)

    def test_an_entitled_header_replaces_chromes_row_rather_than_adding_one(
        self, monkeypatch: pytest.MonkeyPatch
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
                _request_paused("https://public.example/next", {"Cookie": "chrome=own"})
            ]
        )
        _patch_pool(monkeypatch, browser)
        tab = self._run(browser, headers={"cookie": "caller=seeded"})
        payload = next(
            c
            for c in tab.wire_commands
            if StrCodec.coerce(c.get("method")) == "Fetch.continueRequest"
        )
        entries = ListCodec.coerce(
            DictCodec.coerce(payload.get("params")).get("headers")
        )
        names = [
            StrCodec.coerce(DictCodec.coerce(e).get("name")).lower() for e in entries
        ]
        assert names.count("cookie") == 1, f"duplicate Cookie row: {names}"
        # The caller's value is the one that must survive: a per-call cookie is
        # an explicit override, matching ``set_session_cookies`` on the curl leg.
        assert _continued_headers(tab)["cookie"] == "caller=seeded"

    def test_origin_bound_headers_are_never_installed_tab_wide(
        self, monkeypatch: pytest.MonkeyPatch
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
        self, monkeypatch: pytest.MonkeyPatch
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
                _request_paused("https://public.example/next", {"accept": "text/html"})
            ]
        )
        _patch_pool(monkeypatch, browser)
        tab = self._run(browser, headers=seeded)
        assert _extra_http_headers(tab).get("sec-ch-ua-model") is None
        assert _continued_headers(tab).get("sec-ch-ua-model") == "Pixel"

    def test_private_redirect_target_is_refused_before_it_is_fetched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        browser = _FakeBrowser(
            paused_events=[_request_paused("http://127.0.0.1:1/secret")]
        )
        _patch_pool(monkeypatch, browser)
        tab = self._run(browser)
        # Failed, not continued: the loopback hop must never reach the socket.
        assert tab.failed_requests == ["req-1"]
        assert tab.continued_requests == []

    def test_public_redirect_target_is_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        browser = _FakeBrowser(
            paused_events=[_request_paused("https://example.com/next")]
        )
        _patch_pool(monkeypatch, browser)
        tab = self._run(browser)
        assert tab.continued_requests == ["req-1"]
        assert tab.failed_requests == []

    def test_internal_trust_permits_a_private_target(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``internal`` is the caller's statement that it authored the URL, so a
        # loopback SearXNG instance must still be reachable through the browser.
        browser = _FakeBrowser(
            paused_events=[_request_paused("http://127.0.0.1:8888/next")]
        )
        _patch_pool(monkeypatch, browser)
        tab = self._run(browser, url="http://127.0.0.1:8888/search", trust="internal")
        assert tab.continued_requests == ["req-1"]
        assert tab.failed_requests == []

    def test_on_redirect_fires_before_the_hop_is_followed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ObserveParams documents on_redirect as "called with the redirect target
        # URL before following; raise to abort". Firing it after the load makes
        # the abort unreachable.
        browser = _FakeBrowser(
            paused_events=[_request_paused("https://example.com/next")]
        )
        _patch_pool(monkeypatch, browser)
        seen: list[str] = []
        tab = self._run(browser, on_redirect=seen.append)
        assert seen == ["https://example.com/next"]
        assert tab.continued_requests == ["req-1"]

    def test_on_redirect_raising_aborts_the_hop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(url: str) -> None:
            del url
            raise RuntimeError("caller refused the hop")

        browser = _FakeBrowser(
            paused_events=[_request_paused("https://example.com/next")]
        )
        _patch_pool(monkeypatch, browser)
        tab = self._run(browser, on_redirect=refuse)
        assert tab.failed_requests == ["req-1"]
        assert tab.continued_requests == []

    def test_interception_is_scoped_to_documents(
        self, monkeypatch: pytest.MonkeyPatch
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
                if StrCodec.coerce(c.get("method")) == "Fetch.enable"
            ),
            None,
        )
        assert enable is not None
        patterns = ListCodec.coerce(
            DictCodec.coerce(enable.get("params")).get("patterns")
        )
        shapes = [DictCodec.coerce(p) for p in patterns]
        assert [StrCodec.coerce(s.get("resourceType")) for s in shapes] == ["Document"]
        assert [StrCodec.coerce(s.get("requestStage")) for s in shapes] == ["Request"]


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
        )
    )
    assert len(browser.cookies.seeded) == 1
    assert browser.cookies.seeded[0].name == "CONSENT"
    assert browser.cookies.seeded[0].value == "YES+"
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
    monkeypatch.setattr(fz_mod, "_pool", lambda: slow_pool)
    with pytest.raises(TimeoutError):
        asyncio.run(
            _navigate(
                "https://example.com/",
                profile_dir=_PROFILE,
                egress="e",
                timeout_sec=0.001,
                headless=True,
                on_redirect=None,
            )
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
    ``fetch_zendriver``'s ``timeout_sec + 30`` (``zendriver.py:296``) before
    ``future.result`` raises a bare, wall-less ``TimeoutError``.

    That is the CI shape: the traceback ended at
    ``concurrent/futures/_base.py:456 raise TimeoutError()`` rather than at
    ``asyncio/timeouts.py __aexit__``, which is what a coroutine reporting its
    own deadline produces.
    """
    browser = _WedgedBrowser()
    _patch_pool(monkeypatch, browser)

    async def go() -> float:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await _navigate(
                "https://example.com/",
                profile_dir=_PROFILE,
                egress="e",
                timeout_sec=0.2,
                headless=True,
                on_redirect=None,
            )
        return time.monotonic() - started

    # The bound that matters is the POOL's, not a round number: the coroutine
    # must finish -- and so raise its own TimeoutError -- before
    # ``fetch_zendriver`` gives up at ``timeout_sec + 30`` and reports a
    # wall-less one instead. Derived from the constants rather than restated, so
    # tuning either budget cannot leave this passing vacuously.
    close_budget = inspect.signature(fz_mod._closed).parameters["budget_sec"].default
    ceiling = 0.2 + close_budget
    elapsed = asyncio.run(asyncio.wait_for(go(), timeout=0.2 + 30))
    assert elapsed < ceiling + 1.0, (
        f"teardown outlived its budget: {elapsed:.2f}s against {ceiling:.2f}s"
    )


def test_navigate_uses_one_overall_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = _FakeBrowser()
    _patch_pool(monkeypatch, browser)

    async def reject_step_timeout(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Per-step timeout resets the request budget.")

    monkeypatch.setattr(asyncio, "wait_for", reject_step_timeout)
    asyncio.run(
        _navigate(
            "https://example.com/",
            profile_dir=_PROFILE,
            egress="e",
            timeout_sec=5.0,
            headless=True,
            on_redirect=None,
        )
    )


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
        )
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
        )
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
        )
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
        )
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
        )
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
        )
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
        )
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
        )
    )
    assert browser.last_tab is not None
    assert browser.last_tab.closed is True


def test_navigate_matches_exact_host_cookie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    browser = _FakeBrowser(
        cookies=[_FakeCookie(name="H", value="1", domain="example.com")]
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
        )
    )
    assert result.cookies == {"H": "1"}


def test_settled_content_returns_promptly_when_the_wall_clears_off_loop() -> None:
    """A challenge that clears must be observed then, not at the deadline.

    zendriver dispatches sync handlers off-loop, and a cross-thread
    ``Event.set()`` does not wake the selector. Asserted on elapsed time: the
    body is identical either way.
    """
    wall = "<html><title>Just a moment...</title></html>"
    handlers: list[Callable[..., None]] = []
    cleared = False

    class _ClearsOffLoopTab:
        """Serves the wall until an off-loop navigation event says otherwise.

        Deliberately not ``_FakeTab``: that one advances its document from
        inside ``get_content``, on the loop thread, which is the arrangement
        that hides a wakeup delivered from anywhere else.
        """

        def add_handler(self, event_type: Any, handler: Any) -> None:
            del event_type
            handlers.append(cast("Callable[..., None]", handler))

        async def get_content(self) -> str:
            return "<html>ok</html>" if cleared else wall

        async def wait_for_ready_state(self, until: str = "complete") -> bool:
            del until
            return True

    async def go() -> float:
        tab = _ClearsOffLoopTab()
        started = time.monotonic()

        def clear_from_another_thread() -> None:
            nonlocal cleared
            time.sleep(0.1)  # Let the settle wait park the loop.
            cleared = True
            handlers[0](_main_frame_navigated())

        threading.Thread(target=clear_from_another_thread, daemon=True).start()
        await fz_mod._settled_content(cast(Any, tab), budget_sec=5.0)
        return time.monotonic() - started

    # Cleared at 0.1s against a 5s budget: finishing under 2s proves the wakeup
    # arrived, while waiting the budget out lands at ~5s.
    assert asyncio.run(go()) < 2.0


def test_settled_content_bounds_a_stalled_document_parse() -> None:
    """``budget_sec`` must bound the whole settle, parse included.

    A document that commits near the deadline and then stalls would otherwise
    spend the caller's entire request timeout in ``wait_for_ready_state``.
    """
    wall = "<html><title>Just a moment...</title></html>"

    class _StalledParseTab:
        def add_handler(self, event_type: Any, handler: Any) -> None:
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
        await fz_mod._settled_content(cast(Any, _StalledParseTab()), budget_sec=0.2)
        return time.monotonic() - started

    # Generous multiple of the budget: this must fail on an UNBOUNDED wait, not
    # on scheduler jitter around a bound that is working.
    assert asyncio.run(asyncio.wait_for(go(), timeout=10.0)) < 3.0


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
        )
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
        )
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
        )
    )
    assert seen == []


# -- _BrowserPool: reuse + replacement ---------------------------------------


def test_pool_reuses_browser_per_key(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[_FakeBrowser] = []

    async def fake_launch(
        self: _BrowserPool, profile_dir: Path, *, headless: bool
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
        self: _BrowserPool, profile_dir: Path, *, headless: bool
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
            with pytest.raises(RuntimeError, match="launch mode"):
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
        self: _BrowserPool, profile_dir: Path, *, headless: bool
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
        self: _BrowserPool, profile_dir: Path, *, headless: bool
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
            cast(Any, first).stopped = True  # simulate Chrome exit
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
        def run(self, coro: Any, *, timeout_sec: float = 0) -> BrowserResult:
            coro.close()
            waits.append(timeout_sec)
            return BrowserResult(body=b"", cookies={}, final_url="")

    pool = _RecordingPool()
    monkeypatch.setattr(fz_mod, "_pool", lambda: pool)
    fz_mod.fetch_zendriver(
        "https://example.com/", profile_dir=_PROFILE, egress="e", timeout_sec=30.0
    )

    assert waits == [pytest.approx(60.0)]


def test_launch_survives_a_reply_to_a_cancelled_cdp_transaction() -> None:
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

    async def fake_start(config: Any) -> _FakeBrowser:
        del config
        return _FakeBrowser()

    async def go() -> None:
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(zendriver, "start", fake_start)
            # Restored to the VENDOR's own ``__call__``, undoing any arming an
            # earlier test in this process did: the guard is installed
            # class-wide, so without this the assertions below pass vacuously.
            patcher.setattr(Transaction, "__call__", _VENDOR_TRANSACTION_CALL)
            await fz_mod._launch_browser(_PROFILE, headless=True)

            # ``result`` alone, though the listener splats the whole message:
            # the vendor reads only ``error`` and ``result``, and its
            # ``**response: dict[str, Any]`` annotation rejects the integer
            # ``id`` a real reply also carries.
            cancelled = Transaction(zendriver.cdp.page.navigate("about:blank"))
            cancelled.cancel()
            cancelled(result={"frameId": "F", "loaderId": "L"})
            assert cancelled.cancelled()

            # The guard must drop only what is already settled: a reply to a
            # LIVE transaction still has to reach the caller awaiting it.
            live = Transaction(zendriver.cdp.page.navigate("about:blank"))
            live(result={"frameId": "F", "loaderId": "L"})
            assert cast("tuple[object, ...]", live.result())[0] == (
                zendriver.cdp.page.FrameId("F")
            )

    asyncio.run(go())


def test_pool_shutdown_joins_thread_and_closes_loop() -> None:
    pool = _BrowserPool(serve_control=False)
    pool.shutdown()

    assert not pool._thread.is_alive()
    assert pool._loop.is_closed()


def test_pool_keys_separate_egress(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[_FakeBrowser] = []

    async def fake_launch(
        self: _BrowserPool, profile_dir: Path, *, headless: bool
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
        assert len(launched) == 2  # distinct egress -> distinct browser
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
    monkeypatch.setattr(fz_mod, "cache_dir", lambda: tmp_path / "cache")
    return puppeteer, playwright


def test_non_macos_hosts_need_no_separate_browser() -> None:
    """Linux and Windows keep the ordinary browser -- there is no capture.

    A URL there reaches a browser through ``xdg-open`` and the desktop file
    rather than a running process, and a headless server has only the stock
    Chrome installed anyway.
    """
    assert fz_mod._fetch_browser(platform="linux") == ""
    assert fz_mod._fetch_browser(platform="win32") == ""


def test_non_macos_hosts_launch_with_no_extra_flags() -> None:
    # Chrome on a headless server has no keychain to mock, so a flag leaking
    # onto that path would break the only browser it has.
    assert fz_mod._fetch_browser_args(fz_mod._fetch_browser(platform="linux")) == []


def test_no_install_falls_back_to_zendriver(roots: tuple[Path, Path]) -> None:
    # A miss must cost click capture, never the fetch: "" means "let zendriver
    # find Chrome".
    del roots

    assert fz_mod._fetch_browser(platform="darwin") == ""


def test_a_puppeteer_install_is_found(roots: tuple[Path, Path]) -> None:
    expected = _fetch_browser_install(roots[0], "mac_arm-147.0.7727.57")

    assert fz_mod._fetch_browser(platform="darwin") == str(expected)


def test_a_playwright_install_is_found(roots: tuple[Path, Path]) -> None:
    expected = _fetch_browser_install(roots[1], "chromium-1217")

    assert fz_mod._fetch_browser(platform="darwin") == str(expected)


def test_the_newest_build_wins(roots: tuple[Path, Path]) -> None:
    # Builds accumulate across upgrades; a stale one eventually cannot read a
    # profile the current browser wrote.
    _ = _fetch_browser_install(roots[0], "mac_arm-131.0.6778.85")
    newest = _fetch_browser_install(roots[0], "mac_arm-147.0.7727.57")

    assert fz_mod._fetch_browser(platform="darwin") == str(newest)


@pytest.mark.parametrize(
    ("older", "newer"),
    [
        ("mac_arm-99.0.4844.51", "mac_arm-100.0.4896.60"),
        ("chromium-999", "chromium-1000"),
    ],
)
def test_build_order_is_numeric_not_lexical(
    roots: tuple[Path, Path], older: str, newer: str
) -> None:
    # Text comparison ranks "99" above "100", which silently pins fetches to
    # the older browser at every version rollover past a digit boundary.
    _ = _fetch_browser_install(roots[0], older)
    expected = _fetch_browser_install(roots[0], newer)

    assert fz_mod._fetch_browser(platform="darwin") == str(expected)


def test_unversioned_build_directories_still_order_deterministically() -> None:
    # Filesystem iteration order is arbitrary, so names carrying no digits
    # must not leave the choice to it. Ranked directly rather than through a
    # fixture: a fixture would have to observe the very order in question.
    unversioned = [Path("beta"), Path("alpha"), Path("dev")]

    ranked = sorted(unversioned, key=fz_mod._build_order, reverse=True)

    assert [path.name for path in ranked] == ["dev", "beta", "alpha"]


def test_a_directory_without_the_binary_is_skipped(roots: tuple[Path, Path]) -> None:
    # Playwright's roots include ``ffmpeg-*`` and other non-browser payloads,
    # and an interrupted download leaves a build directory with no binary.
    (roots[1] / "ffmpeg-1011").mkdir(parents=True)
    expected = _fetch_browser_install(roots[1], "chromium-1217")

    assert fz_mod._fetch_browser(platform="darwin") == str(expected)


def test_an_intel_download_is_found(roots: tuple[Path, Path]) -> None:
    # The only build on an Intel Mac, and runnable under Rosetta on Apple
    # silicon, so keying discovery to the host arch would skip a usable one.
    expected = _fetch_browser_install(
        roots[0], "mac-147.0.7727.57", arch="chrome-mac-x64"
    )

    assert fz_mod._fetch_browser(platform="darwin") == str(expected)


def test_chrome_for_testing_gets_the_mock_keychain_flag() -> None:
    """Pinned verbatim: a typo yields a silent hang, not an error.

    Chrome ignores an unknown flag rather than rejecting it, so a misspelling
    restores the keychain prompt and the launch blocks behind it.
    """
    assert fz_mod._fetch_browser_args("/cache/ChromeForTesting") == [
        "--use-mock-keychain"
    ]


def test_stock_chrome_keeps_its_own_keychain() -> None:
    # It owns a real keychain entry, and mocking that cuts it off from cookies
    # it legitimately has.
    assert fz_mod._fetch_browser_args("") == []


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
