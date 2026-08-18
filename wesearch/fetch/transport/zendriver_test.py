"""Tests for ``wesearch.fetch.transport.zendriver`` (zendriver headless fetch backend).

Hermetic: a fake async browser stands in for zendriver, so the transport logic
(cookie-domain filtering, challenge detection, redirect callback, pool reuse)
is exercised with no Chrome and no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import asyncio
import atexit
import importlib
import subprocess
import tempfile

import pytest
import zendriver

from wesearch.fetch.transport.zendriver import (
    BrowserResult,
    _BrowserPool,
    _navigate,
)

import wesearch.fetch.transport.zendriver as fz_mod


# A fake profile dir; the browser is mocked in every test, so it is never
# touched on disk.
_PROFILE = Path("test-profile")


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
    ) -> None:
        self._documents = documents if documents is not None else [content]
        self._parsing = parsing
        self._href = href
        self.closed = False
        self.navigations: list[str] = []
        self.commands: list[Any] = []
        self.handlers: list[Any] = []
        self.content_reads = 0
        self._index = 0
        # True between a navigation and its ready-state wait -- the window in
        # which the new document exists but has not parsed.
        self._is_parsing = False

    def add_handler(self, event_type: Any, handler: Any) -> None:
        del event_type
        self.handlers.append(handler)

    def _navigate_main_frame(self) -> None:
        """Advance to the next document and notify the transport's handler."""
        if self._index + 1 < len(self._documents):
            self._index += 1
        self._is_parsing = self._parsing is not None
        for handler in self.handlers:
            handler(_main_frame_navigated())

    async def send(self, command: Any) -> None:
        self.commands.append(command)

    async def get(self, url: str) -> _FakeTab:
        self.navigations.append(url)
        if not self._href:
            self._href = url
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
    ) -> None:
        self._content = content
        self._href = href
        self._documents = documents
        self._parsing = parsing
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

    async def fake_start(config: Any) -> _FakeBrowser:
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

    async def fake_start(config: Any) -> _FakeBrowser:
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

    async def fake_start(config: Any) -> _FakeBrowser:
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
        return cast("zendriver.Browser", launched[-1])

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
        assert not cast("_FakeBrowser", held).stopped
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
    assert len(browser.last_tab.commands) == 2


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
            timeout_sec=0.2,
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
            timeout_sec=0.2,
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
            timeout_sec=0.2,  # An unclearable wall; see the note above.
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


# -- _navigate: redirect callback --------------------------------------------


def test_navigate_fires_on_redirect_when_final_url_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    browser = _FakeBrowser(href="https://example.com/landing")
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
            cast("Any", first).stopped = True  # simulate Chrome exit
            second = await pool.browser("e", _PROFILE, headless=True)
            assert second is not first

        pool.run(go())
        assert len(launched) == 2
    finally:
        pool.shutdown()


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


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
