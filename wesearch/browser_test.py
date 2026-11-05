"""Persistent browser actions against an isolated driver boundary."""

from __future__ import annotations

from concurrent import futures
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, NotRequired, TypedDict, TypeVar

import asyncio
import base64
import os
import selectors
import shutil
import subprocess
import sys
import threading
import time

from treekle import from_plain
from zendriver import Browser, cdp
from zendriver.cdp.page import DomContentEventFired
from zendriver.core.keys import KeyEvents, KeyModifiers, SpecialKeys

import pytest

from wesearch import browser as api
from wesearch.fetch.transport import zendriver
from wesearch.fetch.types import FetchSession


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Coroutine, Generator
    from pathlib import Path


_T = TypeVar("_T")


class _Element:
    def __init__(self) -> None:
        self.clicked = 0
        self.node_name = "INPUT"
        self.attrs: dict[str, str] = {"type": "email"}
        self.keys: list[SpecialKeys | list[KeyEvents.Payload]] = []

    async def click(self) -> None:
        self.clicked += 1

    async def send_keys(self, key: SpecialKeys | list[KeyEvents.Payload]) -> None:
        self.keys.append(key)


class _Tab:
    def __init__(self, tab_id: str, url: str) -> None:
        self.target_id = tab_id
        self.target = SimpleNamespace(title="Signup")
        self.url = url
        self.closed = False
        self.element = _Element()
        self.matches = 1
        self.ready = True
        self.ready_state = "interactive"
        self.ready_handlers: list[Callable[..., object]] = []
        self.image = base64.b64encode(b"png-image").decode()
        self.commands: list[dict[str, object]] = []
        self.snapshot: dict[str, object] = {
            "url": url,
            "title": "Signup",
            "text": "Signup form",
            "elements": [
                {
                    "selector": "#email",
                    "tag": "input",
                    "role": "textbox",
                    "label": "Email",
                    "kind": "email",
                },
            ],
        }

    async def get(self, url: str) -> _Tab:
        self.url = url
        self.snapshot["url"] = url
        return self

    async def evaluate(self, expression: str) -> object:
        if expression == "document.readyState":
            return self.ready_state if self.ready else "loading"
        assert expression == api._INSPECT
        return self.snapshot

    def add_handler(self, event_type: object, handler: Callable[..., object]) -> None:
        assert event_type is DomContentEventFired
        self.ready_handlers.append(handler)

    def remove_handlers(
        self,
        event_type: object,
        handler: Callable[..., object],
    ) -> None:
        assert event_type is DomContentEventFired
        self.ready_handlers.remove(handler)

    async def query_selector_all(self, selector: str) -> list[_Element]:
        assert selector == "#email"
        return [self.element] * self.matches

    async def send(self, command: Generator[dict[str, object], object, object]) -> None:
        self.commands.append(next(command))

    async def screenshot_b64(self, *, format: str) -> str:
        assert format == "png"
        return self.image

    async def close(self) -> None:
        self.closed = True


class _Browser:
    def __init__(self) -> None:
        self.tabs: list[_Tab] = [_Tab("existing", "https://example.com/account")]

    async def get(self, url: str, *, new_tab: bool) -> _Tab:
        assert new_tab
        assert url == "about:blank"
        tab = _Tab(f"tab-{len(self.tabs)}", url)
        self.tabs.append(tab)
        return tab

    async def update_targets(self) -> None:
        self.tabs = [tab for tab in self.tabs if not tab.closed]


class _Pool:
    def __init__(self) -> None:
        self.chrome = _Browser()
        self.targets_lock = asyncio.Lock()
        self.locks: dict[str, asyncio.Lock] = {}
        self.attached: list[tuple[Path, bool]] = []
        self.guard_calls: list[dict[str, object]] = []
        self.run_timeouts: list[float] = []
        self.check_inner_timeout = False
        self.failure: Exception | None = None
        self.mode_calls: list[tuple[Path, bool | None, bool]] = []
        self.headless = True

    async def mode(
        self,
        profile_dir: Path,
        *,
        headless: bool | None,
        restart: bool,
    ) -> tuple[bool, bool]:
        self.mode_calls.append((profile_dir, headless, restart))
        changed = headless is not None and headless != self.headless
        if changed and not restart:
            raise zendriver.BrowserUnavailableError(
                "Changing mode requires restart=True.",
            )
        if headless is not None:
            self.headless = headless
        return self.headless, changed

    def run(self, coro: Coroutine[object, object, _T], *, timeout_sec: float) -> _T:
        assert timeout_sec > 0
        self.run_timeouts.append(timeout_sec)
        if self.failure is not None:
            coro.close()
            raise self.failure
        if self.check_inner_timeout:

            async def bounded() -> _T:
                async with asyncio.timeout(0.5) as outer:
                    try:
                        return await coro
                    finally:
                        assert not outer.expired(), (
                            "Action did not enforce its own deadline"
                        )

            return asyncio.run(bounded())
        return asyncio.run(coro)

    async def browser(self, profile_dir: Path, *, headless: bool) -> _Browser:
        self.attached.append((profile_dir, headless))
        return self.chrome

    def tab_lock(self, tab_id: str) -> asyncio.Lock:
        assert any(tab.target_id == tab_id for tab in self.chrome.tabs)
        return self.locks.setdefault(tab_id, asyncio.Lock())

    @asynccontextmanager
    async def guard(self, tab: _Tab, **kwargs: object) -> AsyncGenerator[_Pool]:
        deadline = kwargs["deadline"]
        assert isinstance(deadline, float)
        assert deadline > asyncio.get_running_loop().time()
        self.guard_calls.append({"target": tab.target_id, **kwargs})
        yield self

    async def drain(self) -> None:
        pass


@pytest.fixture
def driver(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[_Pool, FetchSession]:
    pool = _Pool()
    session = FetchSession(profile_dir=tmp_path / "profile", headless=False)

    def get_pool(requested: FetchSession) -> _Pool:
        assert requested is session
        return pool

    monkeypatch.setattr(zendriver, "get_browser_pool", get_pool, raising=False)
    monkeypatch.setattr(zendriver, "guard_requests", pool.guard, raising=False)
    monkeypatch.setattr(api, "pinned_host", _public_host, raising=False)
    return pool, session


def _public_host(url: str, *, allow_private_hosts: bool) -> str:
    assert allow_private_hosts is False
    if not url.startswith("https://example.com/"):
        raise ValueError("not a public URL")
    return "example.com"


@pytest.mark.parametrize("mode", ["headed", "headless"])
def test_mode_reports_actual_state_and_explicitly_switches(
    driver: tuple[_Pool, FetchSession],
    mode: api.BrowserMode,
) -> None:
    pool, session = driver
    assert api.browser("mode", session=session).mode == "headless"
    switched = api.browser("mode", mode=mode, restart=True, session=session)
    assert switched.mode == mode
    assert switched.restarted is (mode == "headed")
    assert api.browser("mode", session=session).mode == mode
    assert pool.mode_calls == [
        (session.profile_dir, None, False),
        (session.profile_dir, mode == "headless", True),
        (session.profile_dir, None, False),
    ]
    assert not pool.attached
    assert not pool.guard_calls


def test_mode_refuses_restart_without_explicit_consent(
    driver: tuple[_Pool, FetchSession],
) -> None:
    pool, session = driver
    with pytest.raises(api.BrowserError, match="restart=True"):
        api.browser("mode", mode="headed", session=session)
    assert pool.headless


@pytest.mark.parametrize(
    "arguments",
    [
        {"action": "tabs", "mode": "headed"},
        {"action": "tabs", "restart": True},
        {"action": "mode", "restart": True},
        {"action": "mode", "tab_id": "existing"},
    ],
)
def test_mode_arguments_are_action_specific(
    driver: tuple[_Pool, FetchSession],
    arguments: dict[str, object],
) -> None:
    pool, session = driver
    args = from_plain(arguments, _ModeArguments)
    with pytest.raises(ValueError, match=r"accepted|requires|required"):
        api.browser(
            args["action"],
            tab_id=args.get("tab_id", ""),
            mode=args.get("mode"),
            restart=args.get("restart", False),
            session=session,
        )
    assert not pool.mode_calls
    assert not pool.attached


class _ModeArguments(TypedDict):
    action: api.BrowserAction
    tab_id: NotRequired[str]
    mode: NotRequired[api.BrowserMode]
    restart: NotRequired[bool]


def test_tabs_open_and_inspect_keep_the_existing_browser_and_user_tab(
    driver: tuple[_Pool, FetchSession],
) -> None:
    pool, session = driver
    initial = api.browser("tabs", session=session)
    assert [tab.tab_id for tab in initial.tabs] == ["existing"]
    opened = api.browser("open", url="https://example.com/signup", session=session)
    assert opened.tab is not None
    assert opened.tab.tab_id == "tab-1"
    assert opened.tab.title == "Signup"
    assert not any(tab.closed for tab in pool.chrome.tabs)
    inspected = api.browser("inspect", tab_id=opened.tab.tab_id, session=session)
    assert inspected.tab == opened.tab
    assert inspected.text == "Signup form"
    assert inspected.elements[0].selector == "#email"
    assert all(
        path == session.profile_dir and headless is False
        for path, headless in pool.attached
    )


def test_navigate_and_close_affect_only_the_named_tab(
    driver: tuple[_Pool, FetchSession],
) -> None:
    pool, session = driver
    result = api.browser(
        "navigate",
        tab_id="existing",
        url="https://example.com/next",
        session=session,
    )
    assert result.tab is not None
    assert result.tab.url == "https://example.com/next"
    assert pool.guard_calls[-1]["initial_url"] == "https://example.com/next"
    opened = api.browser("open", url="https://example.com/signup", session=session)
    assert opened.tab is not None
    closed = api.browser("close", tab_id=opened.tab.tab_id, session=session)
    assert closed.tab == opened.tab
    assert pool.chrome.tabs[1].closed
    assert not pool.chrome.tabs[0].closed


@pytest.mark.parametrize("platform", ["linux", "darwin"])
@pytest.mark.parametrize(
    "kind",
    ["", "text", "search", "tel", "url", "email", "password", "number", "textarea"],
)
def test_fill_uses_native_text_input_not_interpolated_javascript(
    driver: tuple[_Pool, FetchSession],
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    kind: str,
) -> None:
    pool, session = driver
    monkeypatch.setattr(sys, "platform", platform)
    element = pool.chrome.tabs[0].element
    element.node_name = "TEXTAREA" if kind == "textarea" else "INPUT"
    element.attrs = {} if kind == "textarea" else {"type": kind}
    payload = "quote'\";window.hacked=true;\\"
    api.browser(
        "fill",
        tab_id="existing",
        selector="#email",
        text=payload,
        session=session,
    )
    tab = pool.chrome.tabs[0]
    assert tab.commands[-1] == {
        "method": "Input.insertText",
        "params": {"text": payload},
    }
    assert tab.element.keys[-1] == SpecialKeys.BACKSPACE
    select_all = tab.element.keys[0]
    assert isinstance(select_all, list)
    a_events = [event for event in select_all if event["key"] == "a"]
    assert [event["type_"] for event in a_events] == ["keyDown", "keyUp"]
    modifier = KeyModifiers.Meta if sys.platform == "darwin" else KeyModifiers.Ctrl
    assert a_events[0]["modifiers"] == modifier
    assert a_events[1]["modifiers"] == 0
    assert select_all[0]["key"] == ("Meta" if sys.platform == "darwin" else "Control")
    assert select_all[-2]["modifiers"] == 0
    api.browser("fill", tab_id="existing", selector="#email", text="", session=session)
    assert tab.commands[-1] == {"method": "Input.insertText", "params": {"text": ""}}


def test_click_and_press_reach_the_selected_control(
    driver: tuple[_Pool, FetchSession],
) -> None:
    pool, session = driver
    api.browser("click", tab_id="existing", selector="#email", session=session)
    assert pool.chrome.tabs[0].element.clicked == 1
    api.browser(
        "press",
        tab_id="existing",
        selector="#email",
        key="Enter",
        session=session,
    )
    assert len(pool.chrome.tabs[0].element.keys) == 1


@pytest.mark.parametrize("matches", [0, 2])
def test_controls_must_match_exactly_one_element(
    driver: tuple[_Pool, FetchSession],
    matches: int,
) -> None:
    pool, session = driver
    pool.chrome.tabs[0].matches = matches
    with pytest.raises(api.BrowserError, match="exactly one") as caught:
        api.browser("click", tab_id="existing", selector="#email", session=session)
    assert (
        str(caught.value)
        == "Browser selector must match exactly one element; inspect the tab again."
    )
    assert not pool.chrome.tabs[0].element.clicked


def test_screenshot_returns_bytes_without_writing_a_file(
    driver: tuple[_Pool, FetchSession],
) -> None:
    _, session = driver
    result = api.browser("screenshot", tab_id="existing", session=session)
    assert result.screenshot == b"png-image"
    assert result.tab is not None
    assert result.tab.tab_id == "existing"


def test_inspection_is_bounded(driver: tuple[_Pool, FetchSession]) -> None:
    pool, session = driver
    pool.chrome.tabs[0].snapshot["text"] = "abcdefgh"
    result = api.browser("inspect", tab_id="existing", max_chars=3, session=session)
    assert result.text == "abc"
    assert result.truncated


def test_unknown_target_never_falls_back_to_the_active_tab(
    driver: tuple[_Pool, FetchSession],
) -> None:
    pool, session = driver
    with pytest.raises(api.BrowserError, match="not found"):
        api.browser("inspect", tab_id="missing", session=session)
    assert not pool.chrome.tabs[0].closed


@pytest.mark.parametrize(
    "arguments",
    [
        {"action": "inspect"},
        {"action": "open"},
        {"action": "tabs", "tab_id": "existing"},
        {"action": "tabs", "url": "https://example.com/"},
        {"action": "click", "tab_id": "existing"},
        {
            "action": "press",
            "tab_id": "existing",
            "selector": "#email",
            "key": "BadKey",
        },
        {"action": "inspect", "tab_id": "existing", "text": "payload"},
        {"action": "tabs", "max_chars": 0},
        {"action": "tabs", "timeout_sec": 0},
        {"action": "tabs", "timeout_sec": float("nan")},
    ],
)
def test_invalid_actions_fail_before_browser_attachment(
    driver: tuple[_Pool, FetchSession],
    arguments: dict[str, object],
) -> None:
    pool, session = driver
    args = from_plain(arguments, _Arguments)
    with pytest.raises(ValueError, match=r"required|accepted|supported|positive"):
        api.browser(
            args["action"],
            tab_id=args.get("tab_id", ""),
            url=args.get("url", ""),
            selector=args.get("selector", ""),
            text=args.get("text", ""),
            key=args.get("key", ""),
            max_chars=args.get("max_chars", 20_000),
            timeout_sec=args.get("timeout_sec", 30),
            session=session,
        )
    assert not pool.attached


class _Arguments(TypedDict):
    action: api.BrowserAction
    tab_id: NotRequired[str]
    url: NotRequired[str]
    selector: NotRequired[str]
    text: NotRequired[str]
    key: NotRequired[str]
    max_chars: NotRequired[int]
    timeout_sec: NotRequired[float]


def test_private_navigation_is_refused_before_attachment(
    driver: tuple[_Pool, FetchSession],
) -> None:
    pool, session = driver
    with pytest.raises(ValueError, match="public URL"):
        api.browser("open", url="http://127.0.0.1/", session=session)
    assert not pool.attached


def test_actions_apply_the_shared_request_guard(
    driver: tuple[_Pool, FetchSession],
) -> None:
    pool, session = driver
    api.browser("click", tab_id="existing", selector="#email", session=session)
    [guard] = pool.guard_calls
    assert guard["initial_url"] == "https://example.com/account"
    assert guard["allow_private_hosts"] is False
    assert guard["headers"] == {}


@pytest.mark.parametrize("failure", [TimeoutError("private"), RuntimeError("private")])
def test_driver_failures_do_not_echo_request_or_exception_contents(
    driver: tuple[_Pool, FetchSession],
    failure: Exception,
) -> None:
    pool, session = driver
    pool.failure = failure
    with pytest.raises(api.BrowserError, match="Browser inspect") as caught:
        api.browser("inspect", tab_id="existing", session=session)
    assert "private" not in str(caught.value)
    assert ("RuntimeError" in str(caught.value)) == isinstance(failure, RuntimeError)


def test_default_inspection_budget(driver: tuple[_Pool, FetchSession]) -> None:
    pool, session = driver
    pool.chrome.tabs[0].snapshot["text"] = "x" * 20_001
    result = api.browser("inspect", tab_id="existing", session=session)
    assert result.text == "x" * 20_000
    assert result.truncated
    assert pool.run_timeouts == [60]


def test_already_complete_page_is_ready(driver: tuple[_Pool, FetchSession]) -> None:
    pool, session = driver
    pool.chrome.tabs[0].ready_state = "complete"
    result = api.browser(
        "inspect",
        tab_id="existing",
        timeout_sec=0.02,
        session=session,
    )
    assert result.text == "Signup form"


def test_loading_page_waits_for_dom_ready_and_removes_handler(
    driver: tuple[_Pool, FetchSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, session = driver
    evaluate = _Tab.evaluate

    async def loading(self: _Tab, expression: str) -> object:
        if expression == "document.readyState":
            assert len(self.ready_handlers) == 1
            callback = self.ready_handlers[0](object())
            assert asyncio.iscoroutine(callback)
            await callback
            return "loading"
        return await evaluate(self, expression)

    monkeypatch.setattr(_Tab, "evaluate", loading)
    assert (
        api.browser("inspect", tab_id="existing", session=session).text == "Signup form"
    )
    assert not pool.chrome.tabs[0].ready_handlers


def test_failed_existing_tab_is_preserved(driver: tuple[_Pool, FetchSession]) -> None:
    pool, session = driver
    pool.chrome.tabs[0].ready = False
    with pytest.raises(api.BrowserError, match="timed out"):
        api.browser("inspect", tab_id="existing", timeout_sec=0.01, session=session)
    assert not pool.chrome.tabs[0].closed


def test_failed_new_tab_is_closed_without_closing_user_tabs(
    driver: tuple[_Pool, FetchSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, session = driver

    async def unready(self: _Tab, expression: str) -> str:
        del self, expression
        return "loading"

    monkeypatch.setattr(_Tab, "evaluate", unready)
    with pytest.raises(api.BrowserError, match="timed out"):
        api.browser(
            "open",
            url="https://example.com/signup",
            timeout_sec=0.01,
            session=session,
        )
    assert pool.chrome.tabs[1].closed
    assert not pool.chrome.tabs[0].closed


def test_default_session_is_shared(
    driver: tuple[_Pool, FetchSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, session = driver
    monkeypatch.setattr(api, "default_session", lambda: session)
    assert api.browser("inspect", tab_id="existing").text == "Signup form"


def test_corrupt_screenshot_is_not_returned_as_an_image(
    driver: tuple[_Pool, FetchSession],
) -> None:
    pool, session = driver
    pool.chrome.tabs[0].image = "@" + pool.chrome.tabs[0].image
    with pytest.raises(api.BrowserError, match="Browser screenshot failed"):
        api.browser("screenshot", tab_id="existing", session=session)


def test_exactly_bounded_inspection_is_not_truncated(
    driver: tuple[_Pool, FetchSession],
) -> None:
    pool, session = driver
    pool.chrome.tabs[0].snapshot["text"] = "abc"
    assert not api.browser(
        "inspect",
        tab_id="existing",
        max_chars=3,
        session=session,
    ).truncated


def test_action_deadline_interrupts_a_stalled_driver(
    driver: tuple[_Pool, FetchSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, session = driver
    pool.check_inner_timeout = True

    async def stalled(self: _Tab, expression: str) -> str:
        del self, expression
        await asyncio.Event().wait()
        return "complete"

    monkeypatch.setattr(_Tab, "evaluate", stalled)
    with pytest.raises(api.BrowserError, match="timed out"):
        api.browser("inspect", tab_id="existing", timeout_sec=0.01, session=session)
    assert not pool.chrome.tabs[0].closed


@pytest.mark.parametrize(
    "key",
    [
        "Enter",
        "Tab",
        "Escape",
        "Backspace",
        "Delete",
        "ArrowLeft",
        "ArrowRight",
        "ArrowUp",
        "ArrowDown",
    ],
)
def test_press_preserves_the_requested_native_key(
    driver: tuple[_Pool, FetchSession],
    key: str,
) -> None:
    pool, session = driver
    api.browser("press", tab_id="existing", selector="#email", key=key, session=session)
    [pressed] = pool.chrome.tabs[0].element.keys
    assert isinstance(pressed, SpecialKeys)
    assert pressed.value[0] == key


@pytest.mark.parametrize(
    ("node_name", "attributes"),
    [
        ("BUTTON", {}),
        ("INPUT", {"disabled": ""}),
        ("TEXTAREA", {"readonly": ""}),
        ("INPUT", {"type": "file"}),
    ],
)
def test_fill_refuses_noneditable_controls_without_typing(
    driver: tuple[_Pool, FetchSession],
    node_name: str,
    attributes: dict[str, str],
) -> None:
    pool, session = driver
    element = pool.chrome.tabs[0].element
    element.node_name = node_name
    element.attrs = attributes
    with pytest.raises(api.BrowserError, match="editable") as caught:
        api.browser(
            "fill",
            tab_id="existing",
            selector="#email",
            text="value",
            session=session,
        )
    assert str(caught.value) == "Fill requires an editable text input or textarea."
    assert not element.keys
    assert not pool.chrome.tabs[0].commands


@pytest.fixture
def headed_display(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Give headed Chrome an isolated X server on displayless Linux runners."""
    if not sys.platform.startswith("linux") or os.environ.get("DISPLAY", ""):
        yield
        return
    executable = shutil.which("Xvfb")
    if executable is None:
        pytest.fail("Headed Chrome requires DISPLAY or Xvfb on Linux.")
    with subprocess.Popen(  # noqa: S603 -- Xvfb from PATH with fixed arguments.
        [
            executable,
            "-displayfd",
            "1",
            "-screen",
            "0",
            "1280x1024x24",
            "-nolisten",
            "tcp",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ) as display:
        try:
            assert display.stdout is not None
            with selectors.DefaultSelector() as ready:
                ready.register(display.stdout, selectors.EVENT_READ)
                if not ready.select(timeout=10):
                    pytest.fail("Xvfb did not report a display within 10 seconds.")
            number = os.read(display.stdout.fileno(), 32).decode().strip()
            if not number.isdecimal():
                pytest.fail("Xvfb exited without reporting a display number.")
            monkeypatch.setenv("DISPLAY", f":{number}")
            yield
        finally:
            display.terminate()
            display.wait(timeout=10)


@pytest.mark.browser_zendriver
@pytest.mark.usefixtures("headed_display")
@pytest.mark.parametrize("interrupt", [False, True])
def test_real_cli_mode_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    interrupt: bool,
) -> None:
    profile = tmp_path / "chrome"
    watching = threading.Event()
    release = threading.Event()
    watch_window = zendriver._watch_window
    cancel_watch: list[Callable[[], object]] = []

    async def watch(url: str, browser: Browser) -> set[str]:
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        assert isinstance(task, asyncio.Task)
        cancel_watch.append(lambda: loop.call_soon_threadsafe(task.cancel))
        watching.set()
        if interrupt:
            assert await asyncio.to_thread(release.wait, 20)
            raise asyncio.CancelledError
        return await watch_window(url, browser)

    monkeypatch.setattr(zendriver, "_watch_window", watch)
    monkeypatch.setattr(zendriver, "challenged_urls", list)
    with FetchSession(profile_dir=profile) as sibling:
        pool = zendriver.get_browser_pool(sibling)
        original = pool.run(pool.browser(profile, headless=True))
        pool.run(_seed_mode_cookie(pool, sibling))
        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            running = executor.submit(
                zendriver.open_window,
                "about:blank",
                profile_dir=profile,
            )
            try:
                # Chrome's launch alone may take 60s (zendriver._launch's budget).
                assert watching.wait(90)
                assert not running.done()
                assert api.browser("mode", session=sibling).mode == "headed"
                headed = pool.run(pool.browser(profile, headless=True))
                assert headed is not original
                assert pool.run(_read_mode_cookie(pool, sibling)) == "retained"
                port = zendriver._devtools_port(profile)
                assert port is not None
                if interrupt:
                    release.set()
                    with pytest.raises(futures.CancelledError):
                        running.result(timeout=10)
                else:
                    pool.run(headed.stop(), timeout_sec=10)
                    assert running.result(timeout=10) is None
                assert not zendriver._port_answers(port)
                assert headed.connection is not None
                assert headed.connection.listener is not None
                assert not headed.connection.listener.running
                restored = pool.run(pool.browser(profile, headless=True))
                assert restored is not headed
                assert api.browser("mode", session=sibling).mode == "headless"
            finally:
                release.set()
                if not running.done():
                    for cancel in cancel_watch:
                        cancel()
                    running.exception(timeout=40)


@pytest.mark.browser_zendriver
@pytest.mark.usefixtures("headed_display")
def test_real_mode_roundtrip_preserves_profile_and_reconnects(tmp_path: Path) -> None:
    with (
        FetchSession(profile_dir=tmp_path / "chrome") as session,
        FetchSession(profile_dir=tmp_path / "chrome") as sibling,
    ):
        assert api.browser("mode", session=session).mode == "headless"
        original = api.browser("tabs", session=sibling)
        assert original.tabs
        pool = zendriver.get_browser_pool(sibling)
        pool.run(_seed_mode_cookie(pool, sibling))
        pool.run(_retitle_tabs(pool, sibling))
        with pytest.raises(api.BrowserError, match="restart=True"):
            api.browser("mode", mode="headed", session=session)
        # By id alone: a tab's title is the page's to change, and Chrome reports a
        # fresh tab's as "" until its first navigation commits.
        assert _tab_ids(api.browser("tabs", session=sibling)) == _tab_ids(original)
        for mode in ("headed", "headless"):
            switched = api.browser(
                "mode",
                mode=mode,
                restart=True,
                session=session,
            )
            assert switched.mode == mode
            assert switched.restarted
            assert api.browser("mode", session=sibling).mode == mode
            assert api.browser("tabs", session=sibling).tabs
            chrome = pool.run(
                pool.browser(sibling.profile_dir, headless=sibling.headless),
            )
            if mode == "headed":
                assert pool.run(chrome.tabs[0].evaluate("navigator.webdriver")) is False
            unchanged = api.browser("mode", mode=mode, session=sibling)
            assert unchanged.mode == mode
            assert not unchanged.restarted
            assert pool.run(_read_mode_cookie(pool, sibling)) == "retained"


def _tab_ids(result: api.BrowserResult) -> list[str]:
    return [tab.tab_id for tab in result.tabs]


async def _retitle_tabs(pool: zendriver.BrowserPool, session: FetchSession) -> None:
    chrome = await pool.browser(session.profile_dir, headless=session.headless)
    await chrome.update_targets()
    for tab in chrome.tabs:
        await tab.evaluate("document.title = 'retitled'")


async def _seed_mode_cookie(pool: zendriver.BrowserPool, session: FetchSession) -> None:
    chrome = await pool.browser(session.profile_dir, headless=session.headless)
    assert chrome.connection is not None
    await chrome.connection.send(
        cdp.storage.set_cookies(
            [
                cdp.network.CookieParam(
                    name="mode_probe",
                    value="retained",
                    url="https://example.test/",
                    expires=cdp.network.TimeSinceEpoch(time.time() + 3600),
                ),
            ],
        ),
    )


async def _read_mode_cookie(pool: zendriver.BrowserPool, session: FetchSession) -> str:
    chrome = await pool.browser(session.profile_dir, headless=session.headless)
    assert chrome.connection is not None
    cookies = await chrome.connection.send(cdp.storage.get_cookies())
    return next(cookie.value for cookie in cookies if cookie.name == "mode_probe")


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
