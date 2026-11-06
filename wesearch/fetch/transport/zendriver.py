"""Fetch in a real headless Chrome, for pages behind JavaScript or a bot challenge.

One Chrome per profile serves every process on the host; each fetch runs in its
own tab, and the profile keeps Chrome's cookies.

Run ``webrowser URL`` to open URL in a visible Chrome on the same profile, to
see what a fetch saw, clear a block, or log in.
"""

# ruff: noqa: T201 -- Browser command lifecycle output.

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Coroutine, Generator, Mapping
from html import unescape
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Protocol, TypeVar, cast
from urllib.parse import urlparse, urlsplit, urlunsplit

import argparse
import asyncio
import contextlib
import fcntl
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
import warnings

from treekle import from_plain

from wesearch.chrome.capture import chrome_binary
from wesearch.fetch.challenge import (
    classify_challenge,
    classify_http_error,
    is_shell,
    redirects_soon,
)
from wesearch.fetch.common import (
    ORIGIN_BOUND,
    cookie_header,
    origin,
    pinned_host,
)
from wesearch.fetch.gate import challenged_urls, clear_host
from wesearch.fetch.session import default_session
from wesearch.fetch.types import FetchSession
from wesearch.lib.userdirs import cache_dir
from wesearch.types.errors import FetchError
from wesearch.types.params import ignore, no_cookies


if TYPE_CHECKING:
    from concurrent.futures import Future

    from zendriver.core import util

    import zendriver

    from wesearch.types.params import HttpMethod
else:
    from wrapt import lazy_import

    zendriver = lazy_import("zendriver")  # 214ms.
    util = lazy_import("zendriver.core.util")


__all__ = [
    "BrowserPool",
    "BrowserUnavailableError",
    "fetch_zendriver",
    "get_browser_pool",
    "guard_requests",
    "open_window",
]

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Every Chrome this process launched, for :meth:`BrowserPool.shutdown` in tests.
_launched: list[subprocess.Popen[bytes]] = []


class BrowserUnavailableError(OSError):
    """Chrome could not be launched or connected to on this host."""


def fetch_zendriver(
    url: str,
    *,
    method: HttpMethod = "GET",
    headers: Mapping[str, str | None] = MappingProxyType({}),
    body: bytes = b"",
    timeout_sec: float = 30.0,
    connect_timeout_sec: float = 10.0,
    max_redirects: int = 10,
    on_redirect: Callable[[str], None] = ignore,
    on_response: Callable[[int, dict[str, str], str], None] = ignore,
    allow_private_hosts: bool = False,
    session: FetchSession | None = None,
    cookies_for_host: Callable[[str], Mapping[str, str]] = no_cookies,
) -> bytes:
    """Load ``url`` in a real headless Chrome; return the rendered page.

    For pages that need JavaScript or a bot challenge solved. Slower than
    ``fetch_curl``: the first call launches Chrome, which later calls and
    processes reuse.

    Args:
      url: Fully-qualified URL.
      method: Must be ``GET``.
      headers: Sent over Chrome's own for the whole tab, so a ``User-Agent``
        replaces Chrome's on every subresource; a ``None`` value is ignored. A
        ``Cookie`` header's cookies are set in Chrome for ``url``.
      body: Must be empty.
      timeout_sec: Seconds allowed for the whole load.
      connect_timeout_sec: Ignored; Chrome connects on its own schedule.
      max_redirects: Redirects Chrome may follow; past it the load fails.
      on_redirect: Called with each redirect target; raise to stop.
      on_response: Called with ``(status, headers, url)`` for each page
        response; the last carries Chrome's cookies for where the page landed.
      allow_private_hosts: Allow loopback, private, and link-local addresses;
        leave ``False`` for any URL you did not write.
      session: The browser to reuse, and its Chrome profile; ``None`` uses the
        process's shared session.
      cookies_for_host: Ignored; Chrome sends its own cookies on a redirect.

    Returns:
      body: The rendered page, or the payload of a non-HTML response.

    Raises:
      ValueError: On a non-GET method or a body.
      FetchError: On a page status of 400 or more, or a failed load (status 0).
      BrowserUnavailableError: When Chrome cannot be launched or reached.

    """
    del connect_timeout_sec, cookies_for_host
    if method != "GET" or body:
        raise ValueError("The zendriver transport supports only a bodyless GET.")
    pinned_host(url, allow_private_hosts=allow_private_hosts)
    session = default_session() if session is None else session
    sent = {name: value for name, value in headers.items() if value is not None}
    cookie = cookie_header(sent, {})
    pairs = (pair.strip().partition("=") for pair in cookie.split(";"))
    pool = get_browser_pool(session)
    try:
        page = pool.run(
            _load(
                pool,
                url,
                profile_dir=session.profile_dir,
                headless=session.headless,
                headers={k: v for k, v in sent.items() if k.lower() != "cookie"},
                cookies={name: value for name, eq, value in pairs if eq and name},
                timeout_sec=timeout_sec,
                max_redirects=max_redirects,
                on_redirect=on_redirect,
                on_response=on_response,
                allow_private_hosts=allow_private_hosts,
            ),
            # Past ``_load``'s own timeout, so that one fires and closes the tab.
            timeout_sec=timeout_sec + 30,
        )
    except TimeoutError as error:
        raise FetchError(url, 0, {}, b"timed out") from error
    return page


def get_browser_pool(session: FetchSession) -> BrowserPool:
    """Return the session's browser pool, creating it once under its lock."""
    with session.lock:
        pool = session.browsers
        if not isinstance(pool, BrowserPool):
            pool = session.browsers = BrowserPool()
        return pool


@contextlib.asynccontextmanager
async def guard_requests(
    tab: zendriver.Tab,
    *,
    deadline: float,
    allow_private_hosts: bool = False,
    headers: Mapping[str, str] = MappingProxyType({}),
    initial_url: str = "",
    max_redirects: int = 10,
    on_redirect: Callable[[str], None] = ignore,
) -> AsyncGenerator[_RequestGuard]:
    """Intercept requests for one operation without taking ownership of its tab.

    Policy refusals abort subresources locally; document refusals fail the operation.

    Args:
      tab: Tab whose actions the caller serializes.
      deadline: Absolute event-loop deadline for request replies.
      allow_private_hosts: Permit private addresses through the shared host policy.
      headers: Credentials to inject only on the initial URL's origin.
      initial_url: Original navigation URL, never inferred from event order.
      max_redirects: Maximum document hops away from the current URL.
      on_redirect: Callback before following each document hop; raising aborts it.

    Yields:
      guard: Await ``drain()`` before trusting an action's successful output.

    Raises:
      ValueError: A request violates the shared host policy.
      TimeoutError: Request replies or bounded cleanup exceed their deadline.

    """
    guard = _RequestGuard(
        tab,
        deadline=deadline,
        allow_private_hosts=allow_private_hosts,
        headers=headers,
        initial_url=initial_url,
        max_redirects=max_redirects,
        on_redirect=on_redirect,
    )
    register = cast(
        Callable[[type[object], Callable[..., object]], None],
        tab.add_handler,
    )
    remove = cast(
        Callable[[type[object], Callable[..., object]], None],
        tab.remove_handlers,
    )
    register(zendriver.cdp.fetch.RequestPaused, guard.on_paused)
    try:
        async with asyncio.timeout_at(deadline):
            await tab.send(
                zendriver.cdp.fetch.enable(
                    patterns=[
                        zendriver.cdp.fetch.RequestPattern(
                            url_pattern="*",
                            request_stage=zendriver.cdp.fetch.RequestStage.REQUEST,
                        ),
                    ],
                ),
            )
        yield guard
        await guard.drain()
    finally:
        remove(zendriver.cdp.fetch.RequestPaused, guard.on_paused)
        await guard.finish()


def open_window(url: str, *, profile_dir: Path) -> None:
    """Open ``url`` in a visible Chrome on ``profile_dir``; block until it closes.

    Replaces the profile's headless Chrome; fetches meanwhile run in this
    window. Each site a bot challenge locked opens in its own tab, and each site
    you load is unlocked. Cookies you set (a login) persist for later fetches.

    Args:
      url: The page to open.
      profile_dir: Chrome profile, e.g. :attr:`FetchSession.profile_dir`.

    """
    pool = BrowserPool()
    try:
        pool.run(pool.mode(profile_dir, headless=False, restart=True))
        browser = pool.run(pool.browser(profile_dir, headless=False))
        try:
            visited = pool.run(_watch_window(url, browser))
        except BaseException:
            try:
                port = browser.config.port
                assert isinstance(port, int)
                pool.run(_close_chrome(port), timeout_sec=30)
            except Exception:
                logger.warning("browser stop failed during interruption", exc_info=True)
            raise
        # Again at close: an agent may have relocked a site meanwhile.
        for host in visited:
            clear_host(host)
    finally:
        pool.shutdown()


class BrowserPool:
    """Connections to shared Chromes, one per profile, on one event-loop thread.

    Shutting down disconnects; Chrome stays up for other processes.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._browsers: dict[str, zendriver.Browser] = {}
        self._connect_lock = asyncio.Lock()
        self.targets_lock = asyncio.Lock()
        self._tab_locks: dict[str, asyncio.Lock] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="loop-web-browser",
            daemon=True,
        )
        self._thread.start()

    def tab_lock(self, tab_id: str) -> asyncio.Lock:
        """Return the lock serializing actions on ``tab_id`` on the pool's loop."""
        return self._tab_locks.setdefault(tab_id, asyncio.Lock())

    def run(self, coro: Coroutine[object, object, _T], *, timeout_sec: float = 0) -> _T:
        """Run ``coro`` on the pool's loop; return its result.

        Args:
          coro: The coroutine.
          timeout_sec: Longest wait; ``0`` waits forever. A coroutine with its
            own timeout still needs one: a stopped loop never runs it.

        Returns:
          result: What ``coro`` returned.

        Raises:
          TimeoutError: If the wait exceeds ``timeout_sec``.

        """
        future: Future[_T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout_sec or None)
        except BaseException:
            future.cancel()
            raise

    async def browser(self, profile_dir: Path, *, headless: bool) -> zendriver.Browser:
        """Return a live connection to the Chrome on ``profile_dir``, launching one.

        Args:
          profile_dir: Chrome profile.
          headless: Launch mode, if no Chrome runs on the profile yet.

        Returns:
          browser: The connection.

        Raises:
          BrowserUnavailableError: When Chrome cannot be launched or attached.

        """
        key = str(profile_dir.resolve())  # noqa: ASYNC240 -- The browser driver owns the blocking operation on its dedicated worker.
        async with self._connect_lock:
            with self._lock:
                existing = self._browsers.get(key)
            # Not ``Browser.stopped``: it reads a process handle an attached
            # connection lacks.
            live = (
                existing is not None
                and (connection := existing.connection) is not None
                and not connection.closed
                and connection.listener is not None
                and connection.listener.running
            )
            if existing is not None and live:
                return existing
            port = await asyncio.to_thread(
                _shared_chrome,
                profile_dir,
                headless=headless,
            )
            _tolerate_late_cdp_replies()
            try:
                connected = await zendriver.start(
                    host="127.0.0.1",
                    port=port,
                    browser_connection_timeout=0.5,
                    browser_connection_max_tries=6,
                )
            except Exception as error:
                raise BrowserUnavailableError(
                    f"Could not attach to Chrome on DevTools port {port}: {error}",
                ) from error
            with self._lock:
                self._browsers[key] = connected
            return connected

    async def mode(
        self,
        profile_dir: Path,
        *,
        headless: bool | None,
        restart: bool,
    ) -> tuple[bool, bool]:
        """Report or change Chrome's actual mode with explicit restart consent.

        A mode change interrupts shared tabs. Profile data persists, but page
        state and target ids do not; Chrome restores its last session best-effort.
        Cancellation waits for the transition to release the shared launch lock.

        Args:
          profile_dir: Shared Chrome profile.
          headless: Desired mode; None only reports the running mode.
          restart: Consent to interrupt Chrome when its mode differs.

        Returns:
          headless: Actual Chrome mode.
          restarted: Whether an existing Chrome was replaced.

        """
        async with self._connect_lock:
            transition = asyncio.create_task(
                asyncio.to_thread(
                    _switch_mode,
                    self,
                    profile_dir,
                    headless=headless,
                    restart=restart,
                ),
            )
            try:
                return await asyncio.shield(transition)
            except asyncio.CancelledError:
                await asyncio.shield(transition)
                raise

    def shutdown(self, *, budget_sec: float = 5.0, stop_launched: bool = False) -> None:
        """Disconnect from every Chrome and stop the loop thread.

        Args:
          budget_sec: Seconds allowed for all disconnects together.
          stop_launched: Also stop every Chrome this process launched, though
            other processes may use it; for test teardown.

        """
        if self._loop.is_closed():
            return
        with self._lock:
            browsers = list(self._browsers.values())
            self._browsers.clear()
        deadline = time.monotonic() + budget_sec
        for browser in browsers:
            remaining = deadline - time.monotonic()
            connection = browser.connection
            if remaining <= 0 or connection is None:
                continue
            # Not ``Browser.stop``: it closes Chrome for every attached process.
            try:
                self.run(connection.aclose(), timeout_sec=remaining)
            except Exception:
                logger.warning(
                    "browser disconnect failed during shutdown",
                    exc_info=True,
                )
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        # Zendriver's asyncio_atexit makes ``close`` run the loop, which fails on
        # a thread with a loop of its own running, as a caller's often has.
        closer = threading.Thread(target=self._loop.close, name="loop-web-close")
        closer.start()
        closer.join()
        self._tab_locks.clear()
        while stop_launched and _launched:
            process = _launched.pop()
            with contextlib.suppress(OSError):
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    # Under ``-W error`` zendriver's own Deprecation/ResourceWarnings raise inside
    # its CDP handlers and hang the fetch. This thread runs only zendriver.
    def _run_loop(self) -> None:
        """Run the pool's loop until :meth:`shutdown`, muting zendriver's warnings."""
        asyncio.set_event_loop(self._loop)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                module=r"zendriver\..*",
            )
            warnings.filterwarnings("ignore", category=ResourceWarning)
            self._loop.run_forever()


def main() -> int:
    """Open a URL in a visible Chrome on the fetch profile.

    Returns:
      code: The process exit code, ``0``.

    """
    parser = argparse.ArgumentParser(
        prog="webrowser",
        description=(
            "Open a URL in a visible Chrome on the profile the zendriver "
            "transport fetches with, to see what a fetch saw or to log in. "
            "Cookies you set persist for later fetches. Close the window when "
            "done."
        ),
        # Placeholders: a real site here would be a recipe for getting past it.
        epilog=(
            "Examples:\n"
            "  webrowser https://the-site-that-failed.example/\n"
            "  webrowser https://login.example/  # seat a session cookie\n"
            "  webrowser # opens blank; navigate by hand"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    profile_dir = FetchSession().profile_dir
    print(
        f"Opening {flags.url} in Chrome on {profile_dir} -- close the window when done.",
    )
    open_window(flags.url, profile_dir=profile_dir)
    print("Window closed.")
    return 0


# ``shell_wait_sec`` bounds the wait for a script to fill a page that loaded
# empty, which is re-read every ``shell_poll_sec``.
async def _load(
    pool: BrowserPool,
    url: str,
    *,
    profile_dir: Path,
    headless: bool,
    headers: dict[str, str],
    cookies: dict[str, str],
    timeout_sec: float,
    max_redirects: int,
    on_redirect: Callable[[str], None],
    on_response: Callable[[int, dict[str, str], str], None],
    allow_private_hosts: bool,
    shell_wait_sec: float = 5.0,
    shell_poll_sec: float = 0.25,
) -> bytes:
    """Load ``url`` in a new tab, closed after; report and return its page."""
    loop = asyncio.get_running_loop()
    pages: list[tuple[int, dict[str, str], str]] = []
    navigated = asyncio.Event()
    request_deadline = loop.time() + timeout_sec

    def on_document(event: object, *_unused: object) -> None:
        if (
            isinstance(event, zendriver.cdp.network.ResponseReceived)
            and event.type_ == zendriver.cdp.network.ResourceType.DOCUMENT
            and event.frame_id == tab.target_id
        ):
            response = event.response
            pages.append(
                (
                    response.status,
                    {
                        k.lower(): v
                        for k, v in from_plain(response.headers, dict[str, str]).items()
                    },
                    response.url,
                ),
            )

    def on_navigated(event: object, *_unused: object) -> None:
        if (
            isinstance(event, zendriver.cdp.page.FrameNavigated)
            and event.frame.parent_id is None
        ):
            # Off-loop: a plain ``set()`` would not wake the waiter.
            loop.call_soon_threadsafe(navigated.set)

    try:
        async with asyncio.timeout(timeout_sec):
            browser = await pool.browser(profile_dir, headless=headless)
    except TimeoutError as error:
        # Not a timed-out page: callers fall back only from an absent browser, and
        # the launch goes on for the next call.
        raise BrowserUnavailableError(
            f"Chrome was still starting after {timeout_sec:g}s.",
        ) from error
    async with asyncio.timeout_at(request_deadline):
        connection = browser.connection
        if connection is None:
            raise BrowserUnavailableError("The browser has no DevTools connection.")
        # CookieJar borrows an arbitrary page, which another fetch can close.
        if cookies:
            await connection.send(
                zendriver.cdp.storage.set_cookies(
                    [
                        zendriver.cdp.network.CookieParam(
                            name=name,
                            value=value,
                            url=url,
                        )
                        for name, value in cookies.items()
                    ],
                ),
            )
        # The marker URL tells ``open_window`` this tab is an agent's, not the user's.
        tab = await browser.get("about:blank#wesearch-agent", new_tab=True)
        async with contextlib.AsyncExitStack() as stack:
            stack.push_async_callback(_close_tab, tab)
            guard = await stack.enter_async_context(
                guard_requests(
                    tab,
                    deadline=request_deadline,
                    allow_private_hosts=allow_private_hosts,
                    headers=headers,
                    initial_url=url,
                    max_redirects=max_redirects,
                    on_redirect=on_redirect,
                ),
            )
            # `zendriver` types ``add_handler`` with a bare Callable, and a stub for
            # the inherited method would blank the rest of ``Connection``.
            register = cast(
                Callable[[type[object], Callable[..., None]], None],
                tab.add_handler,
            )
            register(zendriver.cdp.network.ResponseReceived, on_document)
            register(zendriver.cdp.page.FrameNavigated, on_navigated)
            # Tab-wide headers reach cross-origin resources; credentials must not.
            ambient = {
                k: v for k, v in headers.items() if k.lower() not in ORIGIN_BOUND
            }
            await tab.send(zendriver.cdp.network.enable())
            if ambient:
                await tab.send(
                    zendriver.cdp.network.set_extra_http_headers(
                        zendriver.cdp.network.Headers(ambient),
                    ),
                )
            await tab.get(url)
            await tab.wait_for_ready_state("complete")
            # A challenge's script, and a meta refresh (Google's enablejs page,
            # PyTorch's docs), load the real page after their own load completes;
            # mid-parse that page is an empty <head>, so wait for its ready state
            # too. Half the budget, so a page still walled returns its wall, not a
            # timeout. ``on_success_body``: a rendered page's CAPTCHA widget (a
            # login form's) is not a wall.
            deadline = loop.time() + timeout_sec / 2
            shell_deadline = loop.time() + shell_wait_sec
            while True:
                # Before the read: a navigation between read and wait must count.
                navigated.clear()
                page = await tab.get_content()
                payload = _unwrap_viewer(page)
                remaining = deadline - loop.time()
                walled = classify_challenge(
                    payload,
                    on_success_body=True,
                ) is not None or redirects_soon(page)
                # Ready state "complete" can come before an app's scripts fill the
                # page in place, with no navigation to wait on.
                filling = (
                    not walled
                    and is_shell(payload.encode())
                    and loop.time() < shell_deadline
                )
                if remaining <= 0 or not (walled or filling):
                    break
                if filling:
                    await asyncio.sleep(shell_poll_sec)
                    continue
                try:
                    await asyncio.wait_for(navigated.wait(), timeout=remaining)
                    await asyncio.wait_for(
                        tab.wait_for_ready_state("complete"),
                        timeout=max(deadline - loop.time(), 0.0),
                    )
                except TimeoutError:
                    page = await tab.get_content()
                    break
            landed = cast(str, await tab.evaluate("document.location.href")) or url
            host = urlparse(landed).hostname or ""
            jar = {
                cookie.name: cookie.value or ""
                for cookie in await connection.send(zendriver.cdp.storage.get_cookies())
                if (domain := cookie.domain.removeprefix("."))
                and (host == domain or host.endswith(f".{domain}"))
            }
            await guard.drain()
    status, received, _ = pages[-1] if pages else (200, {}, landed)
    if jar:
        received = {
            **received,
            "set-cookie": "\n".join(f"{k}={v}" for k, v in jar.items()),
        }
    for hop_status, hop_headers, hop_url in pages[:-1]:
        on_response(hop_status, hop_headers, hop_url)
    on_response(status, received, landed)
    content = _unwrap_viewer(page).encode()
    if status >= 400:
        raise classify_http_error(landed, status, received, content)
    return content


async def _close_tab(tab: zendriver.Tab) -> None:
    """Close a fetch-owned tab without replacing the load's result or error."""
    closing = asyncio.ensure_future(tab.close())
    try:
        async with asyncio.timeout(5.0):
            await asyncio.shield(closing)
    except TimeoutError:
        closing.cancel()
        await asyncio.gather(asyncio.shield(closing), return_exceptions=True)
        logger.debug("tab close timed out; abandoning the tab")
    except Exception:
        logger.debug("tab close failed; abandoning the tab", exc_info=True)


def _with_credentials(
    event: zendriver.cdp.fetch.RequestPaused,
    credentials: dict[str, str],
) -> object:
    """Return the command continuing ``event``'s request with ``credentials``."""
    replaced = {name.lower() for name in credentials}
    kept = {
        k: v
        for k, v in from_plain(event.request.headers, dict[str, str]).items()
        if k.lower() not in replaced
    }
    return zendriver.cdp.fetch.continue_request(
        event.request_id,
        headers=[
            zendriver.cdp.fetch.HeaderEntry(name=k, value=v)
            for k, v in (kept | credentials).items()
        ],
    )


def _unwrap_viewer(
    page: str,
    *,
    # Chrome's JSON formatter mounts empty divs after the </pre>.
    viewer: re.Pattern[str] = re.compile(
        r"^<html[^>]*>\s*<head>.*?color-scheme.*?</head>\s*"
        r"<body[^>]*>\s*<pre[^>]*>(?P<payload>.*)</pre>"
        r"(?:\s*<div[^>]*></div>)*\s*</body>\s*</html>$",
        re.DOTALL | re.IGNORECASE,
    ),
) -> str:
    """Return the payload Chrome wraps a non-HTML response in, else ``page``."""
    match = viewer.match(page.strip())
    return unescape(match["payload"]) if match else page


def _wire_url(url: str) -> str:
    """Return ``url`` as it goes on the wire: no fragment, path never empty."""
    parts = urlsplit(url)
    return urlunsplit(parts._replace(fragment="", path=parts.path or "/"))


@contextlib.contextmanager
def _launch_lock(profile_dir: Path) -> Generator[None]:
    """Hold the host-wide lock serializing Chrome launches on ``profile_dir``."""
    profile_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = profile_dir.parent / f".{profile_dir.name}.launch.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _devtools_port(
    profile_dir: Path,
    *,
    proc_root: Path = Path("/proc"),
    platform: str = sys.platform,
) -> int | None:
    """Return the DevTools port of the Chrome owning ``profile_dir``, if any."""
    lock_dir = profile_dir
    if (
        platform.startswith("linux")
        and not (profile_dir / "SingletonLock").is_symlink()
    ):
        lock_dir = _namespace_profile(profile_dir, proc_root=proc_root)
    try:
        owner = (lock_dir / "SingletonLock").readlink()
        pid = int(str(owner).rsplit("-", 1)[1])
        command_file = proc_root / str(pid) / "cmdline"
        if command_file.is_file() or platform != "darwin":
            command = command_file.read_bytes().replace(b"\0", b" ").decode()
        else:
            ps = subprocess.run(  # noqa: S603 -- PID is parsed as an integer.
                ["/bin/ps", "-p", str(pid), "-o", "command="],
                capture_output=True,
                text=True,
                check=False,
            )
            if ps.returncode:
                return None
            command = ps.stdout
    except (IndexError, OSError, UnicodeError, ValueError):
        return None

    def flag(marker: str) -> str:
        return command.partition(marker)[2].partition(" --")[0].strip()

    owner_profile = flag("--user-data-dir=")
    if not owner_profile or Path(owner_profile).resolve() != profile_dir.resolve():
        return None
    try:
        port = int(flag("--remote-debugging-port=").split()[0])
        return port or int((lock_dir / "DevToolsActivePort").read_text().split()[0])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def _namespace_profile(profile_dir: Path, *, proc_root: Path) -> Path:
    """Locate a profile hidden by its owner's private mount namespace."""
    # Snap's private /tmp hides SingletonLock from the host, causing duplicate launches.
    relative = profile_dir.resolve().relative_to("/")
    for process in proc_root.glob("[0-9]*"):
        candidate = process / "root" / relative
        try:
            owner = (candidate / "SingletonLock").readlink()
        except OSError:
            continue
        if str(owner).rsplit("-", 1)[-1] == process.name:
            return candidate
    return profile_dir


def _port_answers(port: int) -> bool:
    """Whether something accepts connections on the DevTools ``port``."""
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
    except OSError:
        return False
    return True


async def _close_chrome(port: int) -> None:
    """Stop the Chrome on DevTools ``port`` and wait until it is gone."""
    browser = await zendriver.start(host="127.0.0.1", port=port)
    await browser.stop()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not await asyncio.to_thread(_port_answers, port):
            return
        await asyncio.sleep(0.05)
    raise BrowserUnavailableError(f"Chrome on DevTools port {port} did not close.")


# A reply arriving after its request timed out raises InvalidStateError in zendriver's
# listener, which kills it, and the pooled browser then ignores every later fetch.
def _tolerate_late_cdp_replies() -> None:
    """Make zendriver drop a CDP reply whose transaction already finished."""
    # Through the module: patching via the lazy-import proxy misses the class.
    transaction = cast(
        "type[_Transaction]",
        vars(sys.modules[zendriver.Connection.__module__])["Transaction"],
    )
    vendor = transaction.__call__
    if vendor.__module__ == __name__:
        return

    def call(self: _Transaction, **response: dict[str, object]) -> None:
        if not self.done():
            vendor(self, **response)

    transaction.__call__ = call


class _Transaction(Protocol):
    __call__: ClassVar[Callable[..., None]]

    def done(self) -> bool: ...


# On macOS a headless Chrome from the installed bundle receives, and drops, every link
# the user opens. Only another bundle id avoids it, and re-signing a copy gets it
# killed, so use a Chrome for Testing that Puppeteer or Playwright downloaded.
def _chrome_for_testing(*, platform: str = sys.platform) -> str:
    """Return a Chrome that will not capture the user's link clicks, or ``""``."""
    if platform != "darwin":
        return ""
    roots = (
        # Puppeteer uses ~/.cache on every platform, macOS included.
        Path.home() / ".cache" / "puppeteer" / "chrome",  # noqa: TID251 -- vendor fixed path, not ours (AGENTS.md rule 3)  # house-ignore[xdg-literal] -- Vendor CLI's fixed home path, not ours (AGENTS.md rule 3).
        cache_dir() / "ms-playwright",
    )
    for root in roots:
        if not root.is_dir():
            continue
        # Newest first, by number: as text, version 99 ranks above 100.
        for build in sorted(
            root.iterdir(),
            key=lambda b: ([int(m[0]) for m in re.finditer(r"\d+", b.name)], b.name),
            reverse=True,
        ):
            # Both arches: Rosetta runs x64 on Apple silicon.
            for arch in ("chrome-mac-arm64", "chrome-mac-x64"):
                for app, stem in (
                    ("Google Chrome for Testing.app", "Google Chrome for Testing"),
                    ("Chromium.app", "Chromium"),
                ):
                    candidate = build / arch / app / "Contents" / "MacOS" / stem
                    if candidate.is_file():
                        return str(candidate)
    logger.debug("no Chrome for Testing found; falling back to zendriver's Chrome")
    return ""


async def _watch_window(url: str, browser: zendriver.Browser) -> set[str]:
    """Show ``url`` and each locked site until closed; return the hosts loaded."""
    connection = browser.connection
    if connection is None:
        raise BrowserUnavailableError(
            "The visible Chrome has no DevTools connection.",
        )
    # Agent fetches run in this window too; a blocked one must not unlock its
    # own block. Agent tabs open at ``about:blank#wesearch-agent``, and tabs
    # they open are theirs.
    agent_tabs: set[str] = set()
    current: dict[str, str] = {}
    visited: set[str] = set()

    async def on_target(event: object, *_unused: object) -> None:
        if not isinstance(
            event,
            zendriver.cdp.target.TargetCreated | zendriver.cdp.target.TargetInfoChanged,
        ):
            return
        info = event.target_info
        if info.type_ != "page" or info.target_id in agent_tabs:
            return
        if info.url == "about:blank#wesearch-agent" or info.opener_id in agent_tabs:
            agent_tabs.add(info.target_id)
            return
        # Re-fired as the title changes; only a new URL is a visit.
        if current.get(info.target_id) == info.url:
            return
        current[info.target_id] = info.url
        parsed = urlparse(info.url)
        if parsed.scheme in ("http", "https") and parsed.hostname is not None:
            visited.add(parsed.hostname)
            await asyncio.to_thread(clear_host, parsed.hostname)

    register = cast(
        Callable[[type[object], Callable[..., object]], None],
        connection.add_handler,
    )
    register(zendriver.cdp.target.TargetCreated, on_target)
    register(zendriver.cdp.target.TargetInfoChanged, on_target)
    listener = connection.listener
    if (
        connection.closed
        or listener is None
        or listener.task is None
        or listener.task.done()
    ):
        return visited
    opening = asyncio.create_task(_open_window_pages(url, browser))
    try:
        # Tab.get can remain in its idle wait after Chrome has closed.
        await asyncio.wait(
            {opening, listener.task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if opening.done():
            opening.result()
        await asyncio.shield(listener.task)
    finally:
        opening.cancel()
        await asyncio.gather(opening, return_exceptions=True)
    return visited


async def _open_window_pages(url: str, browser: zendriver.Browser) -> None:
    for page in (url, *(locked for locked in challenged_urls() if locked != url)):
        try:
            tab = await browser.get("about:blank", new_tab=True)
            await tab.get(page)
        except Exception:
            logger.warning("could not open %s", page, exc_info=True)


def _switch_mode(
    pool: BrowserPool,
    profile_dir: Path,
    *,
    headless: bool | None,
    restart: bool,
) -> tuple[bool, bool]:
    with _launch_lock(profile_dir):
        port = _devtools_port(profile_dir)
        if port is None or not _port_answers(port):
            desired = True if headless is None else headless
            _start_shared_chrome(profile_dir, headless=desired)
            return desired, False
        # A cold Chrome accepts connections up to 6s before it first answers one.
        actual = pool.run(_read_mode(port), timeout_sec=15)
        if headless is None or actual == headless:
            return actual, False
        if not restart:
            raise BrowserUnavailableError(
                "Changing shared Chrome mode requires restart=True; tabs will be interrupted.",
            )
        pool.run(_close_chrome(port), timeout_sec=15)
        _start_shared_chrome(profile_dir, headless=headless, restore_session=True)
        return headless, True


async def _read_mode(port: int) -> bool:
    browser = await zendriver.start(
        host="127.0.0.1",
        port=port,
        browser_connection_timeout=0.5,
        browser_connection_max_tries=6,
    )
    connection = browser.connection
    if connection is None:
        raise BrowserUnavailableError("The browser has no DevTools connection.")
    try:
        _, _, _, user_agent, _ = await connection.send(
            zendriver.cdp.browser.get_version(),
        )
        return "HeadlessChrome/" in user_agent
    finally:
        await connection.aclose()


def _shared_chrome(profile_dir: Path, *, headless: bool) -> int:
    """Return the DevTools port of the Chrome on ``profile_dir``, launching one."""
    with _launch_lock(profile_dir):
        port = _devtools_port(profile_dir)
        if port is not None and _port_answers(port):
            return port
        return _start_shared_chrome(profile_dir, headless=headless)


def _start_shared_chrome(
    profile_dir: Path,
    *,
    headless: bool,
    restore_session: bool = False,
) -> int:
    """Launch Chrome while the caller holds the profile launch lock."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "DevToolsActivePort").unlink(missing_ok=True)
    executable = _chrome_for_testing()
    port = util.free_port()
    config = zendriver.Config(
        port=port,
        headless=headless,
        user_data_dir=str(profile_dir),
        # Chrome's sandbox refuses to run as root.
        sandbox=os.geteuid() != 0,
        browser_executable_path=executable or chrome_binary() or "",
        browser_args=["--use-mock-keychain"] if executable else [],
        host="127.0.0.1",
    )
    try:
        # Port zero exposes navigator.webdriver even in headed Chrome.
        process = subprocess.Popen(  # noqa: S603 -- argv is zendriver's own launch config.
            [
                str(config.browser_executable_path),
                *config(),
                *(["--restore-last-session"] if restore_session else ["about:blank"]),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as error:
        raise BrowserUnavailableError(f"Could not launch Chrome: {error}") from error
    _launched.append(process)
    # Paging Chrome in from a cold disk took fresh CI runners up to 24s, against
    # 0.2s warm, and a kill makes the next launch pay most of that again. Callers
    # stop waiting at their own deadlines; this bound only frees the launch lock
    # from a Chrome that will never answer.
    budget_sec = 60
    deadline = time.monotonic() + budget_sec
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise BrowserUnavailableError(
                f"Could not launch Chrome: it exited with code {process.returncode}.",
            )
        if _port_answers(port):
            return port
        time.sleep(0.05)
    process.kill()
    process.wait()
    raise BrowserUnavailableError(
        f"Could not launch Chrome: DevTools did not come up in {budget_sec}s.",
    )


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "url",
        nargs="?",
        default="about:blank",
        help="The URL to open; omit for a blank page.",
    )


class _Flags(Protocol):
    url: str


class _RequestGuard:
    """Track intercepted replies and surface their first failure to the caller."""

    def __init__(
        self,
        tab: zendriver.Tab,
        *,
        deadline: float,
        allow_private_hosts: bool,
        headers: Mapping[str, str],
        initial_url: str,
        max_redirects: int,
        on_redirect: Callable[[str], None],
    ) -> None:
        self._tab = tab
        self._deadline = deadline
        self._allow_private_hosts = allow_private_hosts
        self._credentials = {
            k: v for k, v in headers.items() if k.lower() in ORIGIN_BOUND
        }
        self._initial_url = initial_url
        self._position = initial_url
        self._followed = 0
        self._max_redirects = max_redirects
        self._on_redirect = on_redirect
        self._tasks: set[asyncio.Task[object]] = set()
        self._unchecked: set[zendriver.cdp.fetch.RequestId] = set()
        self._errors: list[Exception] = []
        self._finished = False

    async def drain(self) -> None:
        """Wait for pending request replies and raise their first failure.

        Raises:
          Exception: The first blocked-host or reply error recorded by the guard.
          TimeoutError: Pending replies exceeded the operation's deadline.

        """
        async with asyncio.timeout_at(self._deadline):
            await asyncio.sleep(0)
            while self._tasks:
                await asyncio.gather(*self._tasks)
                await asyncio.sleep(0)
        if self._errors:
            raise self._errors[0]

    async def on_paused(self, event: object, *_unused: object) -> None:
        """Answer an intercepted request and retain errors for ``drain``.

        Args:
          event: CDP event supplied by zendriver's listener.
          *_unused: Optional connection supplied by the listener.

        """
        if self._finished or not isinstance(event, zendriver.cdp.fetch.RequestPaused):
            return
        task = asyncio.current_task()
        assert isinstance(task, asyncio.Task)
        self._tasks.add(task)
        self._unchecked.add(event.request_id)
        try:
            command = await self._command(event)
            self._unchecked.remove(event.request_id)
            await self._tab.send(
                cast(
                    Generator[dict[str, object], dict[str, object], None],
                    command,
                ),
            )
        except Exception as error:  # noqa: BLE001 -- Retrieved and raised by drain, never discarded.
            self._errors.append(error)
        finally:
            self._tasks.discard(task)

    async def _command(
        self,
        event: zendriver.cdp.fetch.RequestPaused,
    ) -> object:
        target = event.request.url
        # Browser-local subresources use no network host; documents remain HTTP(S).
        if (
            event.resource_type != zendriver.cdp.network.ResourceType.DOCUMENT
            and urlsplit(target).scheme in ("chrome-extension", "data", "blob", "about")
        ):
            return zendriver.cdp.fetch.continue_request(event.request_id)
        refuse = zendriver.cdp.fetch.fail_request(
            event.request_id,
            zendriver.cdp.network.ErrorReason.ACCESS_DENIED,
        )
        try:
            await asyncio.to_thread(
                pinned_host,
                target,
                allow_private_hosts=self._allow_private_hosts,
            )
        except ValueError as error:
            if event.resource_type == zendriver.cdp.network.ResourceType.DOCUMENT:
                self._errors.append(error)
            return refuse
        if event.resource_type == zendriver.cdp.network.ResourceType.DOCUMENT:
            is_hop = bool(self._position) and _wire_url(target) != _wire_url(
                self._position,
            )
            self._position = target
            if is_hop:
                self._followed += 1
                if self._followed > self._max_redirects:
                    return refuse
                try:
                    self._on_redirect(target)
                except Exception:  # noqa: BLE001 -- The callback's refusal aborts the hop.
                    return refuse
        if (
            self._credentials
            and self._initial_url
            and origin(target) == origin(self._initial_url)
        ):
            return _with_credentials(event, self._credentials)
        return zendriver.cdp.fetch.continue_request(event.request_id)

    async def finish(self) -> None:
        """Reap pending replies and disable interception without closing the tab.

        Raises:
          TimeoutError: Cleanup exceeded its independent one-second deadline.

        """
        async with asyncio.timeout_at(asyncio.get_running_loop().time() + 1.0):
            await asyncio.sleep(0)
            self._finished = True
            tasks = tuple(self._tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # Disabling Fetch resumes paused requests, including cancelled DNS checks.
            for request_id in self._unchecked:
                await self._tab.send(
                    zendriver.cdp.fetch.fail_request(
                        request_id,
                        zendriver.cdp.network.ErrorReason.ACCESS_DENIED,
                    ),
                )
            if tasks or self._errors:
                await self._tab.send(zendriver.cdp.page.stop_loading())
            await self._tab.send(zendriver.cdp.fetch.disable())
