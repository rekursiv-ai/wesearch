#!/bin/sh
# ruff: noqa: EXE003, D300, D205, T201 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync \
  python3 -m wesearch.fetch.transport.zendriver "$@"
Fetch in a real headless Chrome, for pages behind JavaScript or a bot challenge.

One Chrome per profile serves every process on the host; each fetch runs in its
own tab, and the profile keeps Chrome's cookies.

Run ``zendriver URL`` to open URL in a visible Chrome on the same profile, to
see what a fetch saw, clear a block, or log in.
'''
# fmt: on

from __future__ import annotations

from collections.abc import Callable, Coroutine, Generator, Mapping
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

from wesearch.fetch.challenge import classify_challenge, classify_http_error
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

    import zendriver

    from wesearch.types.params import HttpMethod
else:
    from wrapt import lazy_import

    zendriver = lazy_import("zendriver")  # 214ms.


__all__ = ["BrowserPool", "BrowserUnavailableError", "fetch_zendriver", "open_window"]

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
    with session.lock:
        pool = session.browsers
        if not isinstance(pool, BrowserPool):
            pool = session.browsers = BrowserPool()
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
        # Across both, or a fetch in between relaunches headless on the profile.
        with _launch_lock(profile_dir):
            port = _devtools_port(profile_dir)
            if port is not None and _port_answers(port):
                asyncio.run(_close_chrome(port))
            browser = pool.run(_launch(profile_dir))
        # Again at close: an agent may have relocked a site meanwhile.
        for host in pool.run(_watch_window(url, browser)):
            clear_host(host)
    finally:
        pool.shutdown()


class BrowserPool:
    """Connections to shared Chromes, one per profile, on one event-loop thread.

    Shutting down disconnects; Chrome stays up for other processes.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._browsers: dict[str, tuple[bool, zendriver.Browser]] = {}
        self._connect_lock = asyncio.Lock()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="loop-web-browser",
            daemon=True,
        )
        self._thread.start()

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
        except TimeoutError:
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
          RuntimeError: When the live connection has the other mode.

        """
        key = str(profile_dir.resolve())  # noqa: ASYNC240 -- The browser driver owns the blocking operation on its dedicated worker.
        async with self._connect_lock:
            with self._lock:
                existing = self._browsers.get(key)
            # Not ``Browser.stopped``: it reads a process handle an attached
            # connection lacks.
            live = (
                existing is not None
                and (connection := existing[1].connection) is not None
                and not connection.closed
                and connection.listener is not None
                and connection.listener.running
            )
            if existing is not None and live:
                if existing[0] != headless:
                    raise RuntimeError(
                        "Cannot change Zendriver launch mode for a live profile.",
                    )
                return existing[1]
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
                self._browsers[key] = (headless, connected)
            return connected

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
            browsers = [browser for _, browser in self._browsers.values()]
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
        prog="zendriver",
        description=(
            "Open a URL in a visible Chrome on the profile the zendriver "
            "transport fetches with, to see what a fetch saw or to log in. "
            "Cookies you set persist for later fetches. Close the window when "
            "done."
        ),
        # Placeholders: a real site here would be a recipe for getting past it.
        epilog=(
            "Examples:\n"
            "  zendriver https://the-site-that-failed.example/\n"
            "  zendriver https://login.example/  # seat a session cookie\n"
            "  zendriver # opens blank; navigate by hand"
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
) -> bytes:
    """Load ``url`` in a new tab, closed after; report and return its page."""
    loop = asyncio.get_running_loop()
    pages: list[tuple[int, dict[str, str], str]] = []
    # A hop leaves the current page, not the first URL: Chrome rewrites that one.
    position = [url]
    followed = [0]
    # Credentials go only to the origin the caller named, on page loads.
    credentials = {k: v for k, v in headers.items() if k.lower() in ORIGIN_BOUND}
    navigated = asyncio.Event()

    # Chrome follows redirects itself, so this is the only place to check each hop's
    # host, call ``on_redirect``, and enforce ``max_redirects`` before Chrome connects.
    # Two arguments: zendriver retries a TypeError with one, re-running the handler.
    # ``isinstance``: zendriver delivers other event types too.
    def on_paused(event: object, *_unused: object) -> None:
        if not isinstance(event, zendriver.cdp.fetch.RequestPaused):
            return
        target = event.request.url
        is_hop = _wire_url(target) != _wire_url(position[0])
        position[0] = target
        refuse = zendriver.cdp.fetch.fail_request(
            event.request_id,
            zendriver.cdp.network.ErrorReason.ACCESS_DENIED,
        )
        if is_hop:
            followed[0] += 1
            if followed[0] > max_redirects:
                _send(loop, tab, refuse)
                return
        try:
            pinned_host(target, allow_private_hosts=allow_private_hosts)
            if is_hop:
                on_redirect(target)
        except Exception:  # noqa: BLE001 -- any refusal aborts the request.
            _send(loop, tab, refuse)
            return
        # Override only to add credentials: overriding drops Chrome's Cookie
        # intermittently (crbug 40762053).
        if not credentials or origin(target) != origin(url):
            _send(loop, tab, zendriver.cdp.fetch.continue_request(event.request_id))
            return
        replaced = {name.lower() for name in credentials}
        kept = {
            k: v
            for k, v in from_plain(event.request.headers, dict[str, str]).items()
            if k.lower() not in replaced
        }
        _send(
            loop,
            tab,
            zendriver.cdp.fetch.continue_request(
                event.request_id,
                headers=[
                    zendriver.cdp.fetch.HeaderEntry(name=k, value=v)
                    for k, v in (kept | credentials).items()
                ],
            ),
        )

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

    async with asyncio.timeout(timeout_sec):
        browser = await pool.browser(profile_dir, headless=headless)
        if cookies:
            await browser.cookies.set_all(
                [
                    zendriver.cdp.network.CookieParam(name=name, value=value, url=url)
                    for name, value in cookies.items()
                ],
            )
        # The marker URL tells ``open_window`` this tab is an agent's, not the user's.
        tab = await browser.get("about:blank#wesearch-agent", new_tab=True)
        try:
            # `zendriver` types ``add_handler`` with a bare Callable, and a stub for
            # the inherited method would blank the rest of ``Connection``.
            register = cast(
                Callable[[type[object], Callable[..., None]], None],
                tab.add_handler,
            )
            register(zendriver.cdp.fetch.RequestPaused, on_paused)
            register(zendriver.cdp.network.ResponseReceived, on_document)
            register(zendriver.cdp.page.FrameNavigated, on_navigated)
            # Pages only: pausing every subresource stalls heavy pages past the
            # timeout.
            await tab.send(
                zendriver.cdp.fetch.enable(
                    patterns=[
                        zendriver.cdp.fetch.RequestPattern(
                            url_pattern="*",
                            resource_type=zendriver.cdp.network.ResourceType.DOCUMENT,
                            request_stage=zendriver.cdp.fetch.RequestStage.REQUEST,
                        ),
                    ],
                ),
            )
            # Tab-wide headers reach every subresource, cross-origin ones too, so
            # credentials go only on same-origin page loads, in ``on_paused``.
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
            # A challenge's script loads the real page after its own load
            # completes; mid-parse that page is an empty <head>, so wait for its
            # ready state too. Half the budget, so a page still walled returns its
            # wall, not a timeout. ``on_success_body``: a rendered page's CAPTCHA
            # widget (a login form's) is not a wall.
            deadline = loop.time() + timeout_sec / 2
            while True:
                # Before the read: a navigation between read and wait must count.
                navigated.clear()
                page = await tab.get_content()
                remaining = deadline - loop.time()
                if (
                    classify_challenge(page, on_success_body=True) is None
                    or remaining <= 0
                ):
                    break
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
                for cookie in await browser.cookies.get_all()
                if (domain := cookie.domain.removeprefix("."))
                and (host == domain or host.endswith(f".{domain}"))
            }
        finally:
            # ``Tab.close`` can hang on a dead connection, so it is shielded and
            # timed; an abandoned close is awaited after cancel, or it raises later
            # on the pool's loop. Failures are swallowed: raising would replace the
            # caller's result or error.
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
    status, received, _ = pages[-1] if pages else (200, {}, landed)
    if jar:
        received = {
            **received,
            "set-cookie": "\n".join(f"{k}={v}" for k, v in jar.items()),
        }
    for hop_status, hop_headers, hop_url in pages[:-1]:
        on_response(hop_status, hop_headers, hop_url)
    on_response(status, received, landed)
    # Chrome wraps a non-HTML response in its own viewer page; unwrap the payload.
    # Chrome's JSON formatter mounts empty divs after the </pre>.
    viewer = re.match(
        r"^<html[^>]*>\s*<head>.*?color-scheme.*?</head>\s*"
        r"<body[^>]*>\s*<pre[^>]*>(?P<payload>.*)</pre>"
        r"(?:\s*<div[^>]*></div>)*\s*</body>\s*</html>$",
        page.strip(),
        re.DOTALL | re.IGNORECASE,
    )
    content = (unescape(viewer.group("payload")) if viewer else page).encode()
    if status >= 400:
        raise classify_http_error(landed, status, received, content)
    return content


def _wire_url(url: str) -> str:
    """Return ``url`` as it goes on the wire: no fragment, path never empty."""
    parts = urlsplit(url)
    return urlunsplit(parts._replace(fragment="", path=parts.path or "/"))


# Scheduled, never awaited: blocking in a handler deadlocks the connection it answers.
def _send(loop: asyncio.AbstractEventLoop, tab: zendriver.Tab, command: object) -> None:
    """Send a CDP command from zendriver's event-handler thread."""
    coroutine = tab.send(
        cast("Generator[dict[str, object], dict[str, object], object]", command),
    )
    if loop.is_closed():
        coroutine.close()
        return
    loop.call_soon_threadsafe(lambda: loop.create_task(coroutine))


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
    try:
        owner = (profile_dir / "SingletonLock").readlink()
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
        _, found, rest = command.partition(marker)
        return rest.partition(" --")[0].strip() if found else ""

    if flag("--user-data-dir=") != str(profile_dir.resolve()):
        return None
    try:
        port = int(flag("--remote-debugging-port=").split()[0])
        return port or int((profile_dir / "DevToolsActivePort").read_text().split()[0])
    except (FileNotFoundError, IndexError, ValueError):
        return None


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


async def _launch(profile_dir: Path) -> zendriver.Browser:
    """Launch a visible Chrome this process owns on ``profile_dir``."""
    profile_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 -- The browser driver owns the blocking operation on its dedicated worker.
    _tolerate_late_cdp_replies()
    executable = await asyncio.to_thread(_chrome_for_testing)
    try:
        return await zendriver.start(
            zendriver.Config(
                headless=False,
                user_data_dir=str(profile_dir),
                sandbox=os.geteuid() != 0,
                browser_executable_path=executable,
                browser_args=["--use-mock-keychain"] if executable else [],
                browser_connection_timeout=0.5,
                browser_connection_max_tries=6,
            ),
        )
    except Exception as error:
        raise BrowserUnavailableError(f"Could not launch Chrome: {error}") from error


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
    for page in (url, *(locked for locked in challenged_urls() if locked != url)):
        try:
            tab = await browser.get("about:blank", new_tab=True)
            await tab.get(page)
        except Exception:
            logger.warning("could not open %s", page, exc_info=True)
    try:
        while not browser.stopped:  # noqa: ASYNC110 -- no event source; poll the flag.
            await asyncio.sleep(0.5)
    except BaseException:
        # Swallows every failure: raising would replace the error in flight.
        try:
            async with asyncio.timeout(30.0):
                await browser.stop()
        except Exception:
            logger.warning("browser stop failed; killing it", exc_info=True)
            process: object = getattr(browser, "_process", None)
            if process is not None:
                with contextlib.suppress(OSError):
                    cast(_Killable, process).kill()
        raise
    return visited


class _Killable(Protocol):
    def kill(self) -> None: ...


def _shared_chrome(profile_dir: Path, *, headless: bool) -> int:
    """Return the DevTools port of the Chrome on ``profile_dir``, launching one."""
    with _launch_lock(profile_dir):
        port = _devtools_port(profile_dir)
        if port is not None and _port_answers(port):
            return port
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "DevToolsActivePort").unlink(missing_ok=True)
        executable = _chrome_for_testing()
        config = zendriver.Config(
            headless=headless,
            user_data_dir=str(profile_dir),
            # Chrome's sandbox refuses to run as root.
            sandbox=os.geteuid() != 0,
            browser_executable_path=executable,
            browser_args=["--use-mock-keychain"] if executable else [],
            host="127.0.0.1",
        )
        try:
            # Not ``port=0`` in Config: it drops a falsy port, and Chrome then has no
            # DevTools. A new session keeps Chrome alive past this process.
            process = subprocess.Popen(  # noqa: S603 -- argv is zendriver's own launch config.
                [
                    str(config.browser_executable_path),
                    *config(),
                    "--remote-debugging-port=0",
                    "about:blank",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as error:
            raise BrowserUnavailableError(
                f"Could not launch Chrome: {error}",
            ) from error
        _launched.append(process)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise BrowserUnavailableError(
                    f"Could not launch Chrome: it exited with code {process.returncode}.",
                )
            with contextlib.suppress(FileNotFoundError, IndexError, ValueError):
                port = int((profile_dir / "DevToolsActivePort").read_text().split()[0])
                if _port_answers(port):
                    return port
            time.sleep(0.05)
        process.kill()
        process.wait()
        raise BrowserUnavailableError(
            "Could not launch Chrome: DevTools did not come up in 3s.",
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


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
