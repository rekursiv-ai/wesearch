#!/bin/sh
# ruff: noqa: EXE003, D300, D205, T201 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync \
  python3 -m wesearch.fetch.transport.zendriver "$@"
Real-browser fetch backend for ``wesearch.fetch`` (opt-in).

Drives a headless Chrome via ``zendriver`` so pages gated behind a run-the-JS
challenge or an interactive CAPTCHA load where the curl/stdlib backends get a
wall. Select it per call with
``RequestParams(policy=PolicyParams(transport="zendriver"))``; the page runs under a
persistent Chrome profile, so cookies you seat (e.g. by logging in) carry
across fetches.

One Chrome serves every process on the host. The first fetch launches it
DETACHED (its own session, owned by no caller), and every fetch -- from any
process -- attaches over DevTools and works in its own tab. Chrome allows one
owner per profile, so the alternative, a Chrome per process, made each new
process evict the last one's browser mid-fetch.

Run ``zendriver URL`` to open that URL in a HEADED Chrome on the same
profile -- to debug a fetch that errored, to clear a site's block, or to seat a
login.
'''
# fmt: on

from __future__ import annotations

from collections.abc import Callable, Coroutine, Generator, Iterator, Mapping
from html import unescape
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, TypeVar, cast
from urllib.parse import urlparse, urlsplit, urlunsplit

import argparse
import asyncio
import atexit
import contextlib
import fcntl
import inspect
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
import warnings

from wesearch.fetch.challenge import classify_challenge
from wesearch.fetch.common import apply_redirect, origin, pinned_host
from wesearch.fetch.gate import challenged_urls, clear_host
from wesearch.lib.userdirs import cache_dir, data_dir


if TYPE_CHECKING:
    from argparse import ArgumentParser
    from concurrent.futures import Future

    from zendriver.core import connection

    import zendriver

    from wesearch.types.params import Trust
else:
    from wrapt import lazy_import

    # Deferred: importing zendriver pulls a large CDP-binding tree (~200ms,
    # measured) and is paid only when a browser fetch actually runs, never at
    # ``wesearch`` import.
    zendriver = lazy_import("zendriver")
    connection = lazy_import("zendriver.core.connection")


__all__ = [
    "BrowserResult",
    "BrowserUnavailableError",
    "close_launched_browsers",
    "fetch_zendriver",
    "open_instance",
    "shutdown_browsers",
]

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class BrowserUnavailableError(OSError):
    """Chrome could not be launched or connected to on this host.

    An ``OSError`` because it is one at the boundary: an unreachable browser is a
    failed connection like any other. Every caller that already turns a transport's
    ``OSError`` into its own error (a paper backend, a search backend) then does so
    for a host with no Chrome too, instead of the escape that aborted a fused
    paper search although Semantic Scholar had answered.

    A capability condition (Chrome absent, incompatible, or unable to bind its
    DevTools port), distinct from a fetch or parse failure: callers that need a
    browser cannot proceed, and environments without a usable one (CI, headless
    boxes) should treat it as "browser subsystem unavailable" rather than a bug.
    """


class BrowserResult(NamedTuple):
    """What a browser fetch yields: the rendered page and the cookies it holds.

    Attributes:
      body: The rendered ``document`` HTML, UTF-8 encoded.
      cookies: Cookies the browser holds for ``final_url``'s domain
        (``name -> value``), for the caller to persist and thread onward.
      final_url: Where the navigation ENDED UP, which is the origin the cookies
        belong to. Reported rather than left implicit: Chrome follows its own
        redirects, so a caller that assumed the requested URL filed the
        target's session cookie under the source and sent it back there.

    """

    body: bytes
    cookies: dict[str, str]
    final_url: str


def fetch_zendriver(
    url: str,
    *,
    profile_dir: Path,
    timeout_sec: float = 30.0,
    headless: bool = True,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    trust: Trust = "untrusted",
    max_redirects: int = 10,
    on_redirect: Callable[[str], None] | None = None,
) -> BrowserResult:
    """Fetch ``url`` in a tab of the shared headless Chrome; return body and cookies.

    Attaches to the host's one Chrome on ``profile_dir`` (launching it detached
    if none runs), navigates a fresh tab to ``url``, waits for the load to
    complete, then returns the rendered HTML and the cookies the browser holds
    for the URL's domain. Page-state validation belongs to the provider
    consuming the rendered response.

    Args:
      url: Fully-qualified URL to navigate to.
      profile_dir: Chrome ``user_data_dir`` supplying the logged-in identity.
      timeout_sec: Overall budget for navigation + load, in seconds.
      headless: Run Chrome headless (the default); ``False`` opens a window.
      headers: Extra headers for this tab. Browser-identity headers
        (User-Agent, Accept*, Sec-*, Referer, ...) are dropped: Chrome sends
        its own, and overriding them is what gets a real browser walled.
      cookies: Cookies seeded for ``url`` before navigation.
      trust: Provenance of the URL. Under ``"untrusted"`` every DOCUMENT request
        -- the navigation and each redirect hop Chrome follows itself -- is
        validated to a public address before the socket opens. Subresources are
        deliberately not intercepted; see :func:`_guard_requests`. Required
        rather than defaulted, because a transport that cannot express the
        policy cannot be held to it.
      max_redirects: Document hops Chrome may follow; ``0`` disables redirects,
        matching :class:`~wesearch.types.params.RetryParams`. Enforced in
        the request guard, the only seam that sees a hop before it runs.
      on_redirect: Called with each document hop before it is followed; raise to
        abort it.

    Returns:
      result: The rendered body and the browser's cookies for the URL's domain.

    """
    return _pool().run(
        _navigate(
            url,
            profile_dir=profile_dir,
            timeout_sec=timeout_sec,
            headless=headless,
            headers=_without_browser_identity(headers),
            cookies=cookies,
            trust=trust,
            max_redirects=max_redirects,
            on_redirect=on_redirect,
        ),
        # Above the coroutine's own budget, so a normal timeout still raises
        # from inside ``_navigate``, which closes its tab and reports the wall.
        timeout_sec=timeout_sec + 30,
    )


def shutdown_browsers() -> None:
    """Disconnect this process from every shared Chrome and stop the pool's loop.

    Disconnects only: the Chrome stays up for every other process using it.
    Idempotent, and a NO-OP when no pool has been created -- it must never
    construct one just to tear it down. The singleton is cleared so the next
    browser fetch builds a fresh pool.
    """
    global _pool_singleton  # noqa: PLW0603 -- reset the shared pool after teardown.
    with _pool_lock:
        pool = _pool_singleton
        _pool_singleton = None
    if pool is not None:
        pool.shutdown()


def close_launched_browsers() -> None:
    """Terminate every shared Chrome THIS process launched, and reap it.

    Production never calls this -- a shared Chrome must outlive the process that
    happened to launch it. A test run does, so the Chromes its isolated profiles
    started do not outlive the run.
    """
    while _launched:
        process = _launched.pop()
        with contextlib.suppress(OSError):
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def open_instance(url: str, *, profile_dir: Path | None = None) -> None:
    """Open a HEADED Chrome on the profile dir at ``url``; block until closed.

    Closes the shared headless Chrome (a profile has one owner), launches a
    visible one under ``profile_dir``, navigates to ``url``, and blocks until the
    user closes the window. Use it to eyeball a URL the headless
    :func:`fetch_zendriver` backend failed on, or to seat a login: cookies set
    there persist for later headless fetches. Fetches made meanwhile attach to
    the visible window; the next one after it closes relaunches headless.

    Every site a bot challenge locked opens in its own tab. Each site the user
    loads in the window -- never one an agent's tab loads -- has its lock lifted
    on load and again when the window closes.

    Args:
      url: The page to open -- typically the URL whose headless fetch you are
        debugging.
      profile_dir: Chrome ``user_data_dir`` to open; defaults to the
        ``fetch-zendriver`` profile under the wesearch data directory.

    """
    target = (
        data_dir() / "rekursiv-ai" / "wesearch" / "fetch-zendriver"
        if profile_dir is None
        else profile_dir
    )
    # Held across close AND launch: a fetch landing between them would relaunch
    # headless and take the profile lock the headed launch needs.
    with _launch_lock(target):
        _close_shared_browser(target)
        browser = _pool().run(_launch_browser(target, headless=False))
    # Cleared again at close: an agent may have relocked a site while the user
    # was still solving its challenge.
    for host in _pool().run(_open_instance(url, browser)):
        clear_host(host)


# Chrome's generated viewer: a color-scheme meta it inserts itself, then the
# payload as the document's ONLY content. Anchored at both ends so a real page
# that merely opens with a <pre> does not match.
_VIEWER_SHELL = re.compile(
    r"^<html[^>]*>\s*<head>.*?color-scheme.*?</head>\s*"
    r"<body[^>]*>\s*<pre[^>]*>(?P<payload>.*)</pre>"
    # Chrome appends its own JSON-formatter mount AFTER the </pre>, so the
    # payload is not the last thing in the body. Requiring </pre></body>
    # adjacency matched the synthetic fixture and missed every real response.
    r"(?:\s*<div[^>]*></div>)*\s*</body>\s*</html>$",
    re.DOTALL | re.IGNORECASE,
)


# Numeric, not lexical: these names carry a dotted version (``mac_arm-147.0.7727.57``)
# or a bare revision (``chromium-1217``), and comparing them as text ranks 99 above 100.
def _build_order(build: Path) -> tuple[list[int], str]:
    """Sort key ranking a download directory by version, newest highest."""
    # The name second, so unversioned directories still order deterministically
    # rather than by whatever iteration order the filesystem returns.
    return [int(m[0]) for m in re.finditer(r"\d+", build.name)], build.name


def _process_command(pid: int, *, proc_root: Path, platform: str) -> str:
    """Read a process command from the host's process interface."""
    proc_command = proc_root / str(pid) / "cmdline"
    if proc_command.is_file() or platform != "darwin":
        return proc_command.read_bytes().replace(b"\0", b" ").decode()
    result = subprocess.run(  # noqa: S603 -- PID is parsed as an integer.
        ["/bin/ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise OSError(f"Could not inspect process {pid}.")
    return result.stdout


# Chrome's setuid sandbox refuses to start as root, so a root context (CI, containers)
# must pass ``--no-sandbox``. A normal desktop user keeps the sandbox -- disabling it
# there needlessly weakens security AND makes Chrome show a persistent "unsupported
# command-line flag: --no-sandbox" banner.
def _command_flag(command: str, marker: str) -> str:
    """Extract a Chrome flag value from NUL- or space-flattened proc args."""
    _, found, suffix = command.partition(marker)
    if not found:
        return ""
    return suffix.partition(" --")[0].strip()


class _BrowserPool:
    """Connections to shared Chromes over one persistent event loop on a daemon thread.

    zendriver browsers bind to the loop running their coroutines, so a stable
    loop is a hard requirement for reuse across sync calls. This pool owns that
    loop on a background thread and dispatches every browser coroutine to it via
    :meth:`run`, keeping one live DevTools connection per profile.

    It owns connections, never Chromes: a Chrome is shared by every process on
    the host (see :func:`_shared_browser`), so closing this pool disconnects
    and leaves the browser running for everyone else.

    Reuse justifies the machinery: eight matched ``https://example.com/``
    requests measured a 0.130-second median on a warm browser versus 4.196
    seconds when launching Chrome per request, a 32.28x speedup.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._browsers: dict[str, tuple[bool, zendriver.Browser]] = {}
        self._connect_lock = asyncio.Lock()
        self._lock = threading.Lock()
        # Started LAST, so every field _run_loop touches exists before it runs.
        self._thread = threading.Thread(
            target=self._run_loop,
            name="loop-web-browser",
            daemon=True,
        )
        self._thread.start()

    def run(self, coro: Coroutine[object, object, _T], *, timeout_sec: float = 0) -> _T:
        """Run a coroutine on the pool's loop from a sync caller; return its result.

        Args:
          coro: The browser coroutine to run on the pool's loop.
          timeout_sec: Ceiling on the WAIT, in seconds; ``0`` waits forever. A
            self-bounding coroutine still needs it: once the loop stops, the
            coroutine is never scheduled and its own timeout never arms, so the
            caller waits on a future nothing will complete.

        Returns:
          result: Whatever the coroutine returned.

        Raises:
          TimeoutError: If the wait exceeds ``timeout_sec``.

        """
        future: Future[_T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout_sec or None)
        except TimeoutError:
            # A no-op when the coroutine timed out internally; it matters for an
            # unscheduled one, which must not later drive a tab nobody harvests.
            future.cancel()
            raise

    async def browser(self, profile_dir: Path, *, headless: bool) -> zendriver.Browser:
        """Return a live connection to the shared Chrome on ``profile_dir``.

        Args:
          profile_dir: Chrome user-data directory path.
          headless: Launch mode, used only if no Chrome is running yet. A
            running headed Chrome (an operator's ``zendriver`` window)
            serves headless requests as well.

        Returns:
          browser: A connected zendriver.Browser, reused while it stays live.

        """
        # Resolved: two spellings of one profile are one user-data dir, and
        # Chrome allows it a single owner.
        key = str(profile_dir.resolve())  # noqa: ASYNC240 -- The browser driver owns the blocking operation on its dedicated worker.
        async with self._connect_lock:
            with self._lock:
                existing = self._browsers.get(key)
            if existing is not None and _is_live(existing[1]):
                if existing[0] != headless:
                    raise RuntimeError(
                        "Cannot change Zendriver launch mode for a live profile.",
                    )
                return existing[1]
            connected = await self._connect(profile_dir, headless=headless)
            with self._lock:
                self._browsers[key] = (headless, connected)
            return connected

    def shutdown(self, *, budget_sec: float = 5.0) -> None:
        """Disconnect from every shared Chrome and stop the loop thread (idempotent).

        Args:
          budget_sec: TOTAL ceiling on closing every connection, shared across
            them rather than applied to each, so teardown fits inside the grace
            period a supervisor allows between SIGTERM and SIGKILL.

        """
        if self._loop.is_closed():
            return
        with self._lock:
            browsers = [browser for _, browser in self._browsers.values()]
            self._browsers.clear()
        deadline = time.monotonic() + budget_sec
        for browser in browsers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                self.run(_disconnect(browser), timeout_sec=remaining)
            except Exception:
                logger.warning(
                    "browser disconnect failed during shutdown",
                    exc_info=True,
                )
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()

    async def _connect(self, profile_dir: Path, *, headless: bool) -> zendriver.Browser:
        """Attach to the shared Chrome on ``profile_dir``, launching it if needed."""
        port = await asyncio.to_thread(_shared_browser, profile_dir, headless=headless)
        return await _attach(port)

    # zendriver's CDP dispatch calls the deprecated ``asyncio.iscoroutinefunction``
    # (connection.py) and leaves reader pipes for the GC to close, emitting a
    # ``DeprecationWarning`` / ``ResourceWarning`` from inside the browser coroutine.
    # Under a ``-W error`` caller (the repo's pytest turns every warning into an
    # exception) that warning would raise INSIDE the awaited CDP handler and wedge the
    # fetch. The pool loop thread runs only zendriver coroutines, so scoping the filter
    # to this thread's ``run_forever`` mutes the upstream noise without hiding warnings
    # from any caller's own code.
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


# A connection is dead when its websocket closed (Chrome exited, or an operator closed
# the headed window): zendriver then clears ``websocket``, and its listener task ends.
# ``Browser.stopped`` cannot say so -- it reads the Popen handle, which an attached
# connection never has.
def _is_live(browser: zendriver.Browser) -> bool:
    """Whether an attached connection can still carry commands."""
    connection = browser.connection
    return (
        connection is not None
        and not connection.closed
        and connection.listener is not None
        and connection.listener.running
    )


async def _attach(port: int) -> zendriver.Browser:
    """Connect to the Chrome serving DevTools on ``port``."""
    _tolerate_late_cdp_replies()
    try:
        return await zendriver.start(
            host="127.0.0.1",
            port=port,
            browser_connection_timeout=0.5,
            browser_connection_max_tries=6,
        )
    except Exception as error:
        # ``zendriver`` raises a bare ``Exception`` when DevTools never answers.
        raise BrowserUnavailableError(
            f"Could not attach to Chrome on DevTools port {port}: {error}",
        ) from error


# Disconnects WITHOUT ``Browser.stop``: stop sends ``Browser.close``, which would shut
# Chrome down for every other process attached to it.
async def _disconnect(browser: zendriver.Browser) -> None:
    """Drop this process's connection, leaving the shared Chrome running."""
    connection = browser.connection
    if connection is not None:
        await connection.aclose()


# Serialized across processes by a lock file beside the profile, so two processes
# arriving together launch one Chrome, not two fighting over the profile lock.
def _shared_browser(profile_dir: Path, *, headless: bool) -> int:
    """Return the DevTools port of the shared Chrome, launching it if none runs."""
    with _launch_lock(profile_dir):
        port = _devtools_port(profile_dir)
        if port is not None and _port_answers(port):
            return port
        return _launch_detached(profile_dir, headless=headless)


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


def _port_answers(port: int) -> bool:
    """Whether something accepts connections on the DevTools ``port``."""
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
    except OSError:
        return False
    return True


# ``start_new_session`` puts Chrome in its own session and process group, so a caller's
# exit, a Ctrl-C to its terminal, or a group kill of its tree leaves the shared browser
# running. Port 0 lets Chrome pick a free port and report it in ``DevToolsActivePort``,
# which avoids the race of choosing one here and binding it later.
def _launch_detached(
    profile_dir: Path,
    *,
    headless: bool,
    connect_timeout_sec: float = 3.0,
) -> int:
    """Launch a Chrome no caller owns on ``profile_dir``; return its DevTools port."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "DevToolsActivePort").unlink(missing_ok=True)
    executable = _fetch_browser()
    config = zendriver.Config(
        headless=headless,
        user_data_dir=str(profile_dir),
        # Sandboxed unless we are root.
        sandbox=os.geteuid() != 0,
        browser_executable_path=executable,
        browser_args=_fetch_browser_args(executable),
        host="127.0.0.1",
    )
    try:
        # The port flag is appended here, not passed as ``port=0``: ``Config``
        # emits it only for a truthy port, so 0 silently launched Chrome with no
        # DevTools at all.
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
        raise BrowserUnavailableError(f"Could not launch Chrome: {error}") from error
    _launched.append(process)
    deadline = time.monotonic() + connect_timeout_sec
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise BrowserUnavailableError(
                f"Could not launch Chrome: it exited with code {process.returncode}.",
            )
        try:
            port = int((profile_dir / "DevToolsActivePort").read_text().split()[0])
        except (FileNotFoundError, IndexError, ValueError):
            time.sleep(0.05)
            continue
        if _port_answers(port):
            return port
        time.sleep(0.05)
    process.kill()
    process.wait()
    raise BrowserUnavailableError(
        f"Could not launch Chrome: DevTools did not come up in {connect_timeout_sec}s.",
    )


# Every Chrome this process launched, so a test run can reap the ones it started. Not
# consulted in production: a launched Chrome belongs to the host, not to its launcher.
_launched: list[subprocess.Popen[bytes]] = []


# ``Transaction.__call__`` calls ``set_result`` unconditionally, so a reply that lands
# after its future was CANCELLED raises ``InvalidStateError`` -- inside
# ``Listener.listener_loop``, which kills the listener task for the whole connection.
# Every fetch here is cancellable by construction (:func:`_navigate` wraps the
# navigation in ``asyncio.timeout``), and Chrome answers the in-flight ``Page.navigate``
# afterwards, so the race is not exotic: it is what a timed-out fetch does.
#
# The damage outlives the fetch that caused it. A dead listener stops dispatching CDP
# replies AND events, so the pooled browser is silently deaf to every later fetch -- the
# visible symptom is only the "Task exception was never retrieved" traceback pytest
# prints at teardown.
#
# Armed on the launch path, before any CDP traffic exists, and idempotent so repeated
# launches patch once. Upstream carries the same fix as PR #242, which was auto-closed
# for inactivity rather than merged.
def _tolerate_late_cdp_replies() -> None:
    """Make zendriver drop a CDP reply whose transaction already finished."""
    # Reached through the module, not bound at import: ``connection`` here is a
    # ``lazy_import`` proxy, and a proxied CLASS answers ``__call__`` with its
    # own construction hook rather than the class attribute -- patching that
    # leaves the real ``Transaction`` untouched.
    transaction = connection.Transaction
    vendor = cast(Callable[..., None], transaction.__call__)
    # Keyed on the DEFINING module, so a second launch does not stack a second
    # wrapper: an armed class carries this module's function.
    if vendor.__module__ == __name__:
        return

    def call(self: connection.Transaction[object], **response: object) -> None:
        if not self.done():
            vendor(self, **response)

    transaction.__call__ = call


# On macOS a headless Chrome launched from the installed bundle registers as
# ``com.google.Chrome`` and outranks the user's own windows, so LaunchServices hands it
# every ``open https://...`` -- what a command-click becomes -- and it drops them. Links
# stop opening system-wide for the duration of a fetch, and ``open`` still exits 0.
#
# No launch flag avoids this; only a different bundle id does, and it must come from the
# vendor, since editing a copy's breaks Chrome's signature and the kernel then kills it.
# Chrome for Testing ships as ``com.google.chrome.for.testing``.
def _fetch_browser(*, platform: str = sys.platform) -> str:
    """Return a Chrome that will not capture the user's link clicks."""
    # Elsewhere a URL reaches a browser through xdg-open and the desktop
    # file, never a running process, so there is no capture to prevent.
    if platform != "darwin":
        return ""
    for candidate in _fetch_browser_candidates():
        if candidate.is_file():
            return str(candidate)
    logger.debug("no Chrome for Testing found; falling back to zendriver's Chrome")
    return ""


def _fetch_browser_args(
    executable: str,
    *,
    mock_keychain: str = "--use-mock-keychain",
) -> list[str]:
    """Extra Chrome flags the fetch browser needs."""
    return [mock_keychain] if executable else []


# Read from wherever Puppeteer or Playwright already downloaded one; neither is a
# dependency, and nothing here installs a browser.
def _fetch_browser_candidates() -> Iterator[Path]:
    """Chrome for Testing binaries, newest build first."""
    roots = (
        # Puppeteer hardcodes ~/.cache on every platform, so this one does
        # NOT follow the macOS convention that ``cache_dir()`` implements.
        Path.home() / ".cache" / "puppeteer" / "chrome",  # noqa: TID251 -- vendor fixed path, not ours (AGENTS.md rule 3)  # house-ignore[xdg-literal] -- Vendor CLI's fixed home path, not ours (AGENTS.md rule 3).
        cache_dir() / "ms-playwright",
    )
    for root in roots:
        if not root.is_dir():
            continue
        for build in sorted(root.iterdir(), key=_build_order, reverse=True):
            # Both arches: the tag names the download, and Rosetta runs an
            # x64 build on Apple silicon.
            yield from (
                build / arch / name / "Contents" / "MacOS" / stem
                for arch in ("chrome-mac-arm64", "chrome-mac-x64")
                for name, stem in (
                    ("Google Chrome for Testing.app", "Google Chrome for Testing"),
                    ("Chromium.app", "Chromium"),
                )
            )


def _devtools_port(
    profile_dir: Path,
    *,
    proc_root: Path = Path("/proc"),
    platform: str = sys.platform,
) -> int | None:
    """Read the verified profile owner's active DevTools port."""
    try:
        owner = (profile_dir / "SingletonLock").readlink()
        pid = int(str(owner).rsplit("-", 1)[1])
        command = _process_command(pid, proc_root=proc_root, platform=platform)
    except (FileNotFoundError, IndexError, OSError, UnicodeError, ValueError):
        return None
    if _command_flag(command, "--user-data-dir=") != str(profile_dir.resolve()):
        return None
    try:
        port_text = _command_flag(command, "--remote-debugging-port=")
        if not port_text:
            return None
        port = int(port_text.split()[0])
        if port:
            return port
        return int((profile_dir / "DevToolsActivePort").read_text().splitlines()[0])
    except (FileNotFoundError, IndexError, ValueError):
        return None


# The single pooled browser manager, built once on first browser fetch. A
# deliberate module singleton: it owns a live loop thread and DevTools
# connections -- shared runtime resources, not a tunable.

_pool_singleton: _BrowserPool | None = None
_pool_lock = threading.Lock()


def main() -> int:
    """Open a URL in a headed Chrome on the backend's profile; return exit code.

    Returns:
      exit_code: 0 on success.

    """
    parser = argparse.ArgumentParser(
        prog="zendriver",
        description=(
            "Open a URL in a headed Chrome on the zendriver backend's dedicated "
            "profile -- the same profile the headless "
            'RequestParams(policy=PolicyParams(transport="zendriver")) fetch uses. Use '
            "it to debug a fetch that errored: you see exactly what Chrome "
            "renders (a challenge, a login wall, a broken page), and any cookies "
            "you seat while there (e.g. by logging in) persist for later "
            "headless fetches. Close the window when done."
        ),
        # Placeholder hostnames only. Naming a real site here turns a debugging
        # aid into a how-to for getting past that site's defenses, which is the
        # same line the export draws by withholding the learned-domain roster.
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
    print(
        f"Opening {flags.url} in Chrome on "
        f"{data_dir() / 'rekursiv-ai' / 'wesearch' / 'fetch-zendriver'} -- "
        "close the window when done.",
    )
    open_instance(flags.url)
    print("Window closed.")
    return 0


def _add_arguments(parser: ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "url",
        nargs="?",
        default="about:blank",
        help="The URL to open (typically the one whose headless fetch failed). "
        "Omit to open a blank page and navigate by hand.",
    )


# Blocks until the window is closed, after opening a tab per challenge-locked site.
async def _open_instance(url: str, browser: zendriver.Browser) -> set[str]:
    """Navigate the headed ``browser`` to ``url``; return the hosts the user loaded."""
    visited = _watch_visits(browser)
    try:
        await _navigate_tab(browser, url, blank="about:blank")
        for locked in challenged_urls():
            if locked != url:
                await _open_locked(browser, locked)
        # Block until the user closes the window (Chrome exits, so the browser reports
        # stopped). Polled, not event-driven: the window-closed signal is Chrome's
        # process exit, which zendriver exposes only as the polled ``stopped`` flag.
        while not browser.stopped:  # noqa: ASYNC110 -- no event source; poll the flag.
            await asyncio.sleep(0.5)
    except BaseException:
        await _stopped(browser)
        raise
    return visited


# One tab that will not load must not take the window, and the user's other tabs,
# down with it.
async def _open_locked(browser: zendriver.Browser, url: str) -> None:
    """Open a challenge-locked site in its own tab of the headed window."""
    try:
        await _navigate_tab(browser, url, blank="about:blank")
    except Exception:
        logger.warning("could not open locked site %s", url, exc_info=True)


# Agents' fetches run as tabs of this same window, and their loads must not count as
# the user's: a blocked agent fetch would otherwise unlock the site that blocked it.
# Every agent tab is born at :data:`_AGENT_TAB`, and a tab one of them opens is an
# agent's too. Measured: an operator connection sees each agent tab's creation at that
# URL, then every page it commits.
def _watch_visits(browser: zendriver.Browser) -> set[str]:
    """Lift the lock of each site the user loads; return the set of those hosts."""
    agent_tabs: set[str] = set()
    current: dict[str, str] = {}
    visited: set[str] = set()

    async def on_target(
        event: zendriver.cdp.target.TargetCreated
        | zendriver.cdp.target.TargetInfoChanged,
        *_unused: object,
    ) -> None:
        info = event.target_info
        if info.type_ != "page" or info.target_id in agent_tabs:
            return
        if info.url == _AGENT_TAB or info.opener_id in agent_tabs:
            agent_tabs.add(info.target_id)
            return
        # A load re-fires this event as its title changes; only a new URL is a visit.
        if current.get(info.target_id) == info.url:
            return
        current[info.target_id] = info.url
        parsed = urlparse(info.url)
        host = parsed.hostname or ""
        if parsed.scheme in ("http", "https") and host:
            visited.add(host)
            await asyncio.to_thread(clear_host, host)

    connection = browser.connection
    if connection is None:
        raise BrowserUnavailableError("The headed Chrome has no DevTools connection.")
    # Cast for the same reason as in :func:`_main_frame_navigations`.
    register = cast(
        Callable[[type[object], Callable[..., object]], None],
        connection.add_handler,
    )
    register(zendriver.cdp.target.TargetCreated, on_target)
    register(zendriver.cdp.target.TargetInfoChanged, on_target)
    return visited


# The blank page each agent fetch's tab opens at; the fragment marks the tab as an
# agent's for :func:`_watch_visits` and never reaches the network.
_AGENT_TAB: Final = "about:blank#wesearch-agent"


# ``Browser.stop`` awaits ``connection.send(cdp.browser.close())`` with no ceiling; its
# own ``except Exception`` cannot cover this, because an await that never returns raises
# nothing to catch. Unbounded, that turns a reportable setup failure into a hang: this
# runs while the real error is in flight, and nothing above supplies a deadline --
# :func:`open_instance` calls ``run`` with no ``timeout_sec``, which waits forever by
# contract.
#
# ``budget_sec`` matches :meth:`_BrowserPool.shutdown`'s ceiling on the same call. Long,
# deliberately: a healthy stop terminates Chrome and waits up to 3s for the process, so
# a tight bound would abandon live browsers that were about to exit.
#
# Every failure is swallowed, not just the timeout: this runs from an ``except
# BaseException`` with the real error in flight, so anything raised here REPLACES it --
# the operator learns the browser would not close and never learns why the page failed.
# Cleanup reports nothing over its caller.
async def _stopped(browser: zendriver.Browser, *, budget_sec: float = 30.0) -> None:
    """Stop ``browser``, giving up rather than waiting on a wedged connection."""
    try:
        async with asyncio.timeout(budget_sec):
            await browser.stop()
    except TimeoutError:
        logger.warning("browser stop timed out; killing the browser process")
        _kill_browser_process(browser)
    except Exception:
        logger.warning(
            "browser stop failed; killing the browser process",
            exc_info=sys.exc_info(),
        )
        _kill_browser_process(browser)


async def _navigate_tab(
    browser: zendriver.Browser,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    trust: Trust = "untrusted",
    max_redirects: int = 10,
    on_redirect: Callable[[str], None] | None = None,
    blank: str = _AGENT_TAB,
) -> zendriver.Tab:
    """Open a blank tab, arm the per-request guard, apply headers, navigate."""
    tab = await browser.get(blank, new_tab=True)
    try:
        await _guard_requests(
            tab,
            url,
            trust=trust,
            on_redirect=on_redirect,
            max_redirects=max_redirects,
            headers=headers,
        )
        # Installed tab-wide are ONLY the headers no origin owns. An
        # origin-bound one (a credential, an extended client hint) is attached
        # by the guard, per document hop, to the origin entitled to it:
        # ``set_extra_http_headers`` reaches every request the tab makes,
        # including the cross-origin subresources the guard never sees.
        ambient = {
            name: value
            for name, value in (headers or {}).items()
            if name.lower() not in _origin_bound()
        }
        if ambient:
            await tab.send(zendriver.cdp.network.enable())
            await tab.send(
                zendriver.cdp.network.set_extra_http_headers(
                    zendriver.cdp.network.Headers(ambient),
                ),
            )
        await tab.get(url)
        await tab.wait_for_ready_state("complete")
    except BaseException:
        await _closed(tab)
        raise
    return tab


# ``Tab.close`` awaits ``Target.closeTarget``'s reply with no ceiling of its own, so a
# connection that stopped answering parks here forever. That is survivable only while
# the caller can still be cancelled -- and on every path that reaches here from a
# CANCELLED body it cannot: an ``asyncio.timeout`` scope delivers exactly one
# cancellation, the body consumed it, and this cleanup runs unguarded. The enclosing
# coroutine then never completes, so its ``TimeoutError`` is never raised and the caller
# waits out :func:`fetch_zendriver`'s ``timeout_sec + 30`` for a wall-less one instead.
#
# ``shield`` is what makes the bound real here: the cleanup often runs with a
# cancellation already in flight, and an unshielded await would re-deliver it to the
# close rather than time the close out.
#
# An abandoned close is CANCELLED AND AWAITED, never merely cancelled: ``cancel`` only
# requests it, so returning straight after leaves the task pending on a loop that
# outlives this call -- the pool's loop runs forever (see
# :meth:`_BrowserPool._run_loop`), so the task finalizes late or not at all and raises
# into a loop nobody is watching. Under ``asyncio.run`` the loop dies immediately, which
# is why a test on one cannot see this.
#
# Every failure is swallowed, not just the timeout: this runs from :func:`_navigate`'s
# ``finally`` and :func:`_navigate_tab`'s ``except``, so anything raised here replaces
# the result or error the caller was about to see. Cleanup reports nothing over its
# caller.
async def _closed(tab: zendriver.Tab, *, budget_sec: float = 5.0) -> None:
    """Close ``tab``, giving up rather than waiting on a wedged connection."""
    task = asyncio.ensure_future(tab.close())
    try:
        async with asyncio.timeout(budget_sec):
            await asyncio.shield(task)
    except TimeoutError:
        task.cancel()
        # Shielded again: this await inherits any cancellation already in
        # flight, and re-delivering it here would abandon the reaping too.
        await asyncio.gather(asyncio.shield(task), return_exceptions=True)
        logger.debug("tab close timed out; abandoning the tab")
    except Exception:
        logger.debug("tab close failed; abandoning the tab", exc_info=sys.exc_info())


# The header transports re-validate each redirect hop (``curl.py`` and ``stdlib.py``
# both call :func:`pinned_host` per hop) because
# :func:`wesearch.fetch.common.pinned_host` states the rule: a redirect target is a
# URL like any other, and skipping the re-check is the classic SSRF bypass. Chrome
# follows redirects ITSELF, so validating only the URL the caller passed left every
# subsequent hop -- and every subresource -- reaching the network with nothing watching.
# A public URL redirecting to ``169.254.169.254`` or loopback was fetched, and the
# transport learned of it only after the response had already been read.
#
# ``Fetch.requestPaused`` is the one seam that runs before the connection, so the same
# ``pinned_host`` that guards curl guards Chrome, and ``on_redirect`` regains the pre-
# follow abort :class:`~wesearch.types.params.ObserveParams` documents. A rejected
# request is failed with ``AccessDenied`` rather than silently continued: Chrome
# surfaces that as a navigation error, which is the honest outcome for a target policy
# forbids.
#
# ``max_redirects`` is enforced here for the same structural reason: Chrome follows hops
# itself, so this handler is the only place that can refuse one. Without it the browser
# leg silently ignored a budget both header transports honor, and ``RetryParams``
# documents as "0 disables".
async def _guard_requests(
    tab: zendriver.Tab,
    url: str,
    *,
    trust: Trust,
    on_redirect: Callable[[str], None] | None,
    max_redirects: int,
    headers: dict[str, str] | None = None,
) -> None:
    """Validate every DOCUMENT request this tab makes BEFORE Chrome connects."""
    origin_url = url
    # Captured here, where a running loop is guaranteed: zendriver invokes the
    # handler from its own connection thread, which has no running loop of its
    # own, so a ``get_running_loop`` inside the callback raises.
    loop = asyncio.get_running_loop()
    # Where the tab is NOW, not the one URL the caller named. A hop is a
    # document request that moves off the CURRENT document, which only a
    # running position can answer: measured against the origin spelling
    # forever, Chrome's own canonicalization of the initial URL reads as a
    # redirect, and a genuine hop back to the start reads as none. Mutable
    # state is safe here -- zendriver dispatches every event for one connection
    # on a single thread.
    position = [origin_url]
    followed = [0]

    def on_paused(event: object, *_unused: object) -> None:
        # Two-arg tolerant and isinstance-guarded for the same reasons as
        # ``_main_frame_navigations``: zendriver retries a one-arg callback
        # through an exception path, and dispatches by ``type(event)``.
        if not isinstance(event, zendriver.cdp.fetch.RequestPaused):
            return
        target = event.request.url
        # Interception is scoped to documents (see the pattern below), so every
        # event here is a navigation; only one that LEAVES the current document
        # is a redirect the caller should hear about.
        is_hop = _wire_url(target) != _wire_url(position[0])
        position[0] = target
        if is_hop:
            followed[0] += 1
            if followed[0] > max_redirects:
                # Refused exactly like a forbidden host: Chrome reports it as a
                # navigation error, which is the honest outcome for a hop the
                # caller's budget does not cover.
                logger.debug(
                    "redirect budget of %d exhausted at %r",
                    max_redirects,
                    target,
                )
                _dispatch(loop, tab, _fail(event.request_id))
                return
        try:
            pinned_host(target, trust)
            if is_hop and on_redirect is not None:
                on_redirect(target)
        except Exception:  # noqa: BLE001 -- any refusal aborts the request.
            # ``on_redirect`` is documented as "raise to abort", so its
            # exception is a decision, not a fault, and is handled identically
            # to a failed host validation.
            _dispatch(loop, tab, _fail(event.request_id))
            return
        _dispatch(
            loop,
            tab,
            _continue(
                event.request_id,
                _carried_headers(event.request.headers, headers, target, origin_url),
            ),
        )

    event_type = zendriver.cdp.fetch.RequestPaused
    register = cast(
        Callable[[type[object], Callable[..., None]], None],
        tab.add_handler,
    )
    register(event_type, on_paused)
    # DOCUMENT requests only. Intercepting everything pauses each subresource
    # until this handler answers, and the answer costs a DNS resolution on
    # zendriver's connection thread -- measured as a page that never finished
    # loading (Google's live fetch timed out waiting for readyState). Documents
    # are also the whole SSRF surface: a redirect chain is documents, and a
    # subresource cannot redirect the NAVIGATION anywhere.
    pattern = zendriver.cdp.fetch.RequestPattern(
        url_pattern="*",
        resource_type=zendriver.cdp.network.ResourceType.DOCUMENT,
        request_stage=zendriver.cdp.fetch.RequestStage.REQUEST,
    )
    await tab.send(zendriver.cdp.fetch.enable(patterns=[pattern]))


# Compares what the WIRE carries, so the differences Chrome introduces on its way there
# are not mistaken for a redirect:
#
# - The fragment never leaves the client, so ``/page#a`` requests ``/page``. - An empty
# path is canonicalized to ``/``, so ``https://host`` requests ``https://host/``.
#
# Both were measured reporting a spurious first hop, and ``on_redirect`` is raise-to-
# abort -- Google's raises on ``/sorry`` -- so a false hop aborts an ordinary fetch.
def _wire_url(url: str) -> str:
    """Return ``url`` as it goes on the wire: no fragment, path never empty."""
    parts = urlsplit(url)
    return urlunsplit(parts._replace(fragment="", path=parts.path or "/"))


def _fail(request_id: object) -> object:
    """Return the CDP verb refusing one intercepted request."""
    return zendriver.cdp.fetch.fail_request(
        cast("zendriver.cdp.fetch.RequestId", request_id),
        zendriver.cdp.network.ErrorReason.ACCESS_DENIED,
    )


# ``headers=None`` sends no override, which is NOT the same as echoing the request's own
# headers back. An override is unreliable in two documented ways, and both bite a
# challenge handoff:
#
# - It is applied INTERMITTENTLY to ``Cookie`` (crbug 40762053: "setting the cookie
# header override is intermittent -- only 3 of 21 requests ... have the cookie
# override"). A clearance cookie that rides only sometimes reads as a client that never
# solved the challenge. - It does "not extend to subsequent redirect hops" (CDP
# ``Fetch`` docs), and the clear IS a redirect chain -- measured as GET, POST, GET.
#
# So an echo is strictly worse than silence: it can only subtract. Measured on one live
# Cloudflare-fronted URL, interleaved against the no-override control on a fresh egress,
# the echo served the wall every run and the omission served the page every run.
# headers.
def _continue(
    request_id: object,
    headers: list[zendriver.cdp.fetch.HeaderEntry] | None,
) -> object:
    """Return the CDP verb releasing one intercepted request, optionally overriding."""
    typed_request_id = cast("zendriver.cdp.fetch.RequestId", request_id)
    if headers is None:
        return zendriver.cdp.fetch.continue_request(typed_request_id)
    return zendriver.cdp.fetch.continue_request(
        typed_request_id,
        headers=headers,
    )


# An origin-bound header is never installed tab-wide (see :func:`_navigate_tab`), so it
# must be attached HERE, and only for a hop that stays on the origin it was seeded for.
#
# Adding nothing yields ``None`` rather than an echo of ``request_headers``: an override
# is applied unreliably and does not survive a redirect hop (see :func:`_continue`), so
# echoing headers back can only lose them. The withholding case is ``None`` too -- a
# cross-origin hop never carried the credential, since it was never installed tab-wide,
# so there is nothing here to remove.
#
# An override that DOES fire is therefore reserved for the one case worth its
# unreliability: a caller credential the tab cannot hold.
def _carried_headers(
    request_headers: Mapping[str, str],
    caller_headers: dict[str, str] | None,
    target: str,
    origin_url: str,
) -> list[zendriver.cdp.fetch.HeaderEntry] | None:
    """Return the header OVERRIDE for this hop, or ``None`` to send none."""
    caller = caller_headers or {}
    if not caller or origin(target) != origin(origin_url):
        return None
    bound = _origin_bound()
    entitled = {name: value for name, value in caller.items() if name.lower() in bound}
    if not entitled:
        return None
    # Merged case-INSENSITIVELY, unlike a plain dict union: field names are
    # case-insensitive on the wire but not as dict keys, and Chrome reports the
    # paused request Title-Cased, so a caller's lower-case "cookie" beside
    # Chrome's "Cookie" emitted the header twice. ``fetch.py`` collapses the
    # same collision on the curl leg for the same reason -- two Cookie lines
    # are a bot tell. The caller's value wins, matching that leg's
    # ``set_session_cookies``.
    replaced = {name.lower() for name in entitled}
    kept = {
        name: value
        for name, value in request_headers.items()
        if name.lower() not in replaced
    }
    return [
        zendriver.cdp.fetch.HeaderEntry(name=name, value=value)
        for name, value in (kept | entitled).items()
    ]


# Read off :func:`~wesearch.fetch.common.apply_redirect`'s own default rather than
# restated: Chrome follows its own redirects, so this transport applies the rule itself
# instead of through that function, and a second copy is how the two legs came to
# disagree.
def _origin_bound() -> frozenset[str]:
    """Return the header names a cross-origin hop may not carry."""
    default = cast(
        object,
        inspect.signature(apply_redirect).parameters["origin_bound"].default,
    )
    assert isinstance(default, frozenset)
    return cast(frozenset[str], default)


# The handler is invoked by zendriver's callback machinery, which is not a coroutine
# context, so the command is scheduled on the tab's own loop rather than awaited here. A
# handler that blocked on the send would deadlock the connection it is trying to answer.
def _dispatch(
    loop: asyncio.AbstractEventLoop,
    tab: zendriver.Tab,
    command: object,
) -> None:
    """Send a CDP command from the synchronous event-handler thread."""
    coroutine = tab.send(
        cast("Generator[dict[str, object], dict[str, object], object]", command),
    )
    if loop.is_closed():
        coroutine.close()  # Nothing left to answer; do not warn on a stray task.
        return
    # ``call_soon_threadsafe``, not ``create_task``: the callback runs on
    # zendriver's connection thread, and creating a task on another thread's
    # loop is not safe.
    loop.call_soon_threadsafe(lambda: loop.create_task(coroutine))


# Chrome fires ``FrameNavigated`` for every frame, and a challenge page is dense with
# sub-frames (the Turnstile widget alone accounts for most of the 17 events one
# interstitial emits). Only the main frame -- the one with no parent -- means "the
# document you are reading was replaced".
def _main_frame_navigations(tab: zendriver.Tab) -> asyncio.Event:
    """Return an Event set whenever the MAIN frame commits a new document."""
    navigated = asyncio.Event()
    # Captured here: the handler below runs on a thread with no running loop.
    loop = asyncio.get_running_loop()

    # Two-arg tolerant on purpose: zendriver calls a handler as
    # ``callback(event, connection)`` and retries as ``callback(event)`` only
    # after catching TypeError. A one-arg signature reaches the handler through
    # that exception path, where a TypeError raised INSIDE the handler is
    # indistinguishable from the arity mismatch and silently re-runs it.
    #
    # ``isinstance`` rather than a bare attribute read: zendriver dispatches on
    # ``type(event)`` (connection.py), yet a live run delivered a
    # ``FrameStartedLoading`` here and the handler raised AttributeError inside
    # zendriver's callback thread. That exception cannot fail the fetch -- it is
    # logged and swallowed -- so the cost is a silent miss of the wakeup this
    # exists to deliver, not a crash. Guarding the shape is cheap; the event
    # this cares about is the one with a frame.
    def on_navigated(event: object, *_unused: object) -> None:
        if (
            isinstance(event, zendriver.cdp.page.FrameNavigated)
            and event.frame.parent_id is None
        ):
            # Not a bare ``set()``: zendriver runs this handler off-loop, and a
            # cross-thread set flips the flag without waking the selector.
            loop.call_soon_threadsafe(navigated.set)

    # ``zendriver`` annotates this parameter as a bare ``Callable`` -- i.e.
    # ``Callable[..., Unknown]`` -- under a suppression of its own in
    # connection.py, so the bound method is partially unknown before an argument
    # is even passed. A stub cannot repair it in place: ``add_handler`` is
    # inherited from ``Connection``, and a partial ``.pyi`` for that class would
    # blank its other 66 members. Naming the real contract at this one call site
    # is the narrowest fix, and it keeps ``event.frame.parent_id`` checked above.
    event_type = zendriver.cdp.page.FrameNavigated
    register = cast(
        Callable[[type[object], Callable[..., None]], None],
        tab.add_handler,
    )
    register(event_type, on_navigated)
    return navigated


# A single load event is not the end of a challenge-walled fetch. The interstitial is
# itself a complete document: it reaches ``readyState == "complete"``, THEN its JS
# navigates the tab to the real page. Measured on one live URL::
#
# t=0.00s 5516 bytes challenge readyState=complete <- interstitial t=1.10s 386 bytes
# clear readyState=loading <- real doc parsing t=2.20s 380404 bytes clear
# readyState=complete <- the page
#
# Both intermediate states are traps. Harvesting at the first ``complete`` returns the
# wall; harvesting the moment the challenge markup disappears returns a 386-byte
# ``<head>`` whose title looks right and whose body is empty. So each iteration waits
# for a real main-frame navigation and THEN for that new document to finish parsing --
# never for a duration.
#
# This is event-driven rather than polled deliberately: Chrome already knows when it
# replaced the document, so sampling the DOM on a timer both guesses at an interval and
# can only ever observe the states its grid lands on (the 386-byte phase is exactly such
# a miss). One wait per real transition, no sampling rate to tune.
#
# ``on_success_body=True`` is load-bearing: this body came from a browser that rendered
# the page, so a generic CAPTCHA widget in it is ordinary furniture (a login form's
# reCAPTCHA), not proof of a wall. Only structural interstitial evidence means "this
# document is about to replace itself".
#
# ``budget_sec`` bounds a challenge that never clears -- a real block rather than a
# delay -- and must stay under the caller's overall fetch timeout, so a walled page
# surfaces the wall instead of raising ``TimeoutError``. The last body read is returned
# for the caller's classifier to judge; this layer decides only WHEN the page stopped
# changing, never what it means.
async def _settled_content(tab: zendriver.Tab, *, budget_sec: float) -> str:
    """Return the tab's HTML once it is no longer a challenge interstitial."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget_sec
    navigated = _main_frame_navigations(tab)
    while True:
        # Cleared BEFORE the read, never after: a navigation that commits
        # between reading the body and starting the wait must still count. Clear
        # afterwards and that wakeup is dropped, so a page that cleared in the
        # gap blocks for the whole budget -- the classic lost-wakeup.
        navigated.clear()
        body = await tab.get_content()
        if classify_challenge(body, on_success_body=True) is None:
            return body
        remaining = deadline - loop.time()
        if remaining <= 0:
            return body
        try:
            await asyncio.wait_for(navigated.wait(), timeout=remaining)
            # The navigation only COMMITS the new document; its markup arrives as
            # it parses. Without this the next read catches the half-built
            # 386-byte phase above -- right title, empty body.
            #
            # Bounded by the SAME deadline as the wait above, not left open: a
            # document that commits just before the deadline and then stalls
            # would otherwise park here with no ceiling of its own, spending the
            # caller's entire fetch timeout. That is precisely what ``budget_sec``
            # promises not to do -- give up in time to return the last body and
            # let the caller classify the wall.
            await asyncio.wait_for(
                tab.wait_for_ready_state("complete"),
                timeout=max(deadline - loop.time(), 0.0),
            )
        except TimeoutError:
            return await tab.get_content()


# Each fetch runs in its OWN tab that is CLOSED when the fetch returns. The Chrome
# process stays warm and shared (fast reuse), but the tab -- which holds the scraped
# page's DOM, JS heap, and images -- is the unit of memory teardown, so a sequence of
# fetches does not accumulate resident pages. A per-fetch tab also isolates concurrent
# fetches sharing the one browser.
#
# Readiness is Chrome's real load signal (``document.readyState == "complete"``) plus
# :func:`_settled_content` for the interstitial that outlives it, bounded by
# ``timeout_sec``. The transport returns what Chrome rendered without assigning provider
# semantics to it.
async def _navigate(
    url: str,
    *,
    profile_dir: Path,
    timeout_sec: float,
    headless: bool,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    trust: Trust = "untrusted",
    max_redirects: int = 10,
    on_redirect: Callable[[str], None] | None = None,
) -> BrowserResult:
    """Drive the shared browser to ``url`` in a fresh tab; harvest body + cookies."""
    async with asyncio.timeout(timeout_sec):
        browser = await _pool().browser(profile_dir, headless=headless)
        if seeded := cookies or {}:
            await browser.cookies.set_all(
                [
                    zendriver.cdp.network.CookieParam(
                        name=name,
                        value=value,
                        url=url,
                    )
                    for name, value in seeded.items()
                ],
            )
        tab = await _navigate_tab(
            browser,
            url,
            headers=headers,
            trust=trust,
            max_redirects=max_redirects,
            on_redirect=on_redirect,
        )
        try:
            # Half the overall budget: the settle poll must be able to give up
            # and still leave time to harvest cookies and return the wall, so a
            # blocked page surfaces its BotDetectionError instead of a timeout.
            body = await _settled_content(tab, budget_sec=timeout_sec / 2)
            final_url = cast(str, await tab.evaluate("document.location.href")) or url
            # Cookies are browser-wide (shared jar), so harvest before closing the
            # tab; the closed tab's cookies persist in the profile regardless.
            #
            # Keyed on the FINAL url, not the requested one: a cross-origin
            # redirect seats the target's cookies, and filtering by the source
            # host dropped exactly the cookies a following fetch to the target
            # needs. ``on_redirect`` already fired per hop in the guard, before
            # each was followed.
            harvested = await _domain_cookies(browser, final_url)
        finally:
            await _closed(tab)
    return BrowserResult(
        body=_unwrap_viewer(body).encode(),
        cookies=harvested,
        final_url=final_url,
    )


# ``get_content`` serializes the DOM, and a non-HTML response has no DOM of its own --
# Chrome SYNTHESIZES one to display it, re-emitting the bytes inside ``<pre>`` under a
# generated ``<head>``. A caller that asked a JSON endpoint for JSON would otherwise
# receive markup wrapped around valid data and fail to parse it, reporting a malformed
# response the server never sent.
#
# Matched on the synthesized shell specifically, not on "contains a ``<pre>``": a real
# HTML page carrying a code sample must come back whole.
def _unwrap_viewer(body: str) -> str:
    """Return the original payload when Chrome wrapped it in its viewer shell."""
    match = _VIEWER_SHELL.match(body.strip())
    if match is None:
        return body
    return unescape(match.group("payload"))


async def _domain_cookies(browser: zendriver.Browser, url: str) -> dict[str, str]:
    """Return the browser's cookies whose domain matches ``url``'s host."""
    host = urlparse(url).hostname or ""
    jar: dict[str, str] = {}
    for cookie in await browser.cookies.get_all():
        domain = cookie.domain.removeprefix(".")
        if domain and (host == domain or host.endswith(f".{domain}")):
            jar[cookie.name] = cookie.value or ""
    return jar


# Registers disconnect WITH the pool that needs it, on the one path that can create one,
# so a process closes its websocket before the interpreter tears the loop thread down.
# The shared Chrome is deliberately NOT stopped here: one Chrome per profile serves the
# whole host, which is what bounds the browser count -- a Chrome per process once left
# 378 processes holding 27.5 GiB.
def _pool() -> _BrowserPool:
    """Return the process-wide browser pool, creating it once."""
    global _pool_singleton  # noqa: PLW0603 -- memoize the shared pool.
    with _pool_lock:
        if _pool_singleton is None:
            _pool_singleton = _BrowserPool()
            atexit.register(shutdown_browsers)
        return _pool_singleton


class _Flags(Protocol):
    """Typed command-line namespace for the executable entry point."""

    url: str


class _KillableProcess(Protocol):
    """The private process handle slice used during browser cleanup."""

    def kill(self) -> None: ...


async def _close_browser_on_port(port: int) -> None:
    browser = await zendriver.start(host="127.0.0.1", port=port)
    await browser.stop()


def _close_shared_browser(profile_dir: Path) -> None:
    """Close the shared Chrome on ``profile_dir``, if one is running."""
    port = _devtools_port(profile_dir)
    if port is None:
        return
    try:
        connection = socket.create_connection(("127.0.0.1", port), timeout=0.2)
    except OSError:
        return
    connection.close()
    asyncio.run(_close_browser_on_port(port))
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            connection = socket.create_connection(("127.0.0.1", port), timeout=0.1)
        except OSError:
            return
        connection.close()
        time.sleep(0.05)
    # A capability condition, not a fetch fault: it blocks every later browser
    # fetch, so it must be the error such callers already catch.
    raise BrowserUnavailableError(f"Chrome on DevTools port {port} did not close.")


# Fetches never use this: they attach to the detached shared Chrome (see
# :func:`_shared_browser`). The operator's window is owned so the CLI can block on it,
# and while it is open, fetches attach to it instead.
async def _launch_browser(
    profile_dir: Path,
    *,
    headless: bool,
) -> zendriver.Browser:
    """Launch a Chrome THIS process owns on the profile (the headed operator window)."""
    profile_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 -- The browser driver owns the blocking operation on its dedicated worker.
    _tolerate_late_cdp_replies()
    # Headed too, not just headless: an instance answering for
    # ``com.google.Chrome`` takes the operator's links either way, and one
    # rule is easier to keep true than a per-mode one.
    executable = await asyncio.to_thread(_fetch_browser)
    try:
        return await zendriver.start(
            zendriver.Config(
                headless=headless,
                user_data_dir=str(profile_dir),
                # Sandboxed unless we are root.
                sandbox=os.geteuid() != 0,
                browser_executable_path=executable,
                browser_args=_fetch_browser_args(executable),
                # ``zendriver`` retries the DevTools connection ``max_tries`` times,
                # each bounded by ``timeout``; the product is the launch's dead
                # time when Chrome cannot connect. Healthy Chrome exposes
                # DevTools in ~0.3s, so 0.5s clears it with margin while 6 tries
                # cap a dead-browser launch at 3s -- a fast skip rather than the
                # default 10x1.0s=10s hang that stacked past live-test timeouts.
                browser_connection_timeout=0.5,
                browser_connection_max_tries=6,
            ),
        )
    except Exception as error:
        # ``zendriver`` raises a bare ``Exception`` when Chrome cannot start or the
        # DevTools connection never comes up. Re-raise it typed so callers can
        # tell "no usable browser here" apart from a fetch/parse failure.
        raise BrowserUnavailableError(f"Could not launch Chrome: {error}") from error


# Abandoning it instead leaks the whole Chrome tree: the process holds its profile's
# ``SingletonLock``, so a later launch on that profile fails with no usable diagnosis.
# Measured on a leaked tree: killing the root alone took all ten of its processes with
# it, because Chrome's zygotes and renderers die with the browser -- so signalling the
# root is sufficient and no process-group handling is needed.
#
# Signals through zendriver's ``Popen`` handle rather than the raw pid. ``Popen.kill``
# polls first and skips a process it has already reaped, which is what makes this safe
# against the kernel recycling that pid onto an unrelated process while a wedged
# ``stop()`` was still in flight.
#
# Cleanup reports nothing over its caller, so every failure is swallowed: this runs with
# the real error in flight.
def _kill_browser_process(browser: zendriver.Browser) -> None:
    """SIGKILL a browser whose cooperative ``stop()`` did not complete."""
    process: object = getattr(
        browser,
        "_process",
        None,
    )
    if process is None:
        return  # Never launched by us, or already cleared by a completed stop.
    with contextlib.suppress(OSError):
        cast(_KillableProcess, process).kill()


# Chrome sends its own coherent identity; a caller's curl-era impersonation overrides it
# tab-wide and makes Chrome contradict itself. Measured on Google: Chrome's own headers
# got the 553 KB results page with 9 hits; the same tab with the Android GSA User-Agent
# got the 61 KB enablejs shell with none. Credentials and API keys still pass.
def _without_browser_identity(
    headers: Mapping[str, str] | None,
    *,
    owned: frozenset[str] = frozenset(
        {
            "user-agent",
            "accept",
            "accept-language",
            "accept-encoding",
            "referer",
            "upgrade-insecure-requests",
            "priority",
            "x-client-data",
        },
    ),
    owned_prefixes: tuple[str, ...] = ("sec-", "x-browser-"),
) -> dict[str, str] | None:
    """Drop every header Chrome sends itself, keeping only what the caller adds."""
    if headers is None:
        return None
    kept = {
        name: value
        for name, value in headers.items()
        if name.lower() not in owned and not name.lower().startswith(owned_prefixes)
    }
    return kept or None


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
