#!/bin/sh
# ruff: noqa: EXE003, D300 -- Polyglot shell/Python script.
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

Run ``fetch-zendriver URL`` to open that URL in a HEADED Chrome on the same
profile -- to debug a fetch that errored, or to seat a login.
'''
# fmt: on

from __future__ import annotations

from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from html import unescape
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar, cast, override
from urllib.parse import urlparse

import asyncio
import atexit
import hashlib
import logging
import os
import re
import socket
import socketserver
import sys
import tempfile
import threading
import time
import warnings

from wesearch.fetch.challenge import classify_challenge
from wesearch.fetch.common import pinned_host
from wesearch.lib.userdirs import data_dir
from wesearch.ratelimit import clear_domain_cooldowns
from wesearch.types.params import Trust


if TYPE_CHECKING:
    import zendriver
else:
    from wrapt import lazy_import

    # Deferred: importing zendriver pulls a large CDP-binding tree (~200ms,
    # measured) and is paid only when a browser fetch actually runs, never at
    # ``wesearch`` import.
    zendriver = lazy_import("zendriver")


__all__ = [
    "BrowserResult",
    "BrowserUnavailableError",
    "fetch_zendriver",
    "open_instance",
    "shutdown_browsers",
]

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class BrowserUnavailableError(RuntimeError):
    """Chrome could not be launched or connected to on this host.

    A capability condition (Chrome absent, incompatible, or unable to bind its
    DevTools port), distinct from a fetch or parse failure: callers that need a
    browser cannot proceed, and environments without a usable one (CI, headless
    boxes) should treat it as "browser subsystem unavailable" rather than a bug.
    """


class _PoolControlServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, profile_dir: Path, release: Callable[[], None]) -> None:
        self.release = release
        address = _control_address(profile_dir)
        self._control_path = None if address.startswith("\0") else Path(address)
        super().__init__(address, _PoolControlHandler)
        self._thread = threading.Thread(
            target=lambda: self.serve_forever(poll_interval=0.01),
            name="loop-web-browser-control",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self.shutdown()
        self.server_close()
        self._thread.join()
        if self._control_path is not None:
            self._control_path.unlink(missing_ok=True)


class _PoolControlHandler(socketserver.StreamRequestHandler):
    @override
    def handle(self) -> None:
        if self.rfile.readline(64) != b"release\n":
            return
        cast(_PoolControlServer, self.server).release()


def _control_address(profile_dir: Path, platform: str = sys.platform) -> str:
    """Return the Unix-socket address coordinating one profile."""
    digest = hashlib.sha256(str(profile_dir.resolve()).encode()).hexdigest()[:24]
    if platform == "linux":
        return f"\0loop-zendriver-{digest}"
    return str(Path(tempfile.gettempdir()) / f"loop-zd-{digest}.sock")


def _request_pool_release(profile_dir: Path) -> None:
    """Ask another process's browser pool to release ``profile_dir``."""
    address = _control_address(profile_dir)
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(10)
    try:
        client.connect(address)
        client.sendall(b"release\n")
        # EOF is the acknowledgement: the handler closes only after the release
        # callback returns, including graceful browser and loop shutdown.
        if client.recv(64) != b"":
            raise RuntimeError("Zendriver browser pool returned an invalid response.")
    except ConnectionRefusedError:
        if not address.startswith("\0"):
            Path(address).unlink(missing_ok=True)
    except FileNotFoundError:
        pass
    finally:
        client.close()
    # Another process can leave a stale control listener that acknowledges this
    # profile without owning its Chrome. Verify and close the actual owner.
    _close_orphan_browser(profile_dir)


def _close_orphan_browser(profile_dir: Path) -> None:
    """Close a live Chrome whose owning pool no longer serves control."""
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


def _devtools_port(profile_dir: Path, *, proc_root: Path = Path("/proc")) -> int | None:
    """Read the verified profile owner's active DevTools port."""
    try:
        owner = (profile_dir / "SingletonLock").readlink()
        pid = int(str(owner).rsplit("-", 1)[1])
        command = (
            (proc_root / str(pid) / "cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode()
        )
    except (FileNotFoundError, IndexError, OSError, UnicodeError, ValueError):
        return None
    if _command_flag(command, "--user-data-dir=") != str(profile_dir.resolve()):
        return None
    try:
        port_text = _command_flag(command, "--remote-debugging-port=") or ""
        port = int(port_text.split(maxsplit=1)[0])
        if port:
            return port
        return int((profile_dir / "DevToolsActivePort").read_text().splitlines()[0])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def _command_flag(command: str, marker: str) -> str | None:
    """Extract a Chrome flag value from NUL- or space-flattened proc args."""
    _, found, suffix = command.partition(marker)
    if not found:
        return None
    return suffix.split(" --", 1)[0].strip()


async def _close_browser_on_port(port: int) -> None:
    browser = await zendriver.start(host="127.0.0.1", port=port)
    await browser.stop()


def _sandbox() -> bool:
    """Whether to run Chrome sandboxed (yes, unless we are root).

    Chrome's setuid sandbox refuses to start as root, so a root context (CI,
    containers) must pass ``--no-sandbox``. A normal desktop user keeps the
    sandbox -- disabling it there needlessly weakens security AND makes Chrome
    show a persistent "unsupported command-line flag: --no-sandbox" banner.
    """
    return os.geteuid() != 0


class BrowserResult(NamedTuple):
    """What a browser fetch yields: the rendered page and the cookies it holds.

    Attributes:
      body: The rendered ``document`` HTML, UTF-8 encoded.
      cookies: Cookies the browser holds for the fetched URL's domain
        (``name -> value``), for the caller to persist and thread onward.

    """

    body: bytes
    cookies: dict[str, str]


def fetch_zendriver(
    url: str,
    *,
    profile_dir: Path,
    egress: str,
    timeout_sec: float = 30.0,
    headless: bool = True,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    trust: Trust = "untrusted",
    on_redirect: Callable[[str], None] | None = None,
) -> BrowserResult:
    """Fetch ``url`` in a pooled headless Chrome; return its body and cookies.

    Navigates a warm browser (one per ``(egress, profile_dir)``) to ``url``,
    waits for the load to complete, then returns the rendered HTML and the
    cookies the browser acquired for the URL's domain. Page-state validation
    belongs to the provider consuming the rendered response.

    Args:
      url: Fully-qualified URL to navigate to.
      profile_dir: Chrome ``user_data_dir`` supplying the logged-in identity.
      egress: Public egress IP the pooled browser is keyed to (a rotation keys
        a fresh browser).
      timeout_sec: Overall budget for navigation + load, in seconds.
      headless: Run Chrome headless (the default); ``False`` opens a window.
      headers: Extra headers applied to this tab before navigation.
      cookies: Cookies seeded for ``url`` before navigation.
      trust: Provenance of the URL. Under ``"untrusted"`` every DOCUMENT request
        -- the navigation and each redirect hop Chrome follows itself -- is
        validated to a public address before the socket opens. Subresources are
        deliberately not intercepted; see :func:`_guard_requests`. Required
        rather than defaulted, because a transport that cannot express the
        policy cannot be held to it.
      on_redirect: Called with each document hop before it is followed; raise to
        abort it.

    Returns:
      result: The rendered body and the browser's cookies for the URL's domain.

    """
    return _pool().run(
        _navigate(
            url,
            profile_dir=profile_dir,
            egress=egress,
            timeout_sec=timeout_sec,
            headless=headless,
            headers=headers,
            cookies=cookies,
            trust=trust,
            on_redirect=on_redirect,
        ),
        # Above the coroutine's own budget, so a normal timeout still raises
        # from inside ``_navigate``, which closes its tab and reports the wall.
        timeout_sec=timeout_sec + 30,
    )


def shutdown_browsers() -> None:
    """Close every pooled browser and stop the pool's loop thread.

    Idempotent, and a NO-OP when no pool has been created -- it must never
    construct one just to tear it down (that would spin up a Chrome-driving loop
    thread only to stop it, and on an egress rotation before any browser fetch it
    would poison the not-yet-used singleton). Only an existing pool is shut down;
    the singleton is then cleared so the next browser fetch builds a fresh pool.
    """
    global _pool_singleton  # noqa: PLW0603 -- reset the shared pool after teardown.
    with _pool_lock:
        pool = _pool_singleton
        _pool_singleton = None
    if pool is not None:
        pool.shutdown()


def open_instance(url: str, *, profile_dir: Path | None = None) -> None:
    """Open a HEADED Chrome on the profile dir at ``url``; block until closed.

    Launches a visible Chrome under ``profile_dir`` (the fresh dedicated dir by
    default), navigates to ``url``, and blocks until the user closes the window.
    Use it to eyeball a URL the headless :func:`fetch_zendriver` backend failed on
    -- you see exactly what Chrome renders (a challenge, a login wall, a broken
    page) under the SAME profile the backend uses, and any cookies you seat while
    there (e.g. by logging in) persist for later headless fetches. Runs OUTSIDE
    the pool (a one-shot headed browser owned by this call).

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
    _request_pool_release(target)
    _pool().run(_open_instance(url, target))
    domain = urlparse(url).hostname
    if domain is not None:
        clear_domain_cooldowns(domain)


async def _open_instance(url: str, profile_dir: Path) -> None:
    """Open a headed browser, navigate to ``url``, and block until it is closed."""
    browser = await _launch_browser(profile_dir, headless=False)
    try:
        await _navigate_tab(browser, url)
        # Block until the user closes the window (Chrome exits, so the browser reports
        # stopped). Polled, not event-driven: the window-closed signal is Chrome's
        # process exit, which zendriver exposes only as the polled ``stopped`` flag.
        while not browser.stopped:  # noqa: ASYNC110 -- no event source; poll the flag.
            await asyncio.sleep(0.5)
    except BaseException:
        await browser.stop()
        raise


async def _launch_browser(
    profile_dir: Path,
    *,
    headless: bool,
) -> zendriver.Browser:
    """Launch vanilla Chrome on the persistent profile."""
    profile_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 -- one-shot setup.
    try:
        return await zendriver.start(
            zendriver.Config(
                headless=headless,
                user_data_dir=str(profile_dir),
                sandbox=_sandbox(),
                # zendriver retries the DevTools connection ``max_tries`` times,
                # each bounded by ``timeout``; the product is the launch's dead
                # time when Chrome cannot connect. Healthy Chrome exposes
                # DevTools in ~0.3s, so 0.5s clears it with margin while 6 tries
                # cap a dead-browser launch at 3s -- a fast skip rather than the
                # default 10x1.0s=10s hang that stacked past live-test timeouts.
                browser_connection_timeout=0.5,
                browser_connection_max_tries=6,
            )
        )
    except Exception as error:
        # zendriver raises a bare ``Exception`` when Chrome cannot start or the
        # DevTools connection never comes up. Re-raise it typed so callers can
        # tell "no usable browser here" apart from a fetch/parse failure.
        raise BrowserUnavailableError(f"Could not launch Chrome: {error}") from error


async def _navigate_tab(
    browser: zendriver.Browser,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    trust: Trust = "untrusted",
    on_redirect: Callable[[str], None] | None = None,
) -> zendriver.Tab:
    """Open a blank tab, arm the per-request guard, apply headers, navigate."""
    tab = await browser.get("about:blank", new_tab=True)
    try:
        await _guard_requests(tab, url, trust=trust, on_redirect=on_redirect)
        if headers:
            await tab.send(zendriver.cdp.network.enable())
            await tab.send(
                zendriver.cdp.network.set_extra_http_headers(
                    zendriver.cdp.network.Headers(headers)
                )
            )
        await tab.get(url)
        await tab.wait_for_ready_state("complete")
    except BaseException:
        await tab.close()
        raise
    return tab


async def _guard_requests(
    tab: zendriver.Tab,
    url: str,
    *,
    trust: Trust,
    on_redirect: Callable[[str], None] | None,
) -> None:
    """Validate every DOCUMENT request this tab makes BEFORE Chrome connects.

    The header transports re-validate each redirect hop (``curl.py`` and
    ``stdlib.py`` both call :func:`pinned_host` per hop) because
    :func:`wesearch.fetch.common.pinned_host` states the rule: a redirect
    target is a URL like any other, and skipping the re-check is the classic
    SSRF bypass. Chrome follows redirects ITSELF, so validating only the URL the
    caller passed left every subsequent hop -- and every subresource -- reaching
    the network with nothing watching. A public URL redirecting to
    ``169.254.169.254`` or loopback was fetched, and the transport learned of it
    only after the response had already been read.

    ``Fetch.requestPaused`` is the one seam that runs before the connection, so
    the same ``pinned_host`` that guards curl guards Chrome, and ``on_redirect``
    regains the pre-follow abort :class:`~wesearch.types.params.ObserveParams`
    documents. A rejected request is failed with ``AccessDenied`` rather than
    silently continued: Chrome surfaces that as a navigation error, which is the
    honest outcome for a target policy forbids.
    """
    origin_url = url
    # Captured here, where a running loop is guaranteed: zendriver invokes the
    # handler from its own connection thread, which has no running loop of its
    # own, so a ``get_running_loop`` inside the callback raises.
    loop = asyncio.get_running_loop()

    def on_paused(event: object, *_unused: object) -> None:
        # Two-arg tolerant and isinstance-guarded for the same reasons as
        # ``_main_frame_navigations``: zendriver retries a one-arg callback
        # through an exception path, and dispatches by ``type(event)``.
        if not isinstance(event, zendriver.cdp.fetch.RequestPaused):
            return
        target = event.request.url
        # Interception is scoped to documents (see the pattern below), so every
        # event here is a navigation; only a CHANGE of URL is a redirect the
        # caller should hear about.
        is_hop = target != origin_url
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
        _dispatch(loop, tab, _continue(event.request_id))

    event_type = zendriver.cdp.fetch.RequestPaused
    register = cast(
        "Callable[[type[object], Callable[..., None]], None]", tab.add_handler
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


def _fail(request_id: object) -> object:
    """The CDP verb refusing one intercepted request."""
    return zendriver.cdp.fetch.fail_request(
        cast("Any", request_id), zendriver.cdp.network.ErrorReason.ACCESS_DENIED
    )


def _continue(request_id: object) -> object:
    """The CDP verb releasing one intercepted request."""
    return zendriver.cdp.fetch.continue_request(cast("Any", request_id))


def _dispatch(
    loop: asyncio.AbstractEventLoop, tab: zendriver.Tab, command: object
) -> None:
    """Send a CDP command from the synchronous event-handler thread.

    The handler is invoked by zendriver's callback machinery, which is not a
    coroutine context, so the command is scheduled on the tab's own loop rather
    than awaited here. A handler that blocked on the send would deadlock the
    connection it is trying to answer.
    """
    coroutine = tab.send(cast("Any", command))
    if loop.is_closed():
        coroutine.close()  # Nothing left to answer; do not warn on a stray task.
        return
    # ``call_soon_threadsafe``, not ``create_task``: the callback runs on
    # zendriver's connection thread, and creating a task on another thread's
    # loop is not safe.
    loop.call_soon_threadsafe(lambda: loop.create_task(coroutine))


def _main_frame_navigations(tab: zendriver.Tab) -> asyncio.Event:
    """Return an Event set whenever the MAIN frame commits a new document.

    Chrome fires ``FrameNavigated`` for every frame, and a challenge page is
    dense with sub-frames (the Turnstile widget alone accounts for most of the
    17 events one interstitial emits). Only the main frame -- the one with no
    parent -- means "the document you are reading was replaced".
    """
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

    # zendriver annotates this parameter as a bare ``Callable`` -- i.e.
    # ``Callable[..., Unknown]`` -- under a suppression of its own in
    # connection.py, so the bound method is partially unknown before an argument
    # is even passed. A stub cannot repair it in place: ``add_handler`` is
    # inherited from ``Connection``, and a partial ``.pyi`` for that class would
    # blank its other 66 members. Naming the real contract at this one call site
    # is the narrowest fix, and it keeps ``event.frame.parent_id`` checked above.
    event_type = zendriver.cdp.page.FrameNavigated
    register = cast(
        Callable[[type[object], Callable[..., None]], None], tab.add_handler
    )
    register(event_type, on_navigated)
    return navigated


async def _settled_content(tab: zendriver.Tab, *, budget_sec: float) -> str:
    """Return the tab's HTML once it is no longer a challenge interstitial.

    A single load event is not the end of a challenge-walled fetch. The
    interstitial is itself a complete document: it reaches ``readyState ==
    "complete"``, THEN its JS navigates the tab to the real page. Measured on
    one live URL::

        t=0.00s   5516 bytes  challenge   readyState=complete   <- interstitial
        t=1.10s    386 bytes  clear       readyState=loading    <- real doc parsing
        t=2.20s 380404 bytes  clear       readyState=complete   <- the page

    Both intermediate states are traps. Harvesting at the first ``complete``
    returns the wall; harvesting the moment the challenge markup disappears
    returns a 386-byte ``<head>`` whose title looks right and whose body is
    empty. So each iteration waits for a real main-frame navigation and THEN
    for that new document to finish parsing -- never for a duration.

    This is event-driven rather than polled deliberately: Chrome already knows
    when it replaced the document, so sampling the DOM on a timer both guesses
    at an interval and can only ever observe the states its grid lands on (the
    386-byte phase is exactly such a miss). One wait per real transition, no
    sampling rate to tune.

    ``on_success_body=True`` is load-bearing: this body came from a browser that
    rendered the page, so a generic CAPTCHA widget in it is ordinary furniture
    (a login form's reCAPTCHA), not proof of a wall. Only structural
    interstitial evidence means "this document is about to replace itself".

    ``budget_sec`` bounds a challenge that never clears -- a real block rather
    than a delay -- and must stay under the caller's overall fetch timeout, so a
    walled page surfaces the wall instead of raising ``TimeoutError``. The last
    body read is returned for the caller's classifier to judge; this layer
    decides only WHEN the page stopped changing, never what it means.
    """
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


async def _navigate(
    url: str,
    *,
    profile_dir: Path,
    egress: str,
    timeout_sec: float,
    headless: bool,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    trust: Trust = "untrusted",
    on_redirect: Callable[[str], None] | None = None,
) -> BrowserResult:
    """Drive a pooled browser to ``url`` in a fresh tab; harvest body + cookies.

    Each fetch runs in its OWN tab that is CLOSED when the fetch returns. The
    Chrome process stays warm in the pool (fast reuse), but the tab -- which
    holds the scraped page's DOM, JS heap, and images -- is the unit of memory
    teardown, so a sequence of fetches does not accumulate resident pages. A
    per-fetch tab also isolates concurrent fetches sharing the one browser.

    Readiness is Chrome's real load signal (``document.readyState ==
    "complete"``) plus :func:`_settled_content` for the interstitial that
    outlives it, bounded by ``timeout_sec``. The transport returns what Chrome
    rendered without assigning provider semantics to it.
    """
    async with asyncio.timeout(timeout_sec):
        browser = await _pool().browser(egress, profile_dir, headless=headless)
        if cookies:
            await browser.cookies.set_all(
                [
                    zendriver.cdp.network.CookieParam(
                        name=name,
                        value=value,
                        url=url,
                    )
                    for name, value in cookies.items()
                ]
            )
        tab = await _navigate_tab(
            browser, url, headers=headers, trust=trust, on_redirect=on_redirect
        )
        try:
            # Half the overall budget: the settle poll must be able to give up
            # and still leave time to harvest cookies and return the wall, so a
            # blocked page surfaces its BotDetectionError instead of a timeout.
            body = await _settled_content(tab, budget_sec=timeout_sec / 2)
            final_url = cast("str", await tab.evaluate("document.location.href")) or url
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
            await tab.close()
    return BrowserResult(body=_unwrap_viewer(body).encode(), cookies=harvested)


def _unwrap_viewer(body: str) -> str:
    """Return the original payload when Chrome wrapped it in its viewer shell.

    ``get_content`` serializes the DOM, and a non-HTML response has no DOM of
    its own -- Chrome SYNTHESIZES one to display it, re-emitting the bytes
    inside ``<pre>`` under a generated ``<head>``. A caller that asked a JSON
    endpoint for JSON would otherwise receive markup wrapped around valid data
    and fail to parse it, reporting a malformed response the server never sent.

    Matched on the synthesized shell specifically, not on "contains a ``<pre>``":
    a real HTML page carrying a code sample must come back whole.
    """
    match = _VIEWER_SHELL.match(body.strip())
    if match is None:
        return body
    return unescape(match.group("payload"))


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


async def _domain_cookies(browser: zendriver.Browser, url: str) -> dict[str, str]:
    """Return the browser's cookies whose domain matches ``url``'s host."""
    host = urlparse(url).hostname or ""
    jar: dict[str, str] = {}
    for cookie in await browser.cookies.get_all():
        domain = (cookie.domain or "").lstrip(".")
        if domain and (host == domain or host.endswith(f".{domain}")):
            jar[cookie.name] = cookie.value or ""
    return jar


# The single pooled browser manager, built once on first browser fetch. A
# deliberate module singleton: it owns a live loop thread and open Chrome
# processes -- shared runtime resources, not a tunable.
# config-globals: ignore -- live pool of open browsers + its loop thread.
_pool_singleton: _BrowserPool | None = None
_pool_lock = threading.Lock()  # config-globals: ignore -- guards the singleton.


def _pool() -> _BrowserPool:
    """Return the process-wide browser pool, creating it once.

    Registers teardown WITH the pool that needs it, on the one path that can
    create one. A browser is ~70 MB across ~17 processes and nothing else ever
    closes one, so a process that exits without this leaves every browser it
    opened resident: measured at 378 processes holding 27.5 GiB, 225 of them
    older than the session that spawned them.

    ``atexit`` and not a parent-death signal, because the exit that leaked was
    an ORDINARY one -- pytest finished normally. It fires here under plain
    pytest and under `xdist -n=2` (both measured), and can still drive the
    pool's daemon loop thread, which is alive until interpreter teardown.
    """
    global _pool_singleton  # noqa: PLW0603 -- memoize the shared pool.
    with _pool_lock:
        if _pool_singleton is None:
            _pool_singleton = _BrowserPool()
            atexit.register(shutdown_browsers)
        return _pool_singleton


class _BrowserPool:
    """Pooled headless browsers over one persistent event loop on a daemon thread.

    zendriver browsers bind to the loop running their coroutines, so a stable
    loop is a hard requirement for reuse across sync calls. This pool owns that
    loop on a background thread and dispatches every browser coroutine to it via
    :meth:`run`, keeping one warm :class:`zendriver.Browser` per
    ``(egress, profile_dir)`` key and rejecting incompatible launch modes.

    The hot spare justifies this lifecycle machinery: eight matched
    ``https://example.com/`` requests measured a 0.130-second median with one
    reused browser versus 4.196 seconds when launching Chrome per request, a
    32.28x speedup.
    """

    def __init__(self, *, serve_control: bool = True) -> None:
        self._loop = asyncio.new_event_loop()
        self._serve_control = serve_control
        self._controls: dict[str, _PoolControlServer] = {}
        self._browsers: dict[tuple[str, str], tuple[bool, zendriver.Browser]] = {}
        self._launch_lock = asyncio.Lock()
        self._lock = threading.Lock()
        # Started LAST, so every field _run_loop touches exists before it runs.
        self._thread = threading.Thread(
            target=self._run_loop, name="loop-web-browser", daemon=True
        )
        self._thread.start()

    def run(self, coro: Coroutine[Any, Any, _T], *, timeout_sec: float = 0) -> _T:
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

    async def browser(
        self,
        egress: str,
        profile_dir: Path,
        *,
        headless: bool,
    ) -> zendriver.Browser:
        """Return the warm browser for one egress and profile."""
        # Resolved, matching ``_control_address``: two spellings of one profile
        # are one user-data dir, and Chrome allows it a single owner.
        key = (egress, str(profile_dir.resolve()))  # noqa: ASYNC240 -- one stat
        async with self._launch_lock:
            control_key = _control_address(profile_dir)
            with self._lock:
                owns_profile = control_key in self._controls
            if self._serve_control and not owns_profile:
                await asyncio.to_thread(_request_pool_release, profile_dir)
            self._ensure_control(profile_dir)
            with self._lock:
                existing = self._browsers.get(key)
            if existing is not None and not existing[1].stopped:
                if existing[0] != headless:
                    raise RuntimeError(
                        "Cannot change Zendriver launch mode for a live profile."
                    )
                return existing[1]
            launched = await self._launch(profile_dir, headless=headless)
            with self._lock:
                self._browsers[key] = (headless, launched)
            return launched

    def shutdown(self) -> None:
        """Close every pooled browser and stop the loop thread (idempotent)."""
        if self._loop.is_closed():
            return
        with self._lock:
            browsers = [browser for _, browser in self._browsers.values()]
            controls = list(self._controls.values())
            self._browsers.clear()
            self._controls.clear()
        for browser in browsers:
            try:
                # Bounded: this runs from ``atexit``, where an unreachable
                # browser would otherwise hold the interpreter open forever.
                self.run(browser.stop(), timeout_sec=30)
            except Exception:
                logger.debug("browser stop failed during shutdown", exc_info=True)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()
        for control in controls:
            control.close()

    def _ensure_control(self, profile_dir: Path) -> None:
        """Serve graceful cross-process release requests for ``profile_dir``."""
        if not self._serve_control:
            return
        key = _control_address(profile_dir)
        with self._lock:
            if key in self._controls:
                return
            self._controls[key] = _PoolControlServer(profile_dir, shutdown_browsers)

    async def _launch(
        self,
        profile_dir: Path,
        *,
        headless: bool,
    ) -> zendriver.Browser:
        """Launch one Chrome under ``profile_dir`` on the pool's loop."""
        return await _launch_browser(profile_dir, headless=headless)

    def _run_loop(self) -> None:
        """Run the pool's loop until :meth:`shutdown`, muting zendriver's warnings.

        zendriver's CDP dispatch calls the deprecated ``asyncio.iscoroutinefunction``
        (connection.py) and leaves reader pipes for the GC to close, emitting a
        ``DeprecationWarning`` / ``ResourceWarning`` from inside the browser
        coroutine. Under a ``-W error`` caller (the repo's pytest turns every
        warning into an exception) that warning would raise INSIDE the awaited CDP
        handler and wedge the fetch. The pool loop thread runs only zendriver
        coroutines, so scoping the filter to this thread's ``run_forever`` mutes
        the upstream noise without hiding warnings from any caller's own code.
        """
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
    """Open a URL in a headed Chrome on the backend's profile; return exit code."""
    import argparse  # noqa: PLC0415 -- CLI-only import, off the library path.

    parser = argparse.ArgumentParser(
        prog="fetch-zendriver",
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
            "  fetch-zendriver https://the-site-that-failed.example/\n"
            "  fetch-zendriver https://login.example/  # seat a session cookie\n"
            "  fetch-zendriver # opens blank; navigate by hand"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "url",
        nargs="?",
        default="about:blank",
        help="The URL to open (typically the one whose headless fetch failed). "
        "Omit to open a blank page and navigate by hand.",
    )
    args = parser.parse_args()
    print(  # noqa: T201 -- CLI user feedback.
        f"Opening {args.url} in Chrome on "
        f"{data_dir() / 'rekursiv-ai' / 'wesearch' / 'fetch-zendriver'} -- "
        "close the window when done."
    )
    open_instance(args.url)
    print("Window closed.")  # noqa: T201 -- CLI feedback.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
