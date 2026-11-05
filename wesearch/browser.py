"""Interact with tabs in wesearch's persistent headed or headless Chrome.

Calls share the fetch session's BrowserPool. Tabs belong to Chrome, survive
calls and session disconnects, and close only through an explicit close action.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, cast

import asyncio
import base64
import binascii
import math
import sys

from treekle import from_plain

from wesearch.fetch.common import pinned_host
from wesearch.fetch.session import default_session
from wesearch.fetch.transport import zendriver
from wesearch.types.errors import FetchError


if TYPE_CHECKING:
    from zendriver import Tab, cdp
    from zendriver.core import keys
    from zendriver.core.element import Element

    from wesearch.fetch.types import FetchSession
else:
    from wrapt import lazy_import

    cdp = lazy_import("zendriver.cdp")
    keys = lazy_import("zendriver.core.keys")


__all__ = [
    "BrowserAction",
    "BrowserElement",
    "BrowserError",
    "BrowserMode",
    "BrowserResult",
    "BrowserTab",
    "browser",
]


class BrowserError(Exception):
    """A browser action failed without exposing form contents in its message."""


BrowserAction = Literal[
    "tabs",
    "open",
    "navigate",
    "inspect",
    "click",
    "fill",
    "press",
    "screenshot",
    "close",
    "mode",
]
BrowserMode = Literal["headed", "headless"]


@dataclass(frozen=True, slots=True, kw_only=True)
class BrowserTab:
    """A persistent Chrome page target, addressed by its stable target id."""

    tab_id: str
    url: str
    title: str


@dataclass(frozen=True, slots=True, kw_only=True)
class BrowserElement:
    """A visible control's CSS selector and label, without its field value."""

    selector: str
    tag: str
    role: str
    label: str
    kind: str


@dataclass(frozen=True, slots=True, kw_only=True)
class BrowserResult:
    """Tabs, a bounded page inspection, or a PNG screenshot."""

    tab: BrowserTab | None = None
    tabs: tuple[BrowserTab, ...] = ()
    text: str = ""
    elements: tuple[BrowserElement, ...] = ()
    truncated: bool = False
    screenshot: bytes = b""
    mode: BrowserMode | None = None
    restarted: bool = False


def browser(
    action: BrowserAction,
    *,
    tab_id: str = "",
    url: str = "",
    selector: str = "",
    text: str = "",
    key: str = "",
    mode: BrowserMode | None = None,
    restart: bool = False,
    max_chars: int = 20_000,
    timeout_sec: float = 30,
    session: FetchSession | None = None,
) -> BrowserResult:
    """Interact with shared Chrome; change its mode only with restart consent.

    Args:
      action: List tabs, open or navigate a page, inspect it, click or fill a
        control, press a key, capture a screenshot, close a tab, or control mode.
      tab_id: Stable target id; required except for tabs, open, and mode.
      url: Public HTTP(S) destination; required for open/navigate only.
      selector: CSS selector from inspection; required for click/fill/press.
      text: Replacement field contents for fill, including an empty string.
      key: Enter, Tab, Escape, Backspace, Delete, or an Arrow key for press.
      mode: Headed or headless for the mode action; omitted reports actual mode.
      restart: Permit restarting shared Chrome to change mode. Tabs are restored
        best-effort; target ids and unsaved page state are invalidated.
      max_chars: Maximum inspected text length; must be positive.
      timeout_sec: Entire action deadline, including attachment and page loading.
      session: Existing fetch session; omitted uses wesearch's shared session.

    Returns:
      result: Stable tab identity and inspected controls, tab listing, or PNG bytes.

    Raises:
      ValueError: Arguments do not describe a valid action.
      BrowserError: Browser, target, element, or deadline failure.
      FetchError: An HTTP navigation failed.

    """
    _validate(
        action,
        tab_id=tab_id,
        url=url,
        selector=selector,
        text=text,
        key=key,
        mode=mode,
        restart=restart,
        max_chars=max_chars,
        timeout_sec=timeout_sec,
    )
    if url:
        pinned_host(url, allow_private_hosts=False)
    session = session or default_session()
    pool = zendriver.get_browser_pool(session)
    try:
        return pool.run(
            _perform(
                pool,
                session,
                action,
                tab_id=tab_id,
                url=url,
                selector=selector,
                text=text,
                key=key,
                mode=mode,
                restart=restart,
                max_chars=max_chars,
                timeout_sec=timeout_sec,
            ),
            # Let the action's timeout finish interception cleanup before this waiter expires.
            timeout_sec=timeout_sec + 30,
        )
    except TimeoutError as error:
        raise BrowserError(
            f"Browser {action} timed out after {timeout_sec:g} seconds.",
        ) from error
    except (BrowserError, FetchError, ValueError):
        raise
    except zendriver.BrowserUnavailableError as error:
        raise BrowserError(str(error)) from error
    except Exception as error:
        raise BrowserError(
            f"Browser {action} failed ({type(error).__name__}).",
        ) from error


def _validate(
    action: BrowserAction,
    *,
    tab_id: str,
    url: str,
    selector: str,
    text: str,
    key: str,
    mode: BrowserMode | None,
    restart: bool,
    max_chars: int,
    timeout_sec: float,
) -> None:
    if (mode is not None or restart) and action != "mode":
        raise ValueError("mode and restart are accepted only for mode.")
    if restart and mode is None:
        raise ValueError("restart requires an explicit mode.")
    if bool(tab_id) != (action not in ("tabs", "open", "mode")):
        raise ValueError("tab_id is required only for actions on an existing tab.")
    if bool(url) != (action in ("open", "navigate")):
        raise ValueError("url is required only for open/navigate.")
    if bool(selector) != (action in ("click", "fill", "press")):
        raise ValueError("selector is required only for click/fill/press.")
    if text and action != "fill":
        raise ValueError("text is accepted only for fill.")
    if (action == "press" and key not in _KEY_NAMES) or (action != "press" and key):
        raise ValueError("key must name a supported key for press only.")
    if max_chars <= 0 or not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise ValueError("max_chars and timeout_sec must be positive and finite.")


async def _perform(
    pool: zendriver.BrowserPool,
    session: FetchSession,
    action: BrowserAction,
    *,
    tab_id: str,
    url: str,
    selector: str,
    text: str,
    key: str,
    mode: BrowserMode | None,
    restart: bool,
    max_chars: int,
    timeout_sec: float,
) -> BrowserResult:
    async with asyncio.timeout(timeout_sec):
        deadline = asyncio.get_running_loop().time() + timeout_sec
        if action == "mode":
            headless, restarted = await pool.mode(
                session.profile_dir,
                headless=None if mode is None else mode == "headless",
                restart=restart,
            )
            return BrowserResult(
                mode="headless" if headless else "headed",
                restarted=restarted,
            )
        chrome = await pool.browser(session.profile_dir, headless=session.headless)
        async with pool.targets_lock:
            await chrome.update_targets()
            if action == "tabs":
                return BrowserResult(tabs=tuple(_tab_info(tab) for tab in chrome.tabs))
            if action == "open":
                tab = await chrome.get("about:blank", new_tab=True)
            else:
                tab = next(
                    (tab for tab in chrome.tabs if str(tab.target_id) == tab_id),
                    None,
                )
                if tab is None:
                    raise BrowserError(
                        f"Browser tab {tab_id} not found; list tabs again.",
                    )
        async with pool.tab_lock(str(tab.target_id)):
            if action == "close":
                info = _tab_info(tab)
                await tab.close()
                return BrowserResult(tab=info)
            initial_url = url if action in ("open", "navigate") else str(tab.url)
            pinned_host(initial_url, allow_private_hosts=False)
            try:
                async with zendriver.guard_requests(
                    tab,
                    initial_url=initial_url,
                    allow_private_hosts=False,
                    headers={},
                    deadline=deadline,
                ) as guard:
                    if action in ("open", "navigate"):
                        await tab.get(url)
                    elif action in ("click", "fill", "press"):
                        await _control(tab, action, selector, text, key)
                    await _ready(tab)
                    await guard.drain()
                    if action == "screenshot":
                        return BrowserResult(
                            tab=_tab_info(tab),
                            screenshot=await _screenshot(tab),
                        )
                    snapshot = from_plain(
                        cast(object, await tab.evaluate(_INSPECT)),
                        _Snapshot,
                    )
                    pinned_host(snapshot.url, allow_private_hosts=False)
                    return BrowserResult(
                        tab=BrowserTab(
                            tab_id=str(tab.target_id),
                            url=snapshot.url,
                            title=snapshot.title,
                        ),
                        text=snapshot.text[:max_chars],
                        elements=snapshot.elements,
                        truncated=len(snapshot.text) > max_chars,
                    )
            except BaseException:
                if action == "open":
                    await tab.close()
                raise


async def _control(
    tab: Tab,
    action: BrowserAction,
    selector: str,
    text: str,
    key: str,
) -> None:
    elements = await tab.query_selector_all(selector)
    if len(elements) != 1:
        raise BrowserError(
            "Browser selector must match exactly one element; inspect the tab again.",
        )
    element = elements[0]
    if action == "click":
        await element.click()
    elif action == "fill":
        await _fill(tab, element, text)
    else:
        await element.send_keys(keys.SpecialKeys[_KEY_NAMES[key]])


async def _fill(tab: Tab, element: Element, text: str) -> None:
    if (
        element.node_name not in ("INPUT", "TEXTAREA")
        or "readonly" in element.attrs
        or "disabled" in element.attrs
        or element.attrs.get("type", "")
        not in ("", "text", "search", "tel", "url", "email", "password", "number")
    ):
        raise BrowserError("Fill requires an editable text input or textarea.")
    modifier = (
        keys.KeyModifiers.Meta if sys.platform == "darwin" else keys.KeyModifiers.Ctrl
    )
    await element.send_keys(
        keys.KeyEvents("a", modifiers=modifier).to_cdp_events(
            keys.KeyPressEvent.DOWN_AND_UP,
        ),
    )
    await element.send_keys(keys.SpecialKeys.BACKSPACE)
    await tab.send(cdp.input_.insert_text(text))


async def _screenshot(tab: Tab) -> bytes:
    try:
        return base64.b64decode(await tab.screenshot_b64(format="png"), validate=True)
    except binascii.Error as error:
        raise BrowserError(
            "Browser screenshot failed: invalid image encoding.",
        ) from error


async def _ready(tab: Tab) -> None:
    ready = asyncio.Event()

    async def on_ready(event: object) -> None:
        del event
        ready.set()

    register = cast(
        Callable[[type[object], Callable[..., object]], None],
        tab.add_handler,
    )
    remove = cast(
        Callable[[type[object], Callable[..., object]], None],
        tab.remove_handlers,
    )
    register(cdp.page.DomContentEventFired, on_ready)
    try:
        if await tab.evaluate("document.readyState") not in ("interactive", "complete"):
            await ready.wait()
    finally:
        remove(cdp.page.DomContentEventFired, on_ready)


def _tab_info(tab: Tab) -> BrowserTab:
    return BrowserTab(
        tab_id=str(tab.target_id),
        url=str(tab.url),
        title=tab.target.title if tab.target is not None else "",
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Snapshot:
    url: str
    title: str
    text: str
    elements: tuple[BrowserElement, ...]


_KEY_NAMES: Final = MappingProxyType(
    {
        "Enter": "ENTER",
        "Tab": "TAB",
        "Escape": "ESCAPE",
        "Backspace": "BACKSPACE",
        "Delete": "DELETE",
        "ArrowLeft": "ARROW_LEFT",
        "ArrowRight": "ARROW_RIGHT",
        "ArrowUp": "ARROW_UP",
        "ArrowDown": "ARROW_DOWN",
    },
)


_INSPECT: Final = r"""(() => {
  const selector = el => {
    if (el.id && document.querySelectorAll('#' + CSS.escape(el.id)).length === 1)
      return '#' + CSS.escape(el.id);
    const path = [];
    for (let node = el; node && node.nodeType === 1; node = node.parentElement) {
      const tag = node.tagName.toLowerCase();
      const siblings = node.parentElement ? [...node.parentElement.children].filter(x => x.tagName === node.tagName) : [node];
      path.unshift(tag + ':nth-of-type(' + (siblings.indexOf(node) + 1) + ')');
    }
    return path.join(' > ');
  };
  const elements = [...document.querySelectorAll('input:not([type=hidden]),textarea,select,button,a[href],[role=button],[role=checkbox],[contenteditable=true]')]
    .filter(el => el.getClientRects().length && getComputedStyle(el).visibility !== 'hidden')
    .map(el => ({
      selector: selector(el), tag: el.tagName.toLowerCase(),
      role: el.getAttribute('role') || '',
      label: (el.getAttribute('aria-label') || [...(el.labels || [])].map(x => x.innerText).join(' ') || el.getAttribute('placeholder') || (['INPUT', 'TEXTAREA'].includes(el.tagName) ? '' : el.innerText) || '').trim().slice(0, 300),
      kind: el.getAttribute('type') || '',
    }));
  const body = document.body ? document.body.cloneNode(true) : document.createElement('div');
  body.querySelectorAll('input,textarea,script,style,noscript').forEach(el => el.remove());
  return { url: location.href, title: document.title, text: body.textContent || '', elements };
})()"""
