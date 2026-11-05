# ruff: noqa: T201 -- The canary is CLI-only; its report is its output.
"""Probe every search backend and the shared browser once; report what is blocked.

Runs one live query per backend, bypassing the result cache and the fallback
chain, so a block shows up here before agents hit it. With ``--processes N``
it also fans N processes out over the shared Chrome and reports whether every
fetch landed and how many Chromes the profile ended up with -- the
36-agent contention that used to evict browsers mid-fetch.

Exit status is the number of failed probes, so a scheduler can alert on it.

Examples:
  python -m wesearch.tools
  python -m wesearch.tools --processes 12
  python -m wesearch.tools --backends duckduckgo searxng

"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Protocol, cast, get_args

import argparse
import time

from wesearch.browser import BrowserError, browser
from wesearch.fetch.transport import zendriver
from wesearch.fetch.types import FetchSession
from wesearch.search.search import search
from wesearch.search.types import SearchBackends


def main() -> int:
    """Run the probes and print one line per probe.

    Returns:
      failures: The number of probes that failed, used as the exit status.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    failures = 0
    for backend in flags.backends:
        ok, detail, elapsed = _probe_backend(backend, flags.query)
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {backend:<11} {elapsed:5.1f}s  {detail}")
    if flags.processes:
        ok, detail, elapsed = _probe_shared_browser(flags.processes, flags.url)
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {'browser':<11} {elapsed:5.1f}s  {detail}")
    if flags.threads:
        ok, detail, elapsed = _probe_shared_browser(
            flags.threads,
            flags.url,
            threads=True,
        )
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {'browser':<11} {elapsed:5.1f}s  {detail}")
    if flags.browser_actions:
        ok, detail, elapsed = _probe_browser_actions()
        failures += not ok
        print(
            f"{'ok  ' if ok else 'FAIL'} {'browser-actions':<11} {elapsed:5.1f}s  {detail}",
        )
    return failures


class Flags(Protocol):
    """Parsed command-line flags."""

    backends: list[str]
    query: str
    processes: int
    threads: int
    url: str
    browser_actions: bool


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--backends",
        nargs="*",
        default=list(get_args(SearchBackends)),
        choices=list(get_args(SearchBackends)),
        help="Backends to probe (default: all).",
    )
    parser.add_argument(
        "--query",
        default="python asyncio timeout",
        help="Query sent to each backend.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=0,
        help="Fan this many processes over the shared Chrome (0 skips).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Also probe N concurrent fetches sharing one browser connection.",
    )
    parser.add_argument(
        "--url",
        default="https://example.com/",
        help="Page each browser process fetches.",
    )
    parser.add_argument(
        "--browser-actions",
        action="store_true",
        help="Exercise a synthetic httpbin form, screenshots, and persistent tabs.",
    )


def _probe_backend(backend: str, query: str) -> tuple[bool, str, float]:
    """Query one backend live; return ``(ok, detail, seconds)``."""
    started = time.monotonic()
    try:
        # Uncached: a cached answer would report healthy a backend blocked now.
        results = search(
            query,
            backend=cast(SearchBackends, backend),
            num_results=3,
            cache_ttl_sec=0,
        )
    except Exception as error:  # noqa: BLE001 -- every failure is a probe result.
        return (
            False,
            f"{type(error).__name__}: {error}"[:160],
            time.monotonic() - started,
        )
    elapsed = time.monotonic() - started
    if not results:
        return False, "no results (soft block?)", elapsed
    return True, f"{len(results)} results; first {results[0].url}", elapsed


def _probe_shared_browser(
    workers: int,
    url: str,
    *,
    threads: bool = False,
) -> tuple[bool, str, float]:
    """Fetch ``url`` concurrently through one connection or independent processes."""
    started = time.monotonic()
    executor = ThreadPoolExecutor if threads else ProcessPoolExecutor
    with executor(workers) as pool:
        landed = list(pool.map(_browser_fetch, [url] * workers))
    elapsed = time.monotonic() - started
    ok = sum(landed)
    kind = "threads" if threads else "processes"
    return ok == workers, f"{ok}/{workers} {kind} fetched {url}", elapsed


def _probe_browser_actions() -> tuple[bool, str, float]:
    """Exercise a public synthetic form across persistent browser connections."""
    started = time.monotonic()
    try:
        _exercise_browser_actions()
    except (BrowserError, ValueError, OSError) as error:
        return (
            False,
            f"{type(error).__name__}: {error}"[:160],
            time.monotonic() - started,
        )
    return (
        True,
        "persistent tab; fill, press, click, PNG, submitted value, close passed",
        time.monotonic() - started,
    )


def _exercise_browser_actions() -> None:
    """Submit only synthetic public data; preserve every pre-existing tab."""
    with FetchSession() as session:
        before = {tab.tab_id for tab in browser("tabs", session=session).tabs}
        opened = browser("open", url="https://httpbin.org/forms/post", session=session)
        if opened.tab is None:
            raise ValueError("Open returned no tab")
        tab_id = opened.tab.tab_id
    with FetchSession() as session:
        try:
            browser("inspect", tab_id=tab_id, session=session)
            browser(
                "fill",
                tab_id=tab_id,
                selector="input[name=custname]",
                text="wesearch-canary",
                session=session,
            )
            browser(
                "press",
                tab_id=tab_id,
                selector="input[name=custname]",
                key="Tab",
                session=session,
            )
            browser(
                "click",
                tab_id=tab_id,
                selector="input[name=size][value=small]",
                session=session,
            )
            image = browser("screenshot", tab_id=tab_id, session=session).screenshot
            if not image.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Screenshot returned no PNG")
            echoed = browser("click", tab_id=tab_id, selector="button", session=session)
            if "wesearch-canary" not in echoed.text:
                raise ValueError("Submitted form did not echo the filled value")
        finally:
            browser("close", tab_id=tab_id, session=session)
        after = {tab.tab_id for tab in browser("tabs", session=session).tabs}
        if before - after or tab_id in after:
            raise ValueError("Probe changed unrelated tabs or failed to close its tab")


def _browser_fetch(url: str) -> bool:
    """Fetch ``url`` through the shared Chrome from a worker process."""
    try:
        return bool(zendriver.fetch_zendriver(url))
    except Exception:  # noqa: BLE001 -- a failed fetch is a probe result.
        return False
