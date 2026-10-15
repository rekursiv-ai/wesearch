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

from concurrent.futures import ProcessPoolExecutor
from typing import Protocol, cast, get_args

import argparse
import time

from wesearch.fetch.transport import zendriver
from wesearch.lib.userdirs import data_dir
from wesearch.search.custom_types import SearchBackends
from wesearch.search.search import search


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
    return failures


class Flags(Protocol):
    """Parsed command-line flags."""

    backends: list[str]
    query: str
    processes: int
    url: str


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
        "--url",
        default="https://example.com/",
        help="Page each browser process fetches.",
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


def _probe_shared_browser(processes: int, url: str) -> tuple[bool, str, float]:
    """Fetch ``url`` from ``processes`` processes at once through the shared Chrome."""
    started = time.monotonic()
    with ProcessPoolExecutor(processes) as pool:
        landed = list(pool.map(_browser_fetch, [url] * processes))
    elapsed = time.monotonic() - started
    ok = sum(landed)
    return ok == processes, f"{ok}/{processes} processes fetched {url}", elapsed


def _browser_fetch(url: str) -> bool:
    """Fetch ``url`` through the shared Chrome from a worker process."""
    try:
        result = zendriver.fetch_zendriver(
            url,
            profile_dir=data_dir() / "rekursiv-ai" / "wesearch" / "fetch-zendriver",
        )
    except Exception:  # noqa: BLE001 -- a failed fetch is a probe result.
        return False
    finally:
        zendriver.shutdown_browsers()
    return bool(result.body)
