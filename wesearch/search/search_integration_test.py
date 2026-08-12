r"""Live search integration tests across the configured web backends.

Verifies each general-web backend actually returns usable results against the
live network over several distinct queries, plus one SearXNG probe per category
(shape, not content). Carry the ``integration`` marker (deselected by default,
so a plain unit run stays offline). The only precondition is network reach: the
SearXNG tests additionally need ``SEARXNG_URL`` (the same variable the production
code reads) and are skipped without it. Run against the live network with::

    SEARXNG_URL=https://your.searxng \\
        uv run --frozen pytest -m integration \\
        wesearch/search_integration_test.py

Assertions check the SHAPE of the response (result type, populated
``url``/``title``, category-specific fields present and typed), never specific
content -- live search results are non-deterministic. Backend-level infra
conditions that are not parser faults are reported via ``pytest.skip`` rather
than failing: a bot-detection/CAPTCHA block (any backend) and an empty response
(a soft IP-level block or a category an instance's engines do not serve both
return zero results from this egress -- availability, not a parser fault). A
*malformed* result (non-web url, empty title) still fails: that is a real parser
regression, which is what these tests exist to catch.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager

import os
import time

import filelock
import pytest

from wesearch.fetch.transport.zendriver import BrowserUnavailableError
from wesearch.lib.userdirs import state_dir
from wesearch.search.custom_types import (
    CodeResult,
    FileResult,
    ImageResult,
    MapResult,
    MediaResult,
    PackageResult,
    PaperResult,
    SearchResult,
    SearxngCategory,
    TorrentResult,
    VideoResult,
)
from wesearch.search.duckduckgo import duckduckgo
from wesearch.search.searxng import searxng
from wesearch.types.errors import BotDetectionError, FetchError


pytestmark = [
    pytest.mark.integration,
    pytest.mark.xdist_group(name="live_search"),
]

_searxng_required = pytest.mark.skipif(
    not os.environ.get("SEARXNG_URL"),
    reason="SEARXNG_URL not set; live SearXNG instance required.",
)

# A bounded SEMAPHORE, not a mutex. One lock file conflated two unrelated
# needs: Zendriver's persistent control socket, which genuinely admits one
# user, and "do not burst the search instance", which measurement does not
# support -- 10 concurrent SearXNG category queries complete in 2.33s wall with
# zero errors, against 17.55s serialized. Under the repo-default
# ``--dist=worksteal`` (``pyproject.toml:384``) ``xdist_group`` is IGNORED, so
# 24 workers piled onto one file and every loser paid the full timeout and then
# skipped -- the 8.04s cluster, which was queueing, never querying.
#
# Per USER, not per machine. This lived in ``gettempdir()``, which is shared:
# every checkout and every operator on the host queued on one file, so a
# colleague's gate run blocked this one.
_LOCK_DIR = state_dir("rekursiv-ai") / "wesearch"
# ``state_dir`` resolves a path and does not create it (``userdirs.py:138``).
_LOCK_DIR.mkdir(parents=True, exist_ok=True)

# Concurrency admitted against the live backends. Sized under the measured
# ceiling above, not at it, because these run beside the rest of the suite.
_SEARCH_LANES = 4
_SEARCH_LOCKS = tuple(
    filelock.FileLock(str(_LOCK_DIR / f"live-search.{lane}.lock"))
    for lane in range(_SEARCH_LANES)
)

# Backends already proved unroutable, so the remaining cases skip instead of
# each re-paying the connect timeout. An egress either reaches a host or does
# not; every parametrized case re-learning that costs the ceiling again for no
# added signal, and worse, each one holds a lane while it waits.
#
# A marker FILE, not a module-level set: under ``-n`` every xdist worker is a
# separate process, so an in-process set is re-learned once per worker -- and
# each re-learning holds a lane for the full connect timeout.
_UNREACHABLE_DIR = _LOCK_DIR / "unreachable"
_UNREACHABLE_DIR.mkdir(parents=True, exist_ok=True)

# How long a marker suppresses retries. Bounded because the marker outlives the
# run: a route restored after a VPN or network change must not leave the backend
# permanently skipped, which would silently retire a test. One run's worth of
# suppression is the whole benefit; the next run re-probes.
_UNREACHABLE_TTL_SEC = 300.0


@contextmanager
def _admitted(*, timeout_sec: float) -> Generator[None]:
    """Hold one of :data:`_SEARCH_LANES` lanes, or raise ``filelock.Timeout``.

    A counting semaphore built from N lock files: try each in turn without
    blocking, so an idle lane is taken immediately and only a genuinely
    saturated set falls through to a bounded wait. The wait is bounded by ONE
    QUERY -- derived, not tuned. That constant was picked three times (20s, 5s,
    30s) and every value was wrong the same way: a case waiting longer than one
    query pays MORE to be skipped than an uncontended case pays to actually
    run, which is strictly worse than having no lock at all.
    """
    deadline = time.monotonic() + timeout_sec
    while True:
        for lock in _SEARCH_LOCKS:
            try:
                with lock.acquire(blocking=False):
                    yield
                    return
            except filelock.Timeout:
                continue
        # Every lane was busy on that pass. Poll them ALL rather than block on
        # one: waiting on a single lane starves behind whichever peer happens
        # to hold it while three others go idle -- measured as 15 of 25 cases
        # skipping under ``-n 24``.
        if time.monotonic() >= deadline:
            raise filelock.Timeout(str(_LOCK_DIR))
        time.sleep(0.05)


def _query_once[T](
    fetch: Callable[[float], list[T]], *, backend: str, timeout_sec: float = 8.0
) -> list[T]:
    """Run one live query under the search lock and return its results.

    Exactly ONE attempt, deliberately. A retry loop lived here and was the single
    largest cost in the file: every outcome these tests treat as a skip (empty
    page, soft block) first paid the full backoff schedule, so a backend serving
    nothing cost ~30s per case instead of one query. Retrying also cannot change
    any verdict -- an empty result skips whether it took one attempt or three --
    so the wait bought latency and no signal.

    A failure that is availability rather than a parser fault skips: an
    unroutable egress (``FetchError`` with ``status == 0``), a missing browser,
    a bot-detection block, or a lock another worker holds. A malformed result
    still fails, which is what the caller asserts on.

    Args:
      fetch: Performs one live query, given the HTTP ceiling to apply, and
        returns its results.
      backend: Backend name, used to remember an unroutable egress across the
        parametrized cases so only the first one pays the connect timeout.
      timeout_sec: HTTP ceiling handed to ``fetch``, and the lock wait. These
        tests assert PARSER shape, so a backend that answers slowly proves
        nothing a fast answer does not; the ceiling exists to bound a wedged
        egress. Measured: on the library defaults (``timeout_sec=30``,
        ``retries=2``) one DuckDuckGo case took 94s against an unroutable
        egress, tripping the 60s pytest-timeout as a FAILURE rather than the
        availability skip it is. The HANDSHAKE ceiling is deliberately NOT set
        here -- the library's ``connect_timeout_sec`` default already fails an
        unroutable host in 3s, and a value restated here would drift from it.

    Returns:
      results: The backend's results, empty when it served none.

    """
    unreachable_marker = _UNREACHABLE_DIR / backend
    if (
        unreachable_marker.exists()
        and time.time() - unreachable_marker.stat().st_mtime < _UNREACHABLE_TTL_SEC
    ):
        pytest.skip(f"{backend} already proved unreachable from this egress")
    try:
        with _admitted(timeout_sec=timeout_sec):
            return fetch(timeout_sec)
    except filelock.Timeout:
        pytest.skip(f"{backend}: every live-search lane busy")
    except BrowserUnavailableError as error:
        # No usable Chrome on this host (CI, headless box): a capability gap,
        # not a parser fault, and retrying cannot conjure a browser.
        pytest.skip(f"browser subsystem unavailable: {error}")
    except BotDetectionError as error:
        # An egress-IP CAPTCHA/challenge block is persistent (verified), so this
        # is availability too. Ordered BEFORE FetchError: it subclasses it
        # (``errors.py:47``).
        pytest.skip(f"{backend} served an automated-access block: {error}")
    except FetchError as error:
        if error.status != 0:
            raise
        unreachable_marker.touch()
        pytest.skip(f"{backend} unreachable from this egress: {error}")


# Distinct queries spanning different topics, so a general-web pass reflects the
# backend handling varied input rather than one lucky term. TWO, not five: these
# assert response SHAPE (a well-formed SearchResult with a web URL and a
# non-empty title), and a parser that shapes two unrelated topics correctly is
# not made more proven by three more. Five cost 3x the wall time of the whole
# rest of the file and, on a backend this egress cannot reach, five identical
# connect timeouts. Two still catches a parser keyed to one query's markup.
_QUERIES = [
    "open source software",
    "golden gate bridge",
]


def _assert_web_results(results: list[SearchResult], query: str, backend: str) -> None:
    """Assert live web results are well-shaped; skip when the backend served none.

    An empty response is availability (a soft IP-level block from this egress),
    not a parser fault, so it skips rather than fails. Any returned result must
    still be a well-formed :class:`SearchResult`.
    """
    if not results:
        pytest.skip(f"{backend} returned no results for {query!r} (soft block?)")
    for r in results:
        assert isinstance(r, SearchResult)
        assert r.url.startswith(("http://", "https://")), (
            f"result for {query!r} has non-web url: {r.url!r}"
        )
        assert r.title, f"result for {query!r} has empty title"


# ---------------------------------------------------------------------------
# Per-backend general web search
# ---------------------------------------------------------------------------


class TestDuckDuckGoLive:
    @pytest.mark.parametrize("query", _QUERIES)
    def test_returns_web_results(self, query: str) -> None:
        results = _query_once(
            lambda timeout_sec: duckduckgo(
                query, num_results=5, timeout_sec=timeout_sec, retries=0
            ),
            backend="duckduckgo",
        )
        _assert_web_results(results, query, "duckduckgo")


class TestSearxngLive:
    @_searxng_required
    @pytest.mark.parametrize("query", _QUERIES)
    def test_returns_web_results(self, query: str) -> None:
        results = _query_once(
            lambda timeout_sec: list(
                searxng(
                    query, num_results=5, categories="general", timeout_sec=timeout_sec
                )
            ),
            backend="searxng",
        )
        _assert_web_results(results, query, "searxng")


# ---------------------------------------------------------------------------
# SearXNG per-category shape checks
# ---------------------------------------------------------------------------

# Per-category probe queries, broad enough that any reasonably configured
# instance returns at least one hit.
_CATEGORY_QUERY: dict[SearxngCategory, str] = {
    "general": "open source software",
    "images": "golden gate bridge",
    "videos": "python tutorial",
    "news": "technology",
    "map": "Eiffel Tower Paris",
    "music": "Beethoven symphony",
    "it": "numpy",
    "science": "attention is all you need",
    "files": "ubuntu iso",
    "social media": "opensource",
}


def _category_results(category: SearxngCategory) -> list[SearchResult]:
    """Fetch one category's live results."""
    return _query_once(
        lambda timeout_sec: list(
            searxng(
                _CATEGORY_QUERY[category], categories=category, timeout_sec=timeout_sec
            )
        ),
        backend="searxng",
    )


def _require(results: list[SearchResult], category: str) -> SearchResult:
    """Return the first result, or skip when the instance yielded none."""
    if not results:
        pytest.skip(f"instance returned no results for category {category!r}")
    first = results[0]
    # Every result, whatever its subclass, is a SearchResult with a URL.
    assert isinstance(first, SearchResult)
    assert first.url
    return first


@_searxng_required
class TestSearxngCategoriesLive:
    def test_general(self) -> None:
        results = _category_results("general")
        first = _require(results, "general")
        assert type(first) is SearchResult

    def test_science(self) -> None:
        results = _category_results("science")
        first = _require(results, "science")
        assert isinstance(first, PaperResult)

    def test_images(self) -> None:
        results = _category_results("images")
        first = _require(results, "images")
        assert isinstance(first, ImageResult)
        # At least one image in the page should carry a full-image URL.
        assert any(isinstance(r, ImageResult) and r.image_url for r in results), (
            "no ImageResult carried an image_url"
        )

    def test_videos(self) -> None:
        results = _category_results("videos")
        first = _require(results, "videos")
        assert isinstance(first, VideoResult)
        assert isinstance(first, MediaResult)  # VideoResult is-a MediaResult

    def test_news(self) -> None:
        results = _category_results("news")
        first = _require(results, "news")
        assert isinstance(first, MediaResult)
        assert not isinstance(first, VideoResult)

    def test_music(self) -> None:
        results = _category_results("music")
        first = _require(results, "music")
        assert isinstance(first, MediaResult)

    def test_map(self) -> None:
        results = _category_results("map")
        first = _require(results, "map")
        assert isinstance(first, MapResult)
        # A place result should expose coordinates somewhere on the page.
        assert any(
            isinstance(r, MapResult) and r.latitude is not None for r in results
        ), "no MapResult carried a latitude"

    def test_it(self) -> None:
        results = _category_results("it")
        _require(results, "it")
        # Engines emit packages.html / code.html / default.html; every element
        # must be one of the typed IT shapes (or the web fallback).
        assert all(
            isinstance(r, (PackageResult, CodeResult, SearchResult)) for r in results
        )
        assert any(isinstance(r, PackageResult) for r in results), (
            "no PackageResult among IT results"
        )

    def test_files(self) -> None:
        results = _category_results("files")
        _require(results, "files")
        assert all(
            isinstance(r, (FileResult, TorrentResult, SearchResult)) for r in results
        )

    def test_social_media(self) -> None:
        results = _category_results("social media")
        first = _require(results, "social media")
        assert isinstance(first, SearchResult)


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
