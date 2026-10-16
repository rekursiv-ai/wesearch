"""The backend-agnostic entry point: :func:`search`.

A named backend is used alone. When none is given, a non-general
``categories`` selects ``searxng`` -- the only backend that serves result tabs
-- and anything else walks ``custom_types.DEFAULT_SEARCH_CHAIN`` in order,
returning the first non-empty answer. A backend that fails, is cooling down
after a block, or lacks its configuration (key, URL) hands over to the next;
only when every one fails does the caller see an error, naming each. A
category named ALONGSIDE a backend that cannot serve it raises rather than
overriding the stated choice. Every answer passes through the host-wide cache
(:mod:`.cache`). The overloads exist so a ``categories`` argument narrows the
return type to that category's record.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload
from urllib import error

import json
import logging
import sys

from wesearch.search.brave import brave
from wesearch.search.cache import cached_search
from wesearch.search.duckduckgo import duckduckgo
from wesearch.search.searxng import SearxngResult, searxng
from wesearch.search.types import (
    DEFAULT_SEARCH_CHAIN,
    CodeResult,
    FileResult,
    ImageResult,
    MapResult,
    MediaResult,
    PackageResult,
    PaperResult,
    SearchBackends,
    SearchError,
    SearchResult,
    SearxngCategory,
    TorrentResult,
    VideoResult,
)
from wesearch.types.errors import BotDetectionError, FetchError


logger = logging.getLogger(__name__)

# Below the sorted block on purpose: the Google backend is source-only, so this
# import must sit inside a copybarista fence, and ruff's isort pass reorders
# across a fence in the import block -- sweeping unrelated imports inside it and
# stripping them from the export.
if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from wesearch.fetch import FetchSession, Transport


@overload
def search(
    query: str,
    backend: SearchBackends | None = ...,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["science"],
    transport: Transport = ...,
    cache_ttl_sec: float = ...,
    session: FetchSession | None = ...,
) -> Sequence[PaperResult]: ...


@overload
def search(
    query: str,
    backend: SearchBackends | None = ...,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["images"],
    transport: Transport = ...,
    cache_ttl_sec: float = ...,
    session: FetchSession | None = ...,
) -> Sequence[ImageResult]: ...


@overload
def search(
    query: str,
    backend: SearchBackends | None = ...,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["videos"],
    transport: Transport = ...,
    cache_ttl_sec: float = ...,
    session: FetchSession | None = ...,
) -> Sequence[VideoResult]: ...


@overload
def search(
    query: str,
    backend: SearchBackends | None = ...,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["news", "music"],
    transport: Transport = ...,
    cache_ttl_sec: float = ...,
    session: FetchSession | None = ...,
) -> Sequence[MediaResult]: ...


@overload
def search(
    query: str,
    backend: SearchBackends | None = ...,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["map"],
    transport: Transport = ...,
    cache_ttl_sec: float = ...,
    session: FetchSession | None = ...,
) -> Sequence[MapResult]: ...


@overload
def search(
    query: str,
    backend: SearchBackends | None = ...,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["it"],
    transport: Transport = ...,
    cache_ttl_sec: float = ...,
    session: FetchSession | None = ...,
) -> Sequence[PackageResult | CodeResult | SearchResult]: ...


@overload
def search(
    query: str,
    backend: SearchBackends | None = ...,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["files"],
    transport: Transport = ...,
    cache_ttl_sec: float = ...,
    session: FetchSession | None = ...,
) -> Sequence[FileResult | TorrentResult | SearchResult]: ...


@overload
def search(
    query: str,
    backend: SearchBackends | None = ...,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: SearxngCategory = ...,
    transport: Transport = ...,
    cache_ttl_sec: float = ...,
    session: FetchSession | None = ...,
) -> Sequence[SearchResult]: ...


def search(
    query: str,
    backend: SearchBackends | None = None,
    num_results: int = 10,
    headers: dict[str, str] | None = None,
    *,
    categories: SearxngCategory = "general",
    transport: Transport = "auto",
    cache_ttl_sec: float = 3600.0,
    session: FetchSession | None = None,
) -> Sequence[SearxngResult]:
    """Search one named backend, or walk the fallback chain when none is named.

    Args:
      query: Search query string.
      backend: Backend name. ``None`` walks ``DEFAULT_SEARCH_CHAIN`` (or uses
        SearXNG for a non-general category).
      num_results: Maximum results to return.
      headers: Optional override headers forwarded to the backend.
      categories: SearXNG result category; only the ``"searxng"`` backend
        honors a non-default value (the HTML-scraping backends serve general
        web results only). Defaults to ``"general"``.
      transport: Retrieval transport forwarded to the selected backend.
      cache_ttl_sec: Seconds a cached answer stays valid; ``0`` bypasses the
        host-wide cache (a probe that must see the backend as it is now).
      session: Connections to reuse; ``None`` uses the process's shared session.

    Returns:
      results: One typed record per hit. SearXNG categories with extra
        structure yield a category-specific :class:`SearchResult` subclass;
        all other paths yield plain :class:`SearchResult`.

    Raises:
      BotDetectionError: When a NAMED backend was bot-blocked; it carries the
        recovery guidance.
      SearchError: When the named backend, or every backend in the chain, failed.

    """
    if categories != "general" and backend not in (None, "searxng"):
        # An EXPLICIT backend still errors. sagent used to overwrite it, so
        # ``backend="duckduckgo", categories="science"`` silently ran against
        # SearXNG -- a stated choice replaced without a word.
        raise ValueError(
            f"'categories' is only supported by the 'searxng' backend, not {backend!r}.",
        )
    if backend is not None:
        chain: tuple[str, ...] = (backend,)
    elif categories != "general":
        # A non-general tab exists only on SearXNG, so asking for one IS asking
        # for that backend.
        chain = ("searxng",)
    else:
        chain = DEFAULT_SEARCH_CHAIN
    runners = _runners(
        query,
        num_results=num_results,
        headers=headers,
        categories=categories,
        transport=transport,
        session=session,
    )
    unknown = [name for name in chain if name not in runners]
    if unknown:
        raise ValueError(f"Unknown backend: {unknown[0]!r}")
    failures: list[str] = []
    for name in chain:
        try:
            results = cached_search(
                {
                    "backend": name,
                    "query": query,
                    "num_results": num_results,
                    "categories": categories,
                    "headers": headers or {},
                },
                runners[name],
                ttl_sec=cache_ttl_sec,
            )
        except (BotDetectionError, SearchError):
            # A named backend's own error already says what to do (solve a
            # captcha, set SEARXNG_URL), so it propagates intact; inside the
            # chain it is one more reason to move on.
            if len(chain) == 1:
                raise
            failures.append(f"{name}: {sys.exception()}")
            continue
        except (
            FetchError,
            OSError,
            TimeoutError,
            error.URLError,
            json.JSONDecodeError,
        ) as e:
            if len(chain) == 1:
                raise SearchError(f"{name} search failed: {e}") from e
            failures.append(f"{name}: {e}")
            continue
        if results or name == chain[-1]:
            return results
        failures.append(f"{name}: no results")
    raise SearchError(f"every search backend failed: {'; '.join(failures)}")


def _runners(
    query: str,
    *,
    num_results: int,
    headers: dict[str, str] | None,
    categories: SearxngCategory,
    transport: Transport,
    session: FetchSession | None = None,
) -> Mapping[str, Callable[[], Sequence[SearxngResult]]]:
    """Return one zero-argument call per backend, bound to this query."""
    # A table rather than an if-chain: a chain over a closed ``Literal`` exhausts
    # it, so the unknown-backend guard becomes provably dead -- at a DIFFERENT
    # branch per build, since the export drops ``google``.
    return {
        "brave": lambda: brave(
            query,
            num_results,
            headers,
            transport=transport,
            session=session,
        ),
        "searxng": lambda: searxng(
            query,
            num_results,
            headers,
            categories=categories,
            transport=transport,
            session=session,
        ),
        "duckduckgo": lambda: duckduckgo(
            query,
            num_results,
            headers,
            transport=transport,
            session=session,
        ),
    }
