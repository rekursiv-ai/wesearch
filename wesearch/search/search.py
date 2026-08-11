"""The backend-agnostic entry point: :func:`search`.

Dispatches to the backend named by the caller, or to
``custom_types.DEFAULT_SEARCH_BACKEND`` when none is given -- a build-time
constant, NOT an environment probe. Nothing here inspects ``SEARXNG_URL``;
asking for SearXNG is explicit. The overloads exist so a
``categories`` argument narrows the return type to that category's record.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload

import json
import logging
import urllib.error

from wesearch.fetch import Transport
from wesearch.search.custom_types import (
    DEFAULT_SEARCH_BACKEND,
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
    TorrentResult,
    VideoResult,
)
from wesearch.search.duckduckgo import duckduckgo
from wesearch.search.searxng import SearxngCategory, SearxngResult, searxng
from wesearch.types.errors import BotDetectionError, FetchError


logger = logging.getLogger(__name__)


# Below the sorted block on purpose: the Google backend is source-only, so this
# import must sit inside a copybarista fence, and ruff's isort pass reorders
# across a fence in the import block -- sweeping unrelated imports inside it and
# stripping them from the export.
@overload
def search(
    query: str,
    backend: Literal["searxng"],
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["science"],
    transport: Transport = ...,
) -> Sequence[PaperResult]: ...


@overload
def search(
    query: str,
    backend: Literal["searxng"],
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["images"],
    transport: Transport = ...,
) -> Sequence[ImageResult]: ...


@overload
def search(
    query: str,
    backend: Literal["searxng"],
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["videos"],
    transport: Transport = ...,
) -> Sequence[VideoResult]: ...


@overload
def search(
    query: str,
    backend: Literal["searxng"],
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["news", "music"],
    transport: Transport = ...,
) -> Sequence[MediaResult]: ...


@overload
def search(
    query: str,
    backend: Literal["searxng"],
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["map"],
    transport: Transport = ...,
) -> Sequence[MapResult]: ...


@overload
def search(
    query: str,
    backend: Literal["searxng"],
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["it"],
    transport: Transport = ...,
) -> Sequence[PackageResult | CodeResult | SearchResult]: ...


@overload
def search(
    query: str,
    backend: Literal["searxng"],
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["files"],
    transport: Transport = ...,
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
) -> Sequence[SearchResult]: ...


def search(
    query: str,
    backend: SearchBackends | None = None,
    num_results: int = 10,
    headers: dict[str, str] | None = None,
    *,
    categories: SearxngCategory = "general",
    transport: Transport = "auto",
) -> Sequence[SearxngResult]:
    """Dispatch to the named search backend.

    Args:
      query: Search query string.
      backend: Backend name. Defaults to ``DEFAULT_SEARCH_BACKEND``.
      num_results: Maximum results to return.
      headers: Optional override headers forwarded to the backend.
      categories: SearXNG result category; only the ``"searxng"`` backend
        honors a non-default value (the HTML-scraping backends serve general
        web results only). Defaults to ``"general"``.
      transport: Retrieval transport forwarded to the selected backend.

    Returns:
      results: One typed record per hit. SearXNG categories with extra
        structure yield a category-specific :class:`SearchResult` subclass;
        all other paths yield plain :class:`SearchResult`.

    """
    if backend is None:
        backend = DEFAULT_SEARCH_BACKEND
    if categories != "general" and backend != "searxng":
        raise ValueError(
            f"'categories' is only supported by the 'searxng' backend, not {backend!r}."
        )
    try:
        if backend == "searxng":
            return searxng(
                query,
                num_results,
                headers,
                categories=categories,
                transport=transport,
            )
        if backend == "duckduckgo":
            return duckduckgo(query, num_results, headers, transport=transport)

    except BotDetectionError:
        # A bot-detection block carries actionable, type-specific guidance
        # (solve captcha / rotate IP). It is-a FetchError, so it MUST be caught
        # before the generic handler below, or that handler would flatten it into
        # a guidance-less SearchError. Propagate it intact.
        raise
    except (
        FetchError,
        OSError,
        TimeoutError,
        urllib.error.URLError,
        json.JSONDecodeError,
    ) as e:
        raise SearchError(f"{backend} search failed: {e}") from e
    # Unreachable in EVERY build, not just this one: the branches above exhaust
    # `SearchBackends`, so basedpyright proves the line dead whether the Literal
    # has two members (export) or three (monorepo). Kept anyway -- it is what
    # turns a runtime-invalid backend string into a named error rather than a
    # silent `None` return.
    raise ValueError(  # pyright: ignore[reportUnreachable] -- see above
        f"Unknown backend: {backend!r}"
    )
