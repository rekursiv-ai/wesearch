"""The backend-agnostic entry point: :func:`search`.

Dispatches to the backend named by the caller. When none is given, a
non-general ``categories`` selects ``searxng`` -- the only backend that serves
result tabs -- and anything else takes ``custom_types.DEFAULT_SEARCH_BACKEND``,
a build-time constant, NOT an environment probe. Nothing here inspects
``SEARXNG_URL``; asking for SearXNG is explicit. A category named ALONGSIDE a
backend that cannot serve it raises rather than overriding the stated choice.
The overloads exist so a ``categories`` argument narrows the return type to
that category's record.
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
    SearxngCategory,
    TorrentResult,
    VideoResult,
)
from wesearch.search.duckduckgo import duckduckgo
from wesearch.search.searxng import SearxngResult, category_gloss, searxng
from wesearch.types.errors import BotDetectionError, FetchError
from wesearch.types.params import PolicyParams
from wesearch.types.schema import Field, Schema


logger = logging.getLogger(__name__)


# Below the sorted block on purpose: the Google backend is source-only, so this
# import must sit inside a copybarista fence, and ruff's isort pass reorders
# across a fence in the import block -- sweeping unrelated imports inside it and
# stripping them from the export.
class SearchParamsSchema(Schema):
    """What every search surface accepts.

    The counterpart to
    :class:`wesearch.fetch.custom_types.FetchParamsSchema`, and here for
    the same reason: the sagent tool's schema, that tool's directive
    validation, and the MCP server's signature are three renderings of one
    description rather than three copies of it.

    Beside :func:`search` rather than in ``custom_types`` because the
    ``categories`` prose is rendered from ``searxng.CATEGORIES``, which reads
    the vocabulary FROM ``custom_types``. Declaring the schema at the leaf
    would close that loop; declaring it here, where importing a backend is
    already the point, leaves each tab's gloss beside the parser it names.
    """

    query = Field[str](
        annotation=str, required=True, description="Search query string."
    )
    backend = Field[SearchBackends](
        annotation=SearchBackends,
        # No default: an unnamed backend is RESOLVED by ``search`` -- a
        # non-general category selects SearXNG, anything else takes the
        # build's default. Naming one here would preempt that.
        description=(
            "Search backend. Omit to let a category choose, else this build's"
            f' default ("{DEFAULT_SEARCH_BACKEND}").'
        ),
    )
    categories = Field[SearxngCategory](
        annotation=SearxngCategory,
        default="general",
        description=(
            "SearXNG result category (tab). A non-default value selects the "
            "SearXNG backend when none is named, and is rejected alongside an "
            f"explicit non-SearXNG one.\n{category_gloss()}"
        ),
    )
    transport = Field[Transport](
        annotation=Transport,
        default=PolicyParams.field_default("transport", Transport),
        description=(
            "Retrieval path. 'auto' tries curl and escalates to Zendriver when "
            "a site bot-blocks it. Set an explicit transport to stress a path."
        ),
    )


@overload
def search(
    query: str,
    backend: SearchBackends | None = ...,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["science"],
    transport: Transport = ...,
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
        # A non-general tab exists only on SearXNG, so asking for one IS asking
        # for that backend when the caller named none. Resolved here rather than
        # in each adapter: sagent forced it and the MCP server did not, so the
        # same call raised on one surface and worked on the other -- and only in
        # the public build, whose default backend is not SearXNG.
        backend = "searxng" if categories != "general" else DEFAULT_SEARCH_BACKEND
    if categories != "general" and backend != "searxng":
        # An EXPLICIT backend still errors. sagent used to overwrite it, so
        # ``backend="duckduckgo", categories="science"`` silently ran against
        # SearXNG -- a stated choice replaced without a word.
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
