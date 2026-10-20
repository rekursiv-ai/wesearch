"""SearXNG search backend: queries the JSON API of the instance at ``SEARXNG_URL``.

Each category (tab) maps to a typed result shape; see :data:`CATEGORIES`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast, get_args, overload
from urllib.parse import urlencode

import json
import logging
import math
import os
import re

from treekle import from_plain, json_loads

from wesearch.fetch import (
    ContentParams,
    PolicyParams,
    RequestParams,
    RetryParams,
    Transport,
    fetch,
)
from wesearch.search.types import (
    CodeResult,
    FileResult,
    ImageResult,
    MapResult,
    MediaResult,
    PackageResult,
    PaperResult,
    SearchError,
    SearchResult,
    SearxngCategory,
    TorrentResult,
    VideoResult,
    clean_text,
)


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from wesearch.fetch import FetchSession


logger = logging.getLogger(__name__)


# ``VideoResult`` is covered by its base ``MediaResult``.
type SearxngResult = (
    PaperResult
    | ImageResult
    | MediaResult
    | MapResult
    | PackageResult
    | CodeResult
    | FileResult
    | TorrentResult
    | SearchResult
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CategoryInfo:
    """What a tab carries beyond its name: prose and a result parser.

    Attributes:
      gloss: One phrase naming what the tab returns, rendered into every tool
        description.
      parser: Reader for the tab's result template. ``None`` means the tab has
        no structured template and reads through the generic web parser.

    """

    gloss: str

    parser: Callable[[dict[str, object]], SearxngResult] | None = None


def _coordinate(value: object) -> float | None:
    """Return one finite map coordinate, or None when unknown."""
    if type(value) not in {int, float}:
        return None
    coordinate = float(cast(float, value))
    return coordinate if math.isfinite(coordinate) else None


# A field's JSON type varies by upstream engine (e.g. a float ``length``); one odd
# field must not reject the whole result.
def _text(item: dict[str, object], key: str) -> str:
    """Return ``item[key]`` when it is a string, else ``""``."""
    return value if isinstance(value := item.get(key), str) else ""


def _published(item: dict[str, object]) -> datetime | None:
    """Parse ``publishedDate``, or None when absent or not ISO 8601."""
    text = _text(item, "publishedDate")
    try:
        return datetime.fromisoformat(text) if text else None
    except ValueError:
        return None


def _count(value: object) -> int | None:
    """Return a JSON integer count, or None for anything else (``"2"``, ``1.9``)."""
    return value if type(value) is int else None


def _searxng_web(item: dict[str, object]) -> SearchResult:
    """Parse a SearXNG ``default.html`` item into a :class:`SearchResult`."""
    return SearchResult(
        url=_text(item, "url"),
        title=clean_text(_text(item, "title")),
        snippet=clean_text(_text(item, "content")),
    )


def _searxng_files(
    item: dict[str, object],
) -> FileResult | TorrentResult | SearchResult:
    """Dispatch a ``files`` item by ``template`` to its file/torrent/web reader."""
    template = _text(item, "template")
    if template == "torrent.html":
        return TorrentResult(
            url=_text(item, "url"),
            title=clean_text(_text(item, "title")),
            snippet=clean_text(_text(item, "content")),
            magnet_url=_text(item, "magnetlink"),
            torrent_url=_text(item, "torrentfile"),
            seed=_count(item.get("seed")),
            leech=_count(item.get("leech")),
            filesize=_text(item, "filesize"),
        )
    if template == "file.html":
        return FileResult(
            url=_text(item, "url"),
            title=clean_text(_text(item, "title")),
            snippet=clean_text(
                _text(item, "abstract") or _text(item, "content"),
            ),
            filename=_text(item, "filename"),
            size=_text(item, "size"),
            mimetype=_text(item, "mimetype"),
            author=_text(item, "author"),
        )
    return _searxng_web(item)


def _strings(item: dict[str, object], key: str) -> tuple[str, ...]:
    """Return the string elements of ``item[key]``, dropping any non-string."""
    values = item.get(key)
    if not isinstance(values, list):
        return ()
    return tuple(
        value for value in cast(list[object], values) if isinstance(value, str)
    )


def _searxng_it(
    item: dict[str, object],
) -> PackageResult | CodeResult | SearchResult:
    """Dispatch an ``it`` item by ``template`` to its package/code/web reader."""
    template = _text(item, "template")
    if template == "packages.html":
        return PackageResult(
            url=_text(item, "url"),
            title=clean_text(_text(item, "title")),
            snippet=clean_text(_text(item, "content")),
            package_name=_text(item, "package_name"),
            version=_text(item, "version"),
            maintainer=_text(item, "maintainer"),
            license_name=_text(item, "license_name"),
            homepage=_text(item, "homepage"),
            source_code_url=_text(item, "source_code_url"),
            popularity=_text(item, "popularity"),
            tags=_strings(item, "tags"),
        )
    if template == "code.html":
        return CodeResult(
            url=_text(item, "url"),
            title=clean_text(_text(item, "title")),
            snippet=clean_text(_text(item, "content")),
            repository=_text(item, "repository"),
            filename=_text(item, "filename"),
            code_language=_text(item, "code_language"),
        )
    return _searxng_web(item)


def _searxng_media(item: dict[str, object]) -> MediaResult:
    """Parse a ``news``/``music`` ``default.html`` item into a :class:`MediaResult`."""
    return MediaResult(
        url=_text(item, "url"),
        title=clean_text(_text(item, "title")),
        snippet=clean_text(_text(item, "content")),
        published=_published(item),
        audio_url=_text(item, "audio_src"),
        iframe_url=_text(item, "iframe_src"),
        length=_text(item, "length"),
        thumbnail_url=_text(item, "thumbnail"),
    )


def _searxng_paper(item: dict[str, object]) -> PaperResult:
    """Parse a SearXNG ``paper.html`` item into a :class:`PaperResult`."""
    # Citation count is the leading integer of the humanized ``comments`` ("42
    # citations from ..."); unparseable yields None, not zero. The pattern requires
    # a leading digit (a bare "," would crash int()), caps the run at 24 chars
    # (int() rejects >4300 digits), and refuses a longer run rather than reading
    # its prefix as a count.
    cites = re.match(
        r"^\s*(\d[\d,]{0,23})(?![\d,])",
        _text(item, "comments"),
    )
    return PaperResult(
        url=_text(item, "url"),
        title=clean_text(_text(item, "title")),
        snippet=clean_text(_text(item, "content")),
        authors=_strings(item, "authors"),
        journal=clean_text(_text(item, "journal")),
        doi=_text(item, "doi"),
        pdf_url=_text(item, "pdf_url"),
        published=_published(item),
        tags=_strings(item, "tags"),
        citations=int(cites.group(1).replace(",", "")) if cites else None,
    )


def _string_map(value: object) -> dict[str, str]:
    """Return the string-to-string entries of a JSON object, else ``{}``."""
    if not isinstance(value, dict):
        return {}
    return {
        key: entry
        for key, entry in cast(dict[object, object], value).items()
        if isinstance(key, str) and isinstance(entry, str)
    }


# One record per member of ``SearxngCategory``; the check below enforces it.
CATEGORIES: Mapping[SearxngCategory, CategoryInfo] = {
    "general": CategoryInfo(gloss="web results"),
    "images": CategoryInfo(
        gloss="image URL, resolution, format, source",
        parser=lambda item: ImageResult(
            url=_text(item, "url"),
            title=clean_text(_text(item, "title")),
            snippet=clean_text(_text(item, "content")),
            image_url=_text(item, "img_src"),
            thumbnail_url=_text(item, "thumbnail_src"),
            resolution=_text(item, "resolution"),
            img_format=_text(item, "img_format"),
            source=_text(item, "source"),
            filesize=_text(item, "filesize"),
        ),
    ),
    "videos": CategoryInfo(
        gloss="duration, view count, channel, embed URL",
        parser=lambda item: VideoResult(
            url=_text(item, "url"),
            title=clean_text(_text(item, "title")),
            snippet=clean_text(_text(item, "content")),
            published=_published(item),
            iframe_url=_text(item, "iframe_src"),
            length=_text(item, "length"),
            thumbnail_url=_text(item, "thumbnail"),
            views=_text(item, "views"),
            author=_text(item, "author"),
        ),
    ),
    "news": CategoryInfo(gloss="web results with publish date", parser=_searxng_media),
    "map": CategoryInfo(
        gloss="places with coordinates and structured address",
        parser=lambda item: MapResult(
            url=_text(item, "url"),
            title=clean_text(_text(item, "title")),
            snippet=clean_text(_text(item, "content")),
            latitude=_coordinate(item.get("latitude")),
            longitude=_coordinate(item.get("longitude")),
            address=MappingProxyType(_string_map(item.get("address"))),
        ),
    ),
    "music": CategoryInfo(
        gloss="tracks with audio/embed URL and duration",
        parser=_searxng_media,
    ),
    "it": CategoryInfo(
        gloss="packages (name/version/license/homepage), repos, code",
        parser=_searxng_it,
    ),
    "science": CategoryInfo(
        gloss="papers with authors/DOI/citations",
        parser=_searxng_paper,
    ),
    "files": CategoryInfo(
        gloss="files (filename/size/type) and torrents",
        parser=_searxng_files,
    ),
    "social media": CategoryInfo(gloss="posts from Mastodon/Lemmy (web results)"),
}


@overload
def searxng(
    query: str,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["science"],
    timeout_sec: float = ...,
    connect_timeout_sec: float = ...,
    retries: int = ...,
    transport: Transport = ...,
    session: FetchSession | None = ...,
) -> Sequence[PaperResult]: ...


@overload
def searxng(
    query: str,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["images"],
    timeout_sec: float = ...,
    connect_timeout_sec: float = ...,
    retries: int = ...,
    transport: Transport = ...,
    session: FetchSession | None = ...,
) -> Sequence[ImageResult]: ...


@overload
def searxng(
    query: str,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["videos"],
    timeout_sec: float = ...,
    connect_timeout_sec: float = ...,
    retries: int = ...,
    transport: Transport = ...,
    session: FetchSession | None = ...,
) -> Sequence[VideoResult]: ...


@overload
def searxng(
    query: str,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["news", "music"],
    timeout_sec: float = ...,
    connect_timeout_sec: float = ...,
    retries: int = ...,
    transport: Transport = ...,
    session: FetchSession | None = ...,
) -> Sequence[MediaResult]: ...


@overload
def searxng(
    query: str,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["map"],
    timeout_sec: float = ...,
    connect_timeout_sec: float = ...,
    retries: int = ...,
    transport: Transport = ...,
    session: FetchSession | None = ...,
) -> Sequence[MapResult]: ...


@overload
def searxng(
    query: str,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["it"],
    timeout_sec: float = ...,
    connect_timeout_sec: float = ...,
    retries: int = ...,
    transport: Transport = ...,
    session: FetchSession | None = ...,
) -> Sequence[PackageResult | CodeResult | SearchResult]: ...


@overload
def searxng(
    query: str,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["files"],
    timeout_sec: float = ...,
    connect_timeout_sec: float = ...,
    retries: int = ...,
    transport: Transport = ...,
    session: FetchSession | None = ...,
) -> Sequence[FileResult | TorrentResult | SearchResult]: ...


@overload
def searxng(
    query: str,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: SearxngCategory = ...,
    timeout_sec: float = ...,
    connect_timeout_sec: float = ...,
    retries: int = ...,
    transport: Transport = ...,
    session: FetchSession | None = ...,
) -> Sequence[SearchResult]: ...


def searxng(
    query: str,
    num_results: int = 10,
    headers: dict[str, str] | None = None,
    *,
    categories: SearxngCategory = "general",
    timeout_sec: float = 15.0,
    connect_timeout_sec: float = 3.0,
    retries: int = 1,
    transport: Transport = "auto",
    session: FetchSession | None = None,
) -> Sequence[SearxngResult]:
    """Query a SearXNG instance and return parsed, typed JSON results.

    The return shape follows ``categories``: each tab whose engines emit a
    structured result template gets a richer :class:`SearchResult` subclass
    (``science`` -> :class:`PaperResult`, ``images`` -> :class:`ImageResult`,
    ``videos`` -> :class:`VideoResult`, ``news``/``music`` -> :class:`MediaResult`,
    ``map`` -> :class:`MapResult`, ``it`` -> :class:`PackageResult` /
    :class:`CodeResult`, ``files`` -> :class:`FileResult` /
    :class:`TorrentResult`). Categories with no extra structure return plain
    :class:`SearchResult`. Every subclass keeps ``url``/``title``/``snippet``
    populated. A malformed result degrades to fewer or emptier records.

    Args:
      query: Search query string.
      num_results: Maximum results to return.
      headers: Optional override headers forwarded to fetch.
      categories: SearXNG result category (tab) to query.
      timeout_sec: HTTP ceiling per attempt. It must clear SearXNG's slowest
        upstream engine, since the instance answers only after aggregating.
      connect_timeout_sec: Ceiling on the handshake to the instance alone.
      retries: Retry attempts for a transient failure, including a 429 with
        Retry-After. A failing request costs up to ``(retries + 1) *
        timeout_sec``.
      transport: Retrieval transport; ``"auto"`` applies domain routing.
      session: Connections to reuse; ``None`` uses the process's shared session.

    Returns:
      results: One typed record per hit -- a :class:`SearchResult` or a
        category-specific subclass of it.

    Raises:
      ValueError: If ``num_results`` is negative.
      SearchError: If ``SEARXNG_URL`` is unset or the body is not valid JSON.

    """
    if num_results < 0:
        raise ValueError(f"'num_results' must be >= 0, got {num_results}.")
    if num_results == 0:
        return list[SearxngResult]()
    base_url = _searxng_url()
    params = urlencode(
        {"q": query, "format": "json", "pageno": "1", "categories": categories},
    )
    body = fetch(
        f"{base_url}/search?{params}",
        request=RequestParams(
            content=ContentParams(headers=headers or {}),
            retry=RetryParams(
                retries=retries,
                timeout_sec=timeout_sec,
                connect_timeout_sec=connect_timeout_sec,
            ),
            # SEARXNG_URL is an instance the operator runs, often on loopback.
            policy=PolicyParams(transport=transport, allow_private_hosts=True),
        ),
        session=session,
    )
    # Decoded explicitly so invalid UTF-8 surfaces as SearchError, which the
    # dispatcher catches, rather than as UnicodeDecodeError, which it does not.
    try:
        text = body.decode()
    except UnicodeDecodeError as e:
        raise SearchError(f"searxng returned undecodable bytes: {e}") from e
    # An intermediary (e.g. a Cloudflare rate-limit page) can answer with HTTP 200
    # and a non-JSON body; name it rather than surface a bare JSONDecodeError.
    try:
        payload = json_loads(text)
    except json.JSONDecodeError as e:
        raise SearchError(f"searxng returned {_describe_non_json(text)}") from e
    results = payload.get("results") if isinstance(payload, dict) else None
    items = [
        from_plain(item, dict[str, object])
        for item in (results if isinstance(results, list) else [])
        if isinstance(item, dict)
    ]
    parse = category_parser(categories)
    return [parse(item) for item in items[:num_results]]


# At import, not in a test: a tab missing a record fails before any query.
if set(CATEGORIES) != set(get_args(SearxngCategory.__value__)):
    raise ValueError("every SearxngCategory needs a CATEGORIES record")


def category_parser(
    category: SearxngCategory,
) -> Callable[[dict[str, object]], SearxngResult]:
    """Return the reader for ``category``, or the generic web reader."""
    return CATEGORIES[category].parser or _searxng_web


def category_gloss() -> str:
    """Render every tab as one Markdown bullet line: ``- `name` -- gloss.``."""
    # Not ``get_args(SearxngCategory)``: on a PEP 695 alias it returns nothing.
    return "\n".join(
        f"  - `{name}` -- {info.gloss}." for name, info in CATEGORIES.items()
    )


def _searxng_url() -> str:
    """Return the configured SearXNG base URL without a trailing slash."""
    url = os.environ.get("SEARXNG_URL", "").rstrip("/")
    if not url:
        raise SearchError("SEARXNG_URL must be set to use SearXNG search")
    return url


def _describe_non_json(text: str) -> str:
    """Name what a non-JSON body is (rate-limit page, HTML, other) for an error."""
    # Whole body, not a head slice: the HTML variant of the 1015 page carries the
    # code far past any fixed prefix.
    lowered = text.lower()
    if "error code: 1015" in lowered or "rate limited" in lowered:
        return (
            "a rate-limit page instead of JSON (Cloudflare error 1015). The "
            "edge in front of the instance is throttling this egress IP -- "
            "slow the query rate or retry later; the instance itself is "
            "healthy and never saw the request."
        )
    if lowered.lstrip().startswith(("<!doctype", "<html", "<?xml")):
        return (
            "an HTML page instead of JSON, so an intermediary answered rather "
            f"than SearXNG: {text.strip()[:200]!r}"
        )
    return f"a body that is not JSON: {text.strip()[:200]!r}"
