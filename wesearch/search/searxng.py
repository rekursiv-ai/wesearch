"""The SearXNG backend: a self-hosted metasearch instance over many engines.

Reached at ``SEARXNG_URL``; returns JSON, so this is a mapping layer rather
than a scraper. Its per-category result shapes are what make the category
overloads in :mod:`wesearch.search.search` meaningful.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal, overload
from urllib.parse import urlencode

import json
import logging
import os
import re

from wesearch.fetch import Content, Policy, RequestParams, Retry, Transport, fetch
from wesearch.lib.custom_json import (
    datetime_val,
    dict_val,
    float_val,
    int_val,
    list_val,
    str_val,
)
from wesearch.search.custom_types import (
    CodeResult,
    FileResult,
    ImageResult,
    MapResult,
    MediaResult,
    PackageResult,
    PaperResult,
    SearchError,
    SearchResult,
    TorrentResult,
    VideoResult,
    clean_text,
)


logger = logging.getLogger(__name__)


_SEARXNG_URL_ENV: Final = "SEARXNG_URL"


type SearxngCategory = Literal[
    "general",
    "images",
    "videos",
    "news",
    "map",
    "music",
    "it",
    "science",
    "files",
    "social media",
]

# Union of every result shape a SearXNG query can return -- the implementation
# return type behind the per-category overloads. ``VideoResult`` is omitted as
# a ``MediaResult`` subclass and the leaf subclasses cover the rest.
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


@overload
def searxng(
    query: str,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["science"],
    timeout_sec: float = ...,
    connect_timeout_sec: float = ...,
    transport: Transport = ...,
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
    transport: Transport = ...,
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
    transport: Transport = ...,
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
    transport: Transport = ...,
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
    transport: Transport = ...,
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
    transport: Transport = ...,
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
    transport: Transport = ...,
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
    transport: Transport = ...,
) -> Sequence[SearchResult]: ...


def searxng(
    query: str,
    num_results: int = 10,
    headers: dict[str, str] | None = None,
    *,
    categories: SearxngCategory = "general",
    timeout_sec: float = 15.0,
    connect_timeout_sec: float = 3.0,
    transport: Transport = "auto",
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
    populated, so a web-result consumer works on any return unchanged. The
    discriminating SearXNG ``template`` field is consumed here per result.

    Args:
      query: Search query string.
      num_results: Maximum results to return.
      headers: Optional override headers forwarded to fetch.
      categories: SearXNG result category (tab) to query.
      timeout_sec: HTTP ceiling. SearXNG fans one query out to several upstream
        engines and returns only once they finish or hit its own per-engine
        timeouts (heavy science engines run to ~30s), so the client ceiling must
        clear the aggregation tail, not a single engine's latency: at 10s the
        multi-engine ``it``/``science`` tabs hit a premature client-side timeout
        mid-aggregation (observed live). 15s clears the common tail while still
        bounding an interactive turn.
      connect_timeout_sec: Ceiling on the handshake alone. Deliberately NOT
        scaled up for the fan-out above: SearXNG queries DuckDuckGo and Google
        server-side, so that cost is a READ on this connection and is already
        covered by ``timeout_sec``. The handshake is to the instance itself --
        one hop, however many engines sit behind it.
      transport: Retrieval transport; ``"auto"`` applies domain routing.

    Returns:
      results: One typed record per hit -- a :class:`SearchResult` or a
        category-specific subclass of it.

    """
    if num_results < 0:
        raise ValueError(f"'num_results' must be >= 0, got {num_results}.")
    if num_results == 0:
        return []  # Nothing to return, so do not pay for a round-trip.
    base_url = _searxng_url()
    params = urlencode(
        {"q": query, "format": "json", "pageno": "1", "categories": categories}
    )
    body, _ = fetch(
        f"{base_url}/search?{params}",
        request=RequestParams(
            content=Content(headers=headers),
            retry=Retry(
                timeout_sec=timeout_sec, connect_timeout_sec=connect_timeout_sec
            ),
            # trust="internal": SEARXNG_URL names an instance the OPERATOR
            # runs, which is exactly the private-network case the untrusted
            # default refuses -- a loopback instance failed with "Refusing to
            # fetch '127.0.0.1' (resolves to non-public address)".
            policy=Policy(transport=transport, trust="internal"),
        ),
    )
    # Decoded explicitly: json.loads on raw bytes raises UnicodeDecodeError for
    # invalid UTF-8, which is not in the dispatcher's except tuple and would
    # escape as itself rather than as SearchError.
    try:
        text = body.decode()
    except UnicodeDecodeError as e:
        raise SearchError(f"searxng returned undecodable bytes: {e}") from e
    # The typed extractors validate-and-narrow in one call, so a malformed
    # payload degrades to an empty result without a cast or an isinstance
    # ladder -- the house rule exists because those casts assert rather than
    # check, and each one was a place a hostile payload could walk through.
    # list_val, not dicts_val: the latter drops an EMPTY object, and a result
    # with no recognized fields is still a result the backend returned.
    items = list_val(dict_val(json.loads(text)).get("results"))
    parse = _SEARXNG_PARSERS.get(categories, _searxng_web)
    return [parse(dict_val(item)) for item in items[:num_results]]


def _searxng_web(item: dict[str, object]) -> SearchResult:
    """Parse a SearXNG ``default.html`` item into a :class:`SearchResult`."""
    return SearchResult(
        url=str_val(item.get("url")),
        title=clean_text(str_val(item.get("title"))),
        snippet=clean_text(str_val(item.get("content"))),
    )


def _searxng_image(item: dict[str, object]) -> ImageResult:
    """Parse a SearXNG ``images.html`` item into an :class:`ImageResult`."""
    return ImageResult(
        url=str_val(item.get("url")),
        title=clean_text(str_val(item.get("title"))),
        snippet=clean_text(str_val(item.get("content"))),
        image_url=str_val(item.get("img_src")),
        thumbnail_url=str_val(item.get("thumbnail_src")),
        resolution=str_val(item.get("resolution")),
        img_format=str_val(item.get("img_format")),
        source=str_val(item.get("source")),
        filesize=str_val(item.get("filesize")),
    )


def _searxng_video(item: dict[str, object]) -> VideoResult:
    """Parse a SearXNG ``videos.html`` item into a :class:`VideoResult`."""
    return VideoResult(
        url=str_val(item.get("url")),
        title=clean_text(str_val(item.get("title"))),
        snippet=clean_text(str_val(item.get("content"))),
        published=datetime_val(item.get("publishedDate")),
        iframe_url=str_val(item.get("iframe_src")),
        length=str_val(item.get("length")),
        thumbnail_url=str_val(item.get("thumbnail")),
        views=str_val(item.get("views")),
        author=str_val(item.get("author")),
    )


def _searxng_media(item: dict[str, object]) -> MediaResult:
    """Parse a ``news``/``music`` ``default.html`` item into a :class:`MediaResult`."""
    return MediaResult(
        url=str_val(item.get("url")),
        title=clean_text(str_val(item.get("title"))),
        snippet=clean_text(str_val(item.get("content"))),
        published=datetime_val(item.get("publishedDate")),
        audio_url=str_val(item.get("audio_src")),
        iframe_url=str_val(item.get("iframe_src")),
        length=str_val(item.get("length")),
        thumbnail_url=str_val(item.get("thumbnail")),
    )


def _searxng_map(item: dict[str, object]) -> MapResult:
    """Parse a SearXNG ``map.html`` item into a :class:`MapResult`."""
    return MapResult(
        url=str_val(item.get("url")),
        title=clean_text(str_val(item.get("title"))),
        snippet=clean_text(str_val(item.get("content"))),
        # Presence of the key is not presence of a NUMBER: a null or malformed
        # value took float_val's 0.0 default, turning "unknown" into the Gulf of
        # Guinea. Absent and unparseable both mean None here.
        latitude=_optional_float(item.get("latitude")),
        longitude=_optional_float(item.get("longitude")),
        address=MappingProxyType(dict_val(item.get("address"), str)),
    )


def _searxng_it(item: dict[str, object]) -> PackageResult | CodeResult | SearchResult:
    """Dispatch an ``it`` item by ``template`` to its package/code/web reader."""
    template = str_val(item.get("template"))
    if template == "packages.html":
        return _searxng_package(item)
    if template == "code.html":
        return _searxng_code(item)
    return _searxng_web(item)


def _searxng_package(item: dict[str, object]) -> PackageResult:
    """Parse a SearXNG ``packages.html`` item into a :class:`PackageResult`."""
    return PackageResult(
        url=str_val(item.get("url")),
        title=clean_text(str_val(item.get("title"))),
        snippet=clean_text(str_val(item.get("content"))),
        package_name=str_val(item.get("package_name")),
        version=str_val(item.get("version")),
        maintainer=str_val(item.get("maintainer")),
        license_name=str_val(item.get("license_name")),
        homepage=str_val(item.get("homepage")),
        source_code_url=str_val(item.get("source_code_url")),
        popularity=str_val(item.get("popularity")),
        tags=tuple(list_val(item.get("tags"), str)),
    )


def _searxng_code(item: dict[str, object]) -> CodeResult:
    """Parse a SearXNG ``code.html`` item into a :class:`CodeResult`."""
    return CodeResult(
        url=str_val(item.get("url")),
        title=clean_text(str_val(item.get("title"))),
        snippet=clean_text(str_val(item.get("content"))),
        repository=str_val(item.get("repository")),
        filename=str_val(item.get("filename")),
        code_language=str_val(item.get("code_language")),
    )


def _searxng_files(
    item: dict[str, object],
) -> FileResult | TorrentResult | SearchResult:
    """Dispatch a ``files`` item by ``template`` to its file/torrent/web reader."""
    template = str_val(item.get("template"))
    if template == "torrent.html":
        return _searxng_torrent(item)
    if template == "file.html":
        return _searxng_file(item)
    return _searxng_web(item)


def _searxng_file(item: dict[str, object]) -> FileResult:
    """Parse a SearXNG ``file.html`` item into a :class:`FileResult`."""
    return FileResult(
        url=str_val(item.get("url")),
        title=clean_text(str_val(item.get("title"))),
        snippet=clean_text(
            str_val(item.get("abstract")) or str_val(item.get("content"))
        ),
        filename=str_val(item.get("filename")),
        size=str_val(item.get("size")),
        mimetype=str_val(item.get("mimetype")),
        author=str_val(item.get("author")),
    )


def _searxng_torrent(item: dict[str, object]) -> TorrentResult:
    """Parse a SearXNG ``torrent.html`` item into a :class:`TorrentResult`."""
    return TorrentResult(
        url=str_val(item.get("url")),
        title=clean_text(str_val(item.get("title"))),
        snippet=clean_text(str_val(item.get("content"))),
        magnet_url=str_val(item.get("magnetlink")),
        torrent_url=str_val(item.get("torrentfile")),
        seed=_optional_int(item.get("seed")),
        leech=_optional_int(item.get("leech")),
        filesize=str_val(item.get("filesize")),
    )


# Leading integer of SearXNG's humanized citation ``comments`` (e.g. "42
# citations from the year 2019 to 2024" -> 42). No engine reports the count as a
# structured integer, so this is the only recovery path; an unparseable comment
# yields ``None`` rather than a fabricated zero.
# At least one DIGIT, not just comma-ish characters: the looser `[\d,]+`
# matched a bare "," on hostile JSON, and the int() below then raised
# ValueError out of a parse that must degrade to "unknown citations".
_CITATIONS_RE = re.compile(r"^\s*(\d[\d,]*)")


def _searxng_paper(item: dict[str, object]) -> PaperResult:
    """Parse a SearXNG ``paper.html`` item into a :class:`PaperResult`."""
    cites = _CITATIONS_RE.match(str_val(item.get("comments")))
    return PaperResult(
        url=str_val(item.get("url")),
        title=clean_text(str_val(item.get("title"))),
        snippet=clean_text(str_val(item.get("content"))),
        authors=tuple(list_val(item.get("authors"), str)),
        journal=clean_text(str_val(item.get("journal"))),
        doi=str_val(item.get("doi")),
        pdf_url=str_val(item.get("pdf_url")),
        published=datetime_val(item.get("publishedDate")),
        tags=tuple(list_val(item.get("tags"), str)),
        citations=int(cites.group(1).replace(",", "")) if cites else None,
    )


# Per-category result parser. A category absent here falls back to the generic
# web reader (``general`` and the structure-free ``social media`` tab). Each
# parser returns a ``SearchResult`` or a subclass of it, so the table's value
# type stays uniform while ``searxng``'s overloads narrow the element type.
_SEARXNG_PARSERS: Mapping[
    SearxngCategory, Callable[[dict[str, object]], SearxngResult]
] = {
    "science": _searxng_paper,
    "images": _searxng_image,
    "videos": _searxng_video,
    "news": _searxng_media,
    "music": _searxng_media,
    "map": _searxng_map,
    "it": _searxng_it,
    "files": _searxng_files,
}


def _searxng_url() -> str:
    """Return the configured SearXNG base URL without a trailing slash."""
    url = os.environ.get(_SEARXNG_URL_ENV, "").rstrip("/")
    if not url:
        raise SearchError(
            f"{_SEARXNG_URL_ENV} must be set to use SearXNG search",
        )
    return url


def _optional_float(value: object) -> float | None:
    """Return ``value`` as a float, or ``None`` when it is absent or unparseable."""
    return float_val(value) if isinstance(value, (int, float)) else None


def _optional_int(value: object) -> int | None:
    """Return ``value`` as an int, or ``None`` when it is absent or unparseable."""
    return int_val(value, 0) if isinstance(value, (int, float)) else None
