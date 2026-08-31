"""The SearXNG backend: a self-hosted metasearch instance over many engines.

Reached at ``SEARXNG_URL``; returns JSON, so this is a mapping layer rather
than a scraper. Its per-category result shapes are what make the category
overloads in :mod:`wesearch.search.search` meaningful.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, get_args, overload
from urllib.parse import urlencode

import json
import logging
import math
import os
import re

from wesearch.fetch import (
    ContentParams,
    PolicyParams,
    RequestParams,
    RetryParams,
    Transport,
    fetch,
)
from wesearch.lib.custom_json import (
    DatetimeCodec,
    DictCodec,
    ListCodec,
    StrCodec,
    decode_or_none,
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
    SearxngCategory,
    TorrentResult,
    VideoResult,
    clean_text,
)


logger = logging.getLogger(__name__)


_SEARXNG_URL_ENV: Final = "SEARXNG_URL"


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
    retries: int = ...,
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
    retries: int = ...,
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
    retries: int = ...,
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
    retries: int = ...,
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
    retries: int = ...,
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
    retries: int = ...,
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
    retries: int = ...,
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
    retries: int = ...,
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
    retries: int = 1,
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
      retries: Retry attempts for a transient failure. Defaults to 1 because the
        edge in front of an instance rate-limits a burst with a 429 +
        Retry-After, which the retry loop already honors; without a budget a
        throttled burst surfaces as a hard failure the advertised wait would
        have cleared. Exposed, like ``duckduckgo``'s, because it MULTIPLIES
        ``timeout_sec``: a caller that lowered the ceiling to bound a wedged
        egress otherwise still pays ``(retries + 1)`` times what it asked for.
      transport: Retrieval transport; ``"auto"`` applies domain routing.

    Returns:
      results: One typed record per hit -- a :class:`SearchResult` or a
        category-specific subclass of it.

    """
    if num_results < 0:
        raise ValueError(f"'num_results' must be >= 0, got {num_results}.")
    if num_results == 0:
        return list[SearxngResult]()  # No results wanted; skip the round-trip.
    base_url = _searxng_url()
    params = urlencode(
        {"q": query, "format": "json", "pageno": "1", "categories": categories}
    )
    body, _ = fetch(
        f"{base_url}/search?{params}",
        request=RequestParams(
            content=ContentParams(headers=headers),
            retry=RetryParams(
                retries=retries,
                timeout_sec=timeout_sec,
                connect_timeout_sec=connect_timeout_sec,
            ),
            # trust="internal": SEARXNG_URL names an instance the OPERATOR
            # runs, which is exactly the private-network case the untrusted
            # default refuses -- a loopback instance failed with "Refusing to
            # fetch '127.0.0.1' (resolves to non-public address)".
            policy=PolicyParams(transport=transport, trust="internal"),
        ),
    )
    # Decoded explicitly: json.loads on raw bytes raises UnicodeDecodeError for
    # invalid UTF-8, which is not in the dispatcher's except tuple and would
    # escape as itself rather than as SearchError.
    try:
        text = body.decode()
    except UnicodeDecodeError as e:
        raise SearchError(f"searxng returned undecodable bytes: {e}") from e
    # We asked for format=json, so anything else is an intermediary answering
    # instead of SearXNG -- and it does NOT arrive as an error status. A
    # Cloudflare-fronted instance serves its rate-limit interstitial ("error
    # code: 1015") as HTTP 200 with an HTML body, so ``fetch`` raises nothing
    # and the page reaches json.loads as a bare "line 1 column 1 (char 0)"
    # that names neither the backend nor the cause.
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise SearchError(f"searxng returned {_describe_non_json(text)}") from e
    items = ListCodec.coerce(DictCodec.coerce(payload).get("results"))
    parse = category_parser(categories)
    return [parse(DictCodec.coerce(item)) for item in items[:num_results]]


def _describe_non_json(text: str) -> str:
    """Name what a non-JSON body actually is, for the error message.

    An operator reading "not JSON" still has to go find out who answered and
    why. The two cases that occur in practice -- a rate-limit interstitial and
    some other HTML error page -- are distinguishable from the body, and
    naming them turns the failure into an instruction.
    """
    # Whole body, not a head slice: the live 1015 response is the bare 17-byte
    # string "error code: 1015", while the HTML variant carries the code far
    # past any fixed prefix window. A slice classified both as generic HTML.
    lowered = text.lower()
    # Cloudflare's own code for "rate limited"; nothing else emits it.
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


def _searxng_web(item: dict[str, object]) -> SearchResult:
    """Parse a SearXNG ``default.html`` item into a :class:`SearchResult`."""
    return SearchResult(
        url=StrCodec.coerce(item.get("url")),
        title=clean_text(StrCodec.coerce(item.get("title"))),
        snippet=clean_text(StrCodec.coerce(item.get("content"))),
    )


def _searxng_image(item: dict[str, object]) -> ImageResult:
    """Parse a SearXNG ``images.html`` item into an :class:`ImageResult`."""
    return ImageResult(
        url=StrCodec.coerce(item.get("url")),
        title=clean_text(StrCodec.coerce(item.get("title"))),
        snippet=clean_text(StrCodec.coerce(item.get("content"))),
        image_url=StrCodec.coerce(item.get("img_src")),
        thumbnail_url=StrCodec.coerce(item.get("thumbnail_src")),
        resolution=StrCodec.coerce(item.get("resolution")),
        img_format=StrCodec.coerce(item.get("img_format")),
        source=StrCodec.coerce(item.get("source")),
        filesize=StrCodec.coerce(item.get("filesize")),
    )


def _searxng_video(item: dict[str, object]) -> VideoResult:
    """Parse a SearXNG ``videos.html`` item into a :class:`VideoResult`."""
    return VideoResult(
        url=StrCodec.coerce(item.get("url")),
        title=clean_text(StrCodec.coerce(item.get("title"))),
        snippet=clean_text(StrCodec.coerce(item.get("content"))),
        published=DatetimeCodec.coerce(item.get("publishedDate")),
        iframe_url=StrCodec.coerce(item.get("iframe_src")),
        length=StrCodec.coerce(item.get("length")),
        thumbnail_url=StrCodec.coerce(item.get("thumbnail")),
        views=StrCodec.coerce(item.get("views")),
        author=StrCodec.coerce(item.get("author")),
    )


def _searxng_media(item: dict[str, object]) -> MediaResult:
    """Parse a ``news``/``music`` ``default.html`` item into a :class:`MediaResult`."""
    return MediaResult(
        url=StrCodec.coerce(item.get("url")),
        title=clean_text(StrCodec.coerce(item.get("title"))),
        snippet=clean_text(StrCodec.coerce(item.get("content"))),
        published=DatetimeCodec.coerce(item.get("publishedDate")),
        audio_url=StrCodec.coerce(item.get("audio_src")),
        iframe_url=StrCodec.coerce(item.get("iframe_src")),
        length=StrCodec.coerce(item.get("length")),
        thumbnail_url=StrCodec.coerce(item.get("thumbnail")),
    )


def _searxng_map(item: dict[str, object]) -> MapResult:
    """Parse a SearXNG ``map.html`` item into a :class:`MapResult`."""
    return MapResult(
        url=StrCodec.coerce(item.get("url")),
        title=clean_text(StrCodec.coerce(item.get("title"))),
        snippet=clean_text(StrCodec.coerce(item.get("content"))),
        latitude=_coordinate(item.get("latitude")),
        longitude=_coordinate(item.get("longitude")),
        address=MappingProxyType(DictCodec.coerce(item.get("address"), str)),
    )


def _coordinate(value: object) -> float | None:
    """Return one finite map coordinate, or None when unknown."""
    coordinate = decode_or_none(float, value)
    return coordinate if coordinate is not None and math.isfinite(coordinate) else None


def _searxng_it(item: dict[str, object]) -> PackageResult | CodeResult | SearchResult:
    """Dispatch an ``it`` item by ``template`` to its package/code/web reader."""
    template = StrCodec.coerce(item.get("template"))
    if template == "packages.html":
        return _searxng_package(item)
    if template == "code.html":
        return _searxng_code(item)
    return _searxng_web(item)


def _searxng_package(item: dict[str, object]) -> PackageResult:
    """Parse a SearXNG ``packages.html`` item into a :class:`PackageResult`."""
    return PackageResult(
        url=StrCodec.coerce(item.get("url")),
        title=clean_text(StrCodec.coerce(item.get("title"))),
        snippet=clean_text(StrCodec.coerce(item.get("content"))),
        package_name=StrCodec.coerce(item.get("package_name")),
        version=StrCodec.coerce(item.get("version")),
        maintainer=StrCodec.coerce(item.get("maintainer")),
        license_name=StrCodec.coerce(item.get("license_name")),
        homepage=StrCodec.coerce(item.get("homepage")),
        source_code_url=StrCodec.coerce(item.get("source_code_url")),
        popularity=StrCodec.coerce(item.get("popularity")),
        tags=tuple(ListCodec.coerce(item.get("tags"), str)),
    )


def _searxng_code(item: dict[str, object]) -> CodeResult:
    """Parse a SearXNG ``code.html`` item into a :class:`CodeResult`."""
    return CodeResult(
        url=StrCodec.coerce(item.get("url")),
        title=clean_text(StrCodec.coerce(item.get("title"))),
        snippet=clean_text(StrCodec.coerce(item.get("content"))),
        repository=StrCodec.coerce(item.get("repository")),
        filename=StrCodec.coerce(item.get("filename")),
        code_language=StrCodec.coerce(item.get("code_language")),
    )


def _searxng_files(
    item: dict[str, object],
) -> FileResult | TorrentResult | SearchResult:
    """Dispatch a ``files`` item by ``template`` to its file/torrent/web reader."""
    template = StrCodec.coerce(item.get("template"))
    if template == "torrent.html":
        return _searxng_torrent(item)
    if template == "file.html":
        return _searxng_file(item)
    return _searxng_web(item)


def _searxng_file(item: dict[str, object]) -> FileResult:
    """Parse a SearXNG ``file.html`` item into a :class:`FileResult`."""
    return FileResult(
        url=StrCodec.coerce(item.get("url")),
        title=clean_text(StrCodec.coerce(item.get("title"))),
        snippet=clean_text(
            StrCodec.coerce(item.get("abstract"))
            or StrCodec.coerce(item.get("content"))
        ),
        filename=StrCodec.coerce(item.get("filename")),
        size=StrCodec.coerce(item.get("size")),
        mimetype=StrCodec.coerce(item.get("mimetype")),
        author=StrCodec.coerce(item.get("author")),
    )


def _searxng_torrent(item: dict[str, object]) -> TorrentResult:
    """Parse a SearXNG ``torrent.html`` item into a :class:`TorrentResult`."""
    return TorrentResult(
        url=StrCodec.coerce(item.get("url")),
        title=clean_text(StrCodec.coerce(item.get("title"))),
        snippet=clean_text(StrCodec.coerce(item.get("content"))),
        magnet_url=StrCodec.coerce(item.get("magnetlink")),
        torrent_url=StrCodec.coerce(item.get("torrentfile")),
        seed=decode_or_none(int, item.get("seed")),
        leech=decode_or_none(int, item.get("leech")),
        filesize=StrCodec.coerce(item.get("filesize")),
    )


# Leading integer of SearXNG's humanized citation ``comments`` (e.g. "42
# citations from the year 2019 to 2024" -> 42). No engine reports the count as a
# structured integer, so this is the only recovery path; an unparseable comment
# yields ``None`` rather than a fabricated zero.
# At least one DIGIT, not just comma-ish characters: the looser `[\d,]+`
# matched a bare "," on hostile JSON, and the int() below then raised
# ValueError out of a parse that must degrade to "unknown citations".
# BOUNDED, because format is not magnitude: the unbounded run still matched, and
# CPython refuses int() on a string past 4300 digits -- the same ValueError, out
# of the same parse, on the same hostile-comments input the DIGIT fix addressed.
# A trailing guard so a longer run does not match its own prefix: past ~24
# digits the field is not a count at all, and reading its first 24 would
# fabricate one.
_CITATIONS_RE = re.compile(r"^\s*(\d[\d,]{0,23})(?![\d,])")


def _searxng_paper(item: dict[str, object]) -> PaperResult:
    """Parse a SearXNG ``paper.html`` item into a :class:`PaperResult`."""
    cites = _CITATIONS_RE.match(StrCodec.coerce(item.get("comments")))
    return PaperResult(
        url=StrCodec.coerce(item.get("url")),
        title=clean_text(StrCodec.coerce(item.get("title"))),
        snippet=clean_text(StrCodec.coerce(item.get("content"))),
        authors=tuple(ListCodec.coerce(item.get("authors"), str)),
        journal=clean_text(StrCodec.coerce(item.get("journal"))),
        doi=StrCodec.coerce(item.get("doi")),
        pdf_url=StrCodec.coerce(item.get("pdf_url")),
        published=DatetimeCodec.coerce(item.get("publishedDate")),
        tags=tuple(ListCodec.coerce(item.get("tags"), str)),
        citations=int(cites.group(1).replace(",", "")) if cites else None,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CategoryInfo:
    """What a tab carries beyond its name: prose and a result parser.

    Attributes:
      gloss: One phrase naming what the tab returns, rendered into every tool
        description. Written here rather than in each adapter's prose because
        a hand-copied list drifts: both surfaces omitted ``social media`` for
        as long as the tab has existed.
      parser: Reader for the tab's result template. ``None`` means the tab has
        no structured template and reads through the generic web parser --
        which is what ``general`` and ``social media`` are.

    """

    gloss: str
    parser: Callable[[dict[str, object]], SearxngResult] | None = None


# One record per member of ``SearxngCategory``. The Literal in
# :mod:`.custom_types` stays the only place a tab NAME is spelled; this attaches
# data to those names, and the assertion below makes a missing record an
# ImportError rather than a tab that is silently unglossed and unparsed.
CATEGORIES: Mapping[SearxngCategory, CategoryInfo] = {
    "general": CategoryInfo(gloss="web results"),
    "images": CategoryInfo(
        gloss="image URL, resolution, format, source", parser=_searxng_image
    ),
    "videos": CategoryInfo(
        gloss="duration, view count, channel, embed URL", parser=_searxng_video
    ),
    "news": CategoryInfo(gloss="web results with publish date", parser=_searxng_media),
    "map": CategoryInfo(
        gloss="places with coordinates and structured address", parser=_searxng_map
    ),
    "music": CategoryInfo(
        gloss="tracks with audio/embed URL and duration", parser=_searxng_media
    ),
    "it": CategoryInfo(
        gloss="packages (name/version/license/homepage), repos, code",
        parser=_searxng_it,
    ),
    "science": CategoryInfo(
        gloss="papers with authors/DOI/citations", parser=_searxng_paper
    ),
    "files": CategoryInfo(
        gloss="files (filename/size/type) and torrents", parser=_searxng_files
    ),
    "social media": CategoryInfo(gloss="posts from Mastodon/Lemmy (web results)"),
}

# Not a test: a tab added to the Literal without a record here must fail at
# IMPORT, where the author sees it, rather than at the first query for that tab.
assert set(CATEGORIES) == set(get_args(SearxngCategory.__value__)), (
    "every SearxngCategory needs a CATEGORIES record"
)


def category_parser(
    category: SearxngCategory,
) -> Callable[[dict[str, object]], SearxngResult]:
    """Return the reader for ``category``, or the generic web reader."""
    return CATEGORIES[category].parser or _searxng_web


def category_gloss() -> str:
    """Render every tab as one description line; no hand-copied list."""
    return "\n".join(
        f"  - `{name}` -- {CATEGORIES[name].gloss}."
        for name in get_args(SearxngCategory.__value__)
    )


def _searxng_url() -> str:
    """Return the configured SearXNG base URL without a trailing slash."""
    url = os.environ.get(_SEARXNG_URL_ENV, "").rstrip("/")
    if not url:
        raise SearchError(
            f"{_SEARXNG_URL_ENV} must be set to use SearXNG search",
        )
    return url
