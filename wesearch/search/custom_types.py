"""Result records and the shared vocabulary every search backend returns.

The typed shapes a backend maps its wire payload onto, the vocabularies that
name a backend and a category, and the helpers that mapping needs. Depends on
no backend, so a caller can name a result type or a category without importing
the machinery that produces one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

import hashlib
import re

from wesearch.chrome.useragents import user_agent_pool


if TYPE_CHECKING:
    import bs4
else:
    from wrapt import lazy_import

    bs4 = lazy_import("bs4")


# Internal builds keep extra backends; public exports keep only DuckDuckGo.
_BACKEND_NAMES = Literal["duckduckgo", "searxng"]
DEFAULT_SEARCH_BACKEND: Final[SearchBackends] = "duckduckgo"
SearchBackends: TypeAlias = _BACKEND_NAMES  # noqa: UP040 -- type keyword breaks get_args() at runtime


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


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchResult:
    """A single web search result."""

    url: str
    title: str
    snippet: str


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperResult(SearchResult):
    """A scholarly result from SearXNG's ``science`` category (``paper.html``).

    Subclasses :class:`SearchResult` so any web-result consumer works
    unchanged: ``url``/``title`` carry through and ``snippet`` holds the
    abstract. The added fields surface the structured bibliographic metadata
    SearXNG's science engines (Semantic Scholar, OpenAlex, PubMed, arXiv,
    Crossref, ...) emit. Fields absent from a given engine's response default
    to empty.

    Attributes:
      citations: Citation count when the engine reported one. SearXNG renders
        it as a humanized string in the ``comments`` field (e.g. ``"42
        citations"``); the integer is recovered when parseable, else ``None``.

    """

    authors: tuple[str, ...] = ()
    journal: str = ""
    doi: str = ""
    pdf_url: str = ""
    published: datetime | None = None
    tags: tuple[str, ...] = ()
    citations: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageResult(SearchResult):
    """An image result from SearXNG's ``images`` category (``images.html``).

    ``url`` is the source page; ``image_url`` the full image. ``snippet`` holds
    any caption. Fields absent from a given engine default to empty.
    """

    image_url: str = ""
    thumbnail_url: str = ""
    resolution: str = ""
    img_format: str = ""
    source: str = ""
    filesize: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class MediaResult(SearchResult):
    """A time-based media result -- the ``news`` and ``music`` categories.

    Base for any result with a temporal/playable payload: a publish date, an
    embed or audio URL, a duration, a thumbnail. The ``news`` and ``music``
    tabs emit the generic ``default.html`` template but carry these fields,
    which a bare :class:`SearchResult` drops. :class:`VideoResult` extends this
    with view count and channel. Fields absent from a given engine default to
    empty.
    """

    published: datetime | None = None
    audio_url: str = ""
    iframe_url: str = ""
    length: str = ""
    thumbnail_url: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoResult(MediaResult):
    """A video result from SearXNG's ``videos`` category (``videos.html``).

    Extends :class:`MediaResult` (shared publish date, embed URL, duration,
    thumbnail) with the video-specific view count and channel. The
    ``videos.html`` template adds no fields beyond SearXNG's result base, but
    those base fields are real structure a web result discards. Fields absent
    from a given engine default to empty.
    """

    views: str = ""
    author: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class MapResult(SearchResult):
    """A place result from SearXNG's ``map`` category (``map.html``).

    ``snippet`` holds any description. Coordinates and the structured address
    surface the geographic payload. Fields absent from a given engine default
    to empty / ``None``.

    Attributes:
      address: Structured address components keyed by SearXNG's field names
        (``name``, ``road``, ``house_number``, ``locality``, ``postcode``,
        ``country``); empty when the engine returned none.

    """

    latitude: float | None = None
    longitude: float | None = None
    address: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True, slots=True, kw_only=True)
class PackageResult(SearchResult):
    """A software-package result from the ``it`` category (``packages.html``).

    ``snippet`` holds the package description. Fields absent from a given
    engine default to empty.
    """

    package_name: str = ""
    version: str = ""
    maintainer: str = ""
    license_name: str = ""
    homepage: str = ""
    source_code_url: str = ""
    popularity: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class CodeResult(SearchResult):
    """A source-code result from the ``it`` category (``code.html``).

    ``snippet`` holds the matched code or description. Fields absent from a
    given engine default to empty.
    """

    repository: str = ""
    filename: str = ""
    code_language: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class FileResult(SearchResult):
    """A file result from SearXNG's ``files`` category (``file.html``).

    ``snippet`` holds any abstract. Fields absent from a given engine default
    to empty.
    """

    filename: str = ""
    size: str = ""
    mimetype: str = ""
    author: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class TorrentResult(SearchResult):
    """A torrent result from SearXNG's ``files`` category (``torrent.html``).

    Fields absent from a given engine default to empty / ``None``.
    """

    magnet_url: str = ""
    torrent_url: str = ""
    seed: int | None = None
    leech: int | None = None
    filesize: str = ""


class SearchError(RuntimeError):
    """Raised when a search backend fails before returning results."""


_CLEAN_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")


def strip_scripts(tag: bs4.Tag | bs4.BeautifulSoup) -> None:
    """Remove all ``<script>`` elements from the tree in place."""
    for script in tag.find_all("script"):
        script.decompose()


def clean_text(text: str) -> str:
    """Collapse whitespace runs and drop spaces before punctuation.

    Args:
      text: The raw scraped text to normalize.

    Returns:
      cleaned: ``text`` with whitespace runs collapsed and pre-punctuation
        spaces removed.

    """
    return _CLEAN_SPACE_BEFORE_PUNCT.sub(r"\1", " ".join(text.split()))


def gsa_headers_for_query(query: str) -> dict[str, str]:
    """Build request headers with a query-stable Android-Chrome UA.

    Known coherence debt (REV2A-001): this Android/mobile UA is paired with
    ``fetch``'s default ``impersonate="chrome"`` (a DESKTOP TLS/JA4/HTTP2
    fingerprint + desktop ``sec-ch-ua-mobile: ?0`` / ``sec-ch-ua-platform:
    "macOS"``), so the UA and the wire fingerprint disagree -- normally a bot
    tell. A coherent fix (``impersonate="chrome_android"`` + mobile hints) is
    DEFERRED and UNVERIFIED: Google now JS-gates HTML scraping (the enablejs
    shell) independently of the fingerprint -- a coherent identity did NOT clear
    the wall in offline testing -- so a fingerprint change is unverifiable and
    likely valueless until the JS gate is addressed. Left as-is deliberately.

    Args:
      query: The search query; hashed to pick a stable User-Agent per query.

    Returns:
      headers: A one-entry ``User-Agent`` header dict.

    """
    pool = user_agent_pool("chrome_android")
    idx = int.from_bytes(hashlib.sha256(query.encode()).digest()[:8]) % len(pool)
    return {"User-Agent": f"{pool[idx]} NSTNWV"}
