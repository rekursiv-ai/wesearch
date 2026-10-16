"""Web search across pluggable backends.

The search analogue of :mod:`wesearch.fetch`: one synchronous
backend-agnostic entry point over SearXNG, DuckDuckGo, and Google.

- :mod:`.types` -- ``SearchResult`` and its per-category subclasses,
  ``SearchError``, ``SearchBackends``, ``DEFAULT_SEARCH_CHAIN``,
  and ``SearxngCategory``.
- :mod:`.brave` -- the keyed Brave Search API backend.
- :mod:`.searxng` -- the SearXNG backend, its per-template parsers, and the
  ``CATEGORIES`` table each tab's gloss and parser live in.
- :mod:`.duckduckgo` -- the always-available scraped backend.
- :mod:`.cache` -- the host-wide result cache.
- :mod:`.search` -- ``search``, which tries backends in order and returns the
  first non-empty answer.

This ``__init__`` re-exports the names a caller needs to NAME a result or ask
for one, so the common import is one line rather than three. It deliberately
does not re-export the backend functions themselves: reach into
:mod:`.duckduckgo` or :mod:`.searxng` to drive one directly, and keep one
canonical import path per symbol.
"""

from __future__ import annotations

from wesearch.search.search import search
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


__all__ = [
    "DEFAULT_SEARCH_CHAIN",
    "CodeResult",
    "FileResult",
    "ImageResult",
    "MapResult",
    "MediaResult",
    "PackageResult",
    "PaperResult",
    "SearchBackends",
    "SearchError",
    "SearchResult",
    "SearxngCategory",
    "TorrentResult",
    "VideoResult",
    "search",
]
