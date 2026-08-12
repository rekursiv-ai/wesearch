"""Web search across pluggable backends.

The search analogue of :mod:`wesearch.fetch`: one synchronous
backend-agnostic entry point over SearXNG, DuckDuckGo, and Google.

- :mod:`.custom_types` -- ``SearchResult`` and its per-category subclasses,
  ``SearchError``, ``SearchBackends``, ``DEFAULT_SEARCH_BACKEND``,
  and ``SearxngCategory``.
- :mod:`.searxng` -- the SearXNG backend, its per-template parsers, and the
  ``CATEGORIES`` table each tab's gloss and parser live in.
- :mod:`.duckduckgo` -- the always-available scraped backend.
- :mod:`.search` -- ``search``, which picks a backend and returns its
  records, plus the ``SearchParamsSchema`` every surface renders.

This ``__init__`` re-exports the names a caller needs to NAME a result or ask
for one, so the common import is one line rather than three. It deliberately
does not re-export the backend functions themselves: reach into
:mod:`.duckduckgo` or :mod:`.searxng` to drive one directly, and keep one
canonical import path per symbol.
"""

from __future__ import annotations

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
from wesearch.search.search import search


__all__ = [
    "DEFAULT_SEARCH_BACKEND",
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
