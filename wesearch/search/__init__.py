"""Web search across pluggable backends.

The search analogue of :mod:`wesearch.fetch`: one synchronous
backend-agnostic entry point over SearXNG, DuckDuckGo, and Google.

- :mod:`.custom_types` -- ``SearchResult`` and its per-category subclasses,
  ``SearchError``, ``SearchBackends``, ``DEFAULT_SEARCH_BACKEND``.
- :mod:`.searxng` -- the SearXNG backend + ``SearxngCategory``.
- :mod:`.duckduckgo` -- the always-available scraped backend.
- :mod:`.search` -- ``search``, which picks a backend and returns its records.

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
    TorrentResult,
    VideoResult,
)
from wesearch.search.search import search
from wesearch.search.searxng import SearxngCategory


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
