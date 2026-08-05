"""Scholarly-literature search over multiple backends.

The paper analogue of :func:`wesearch.search.search`: sync, backend-agnostic
text search. ``source`` selects a backend or ``"fused"`` (reciprocal-rank-fuse
Semantic Scholar + OpenAlex, resilient to either being down). Sync by design --
a coroutine call site lifts it into a thread with ``asyncio.to_thread``.

Sources:
  - ``"s2"`` -- Semantic Scholar.
  - ``"openalex"`` -- OpenAlex.
  - ``"searxng"`` -- SearXNG science metasearch.
  - ``"fused"`` (default) -- reciprocal-rank-fuse S2 + OpenAlex.

Some builds enable additional scraped sources (see :mod:`.providers`); the
``Source`` type enumerates exactly those available in this build.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from wesearch.fetch import Transport
from wesearch.paper.custom_types import PaperRecord
from wesearch.paper.errors import PaperError
from wesearch.paper.fuse import fuse
from wesearch.paper.providers import (
    openalex,
    s2,
    searxng,
)


__all__ = [
    "SearchResult",
    "Source",
    "search",
]

Source = Literal["s2", "openalex", "searxng", "fused"]


class _SingleBackend(Protocol):
    """One backend's text search.

    A Protocol rather than ``Callable[..., ...]``: the ellipsis form leaves
    every keyword at the dispatch call site unchecked, so a provider renaming
    a parameter would type-check clean and fail at runtime.
    """

    def __call__(
        self,
        query: str,
        *,
        limit: int | None,
        year_from: int | None,
        year_to: int | None,
        open_access_only: bool,
        transport: Transport,
    ) -> tuple[list[PaperRecord], int, bool]:
        """Return this backend's ``(records, total, complete)``."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchResult:
    """A paper search result set.

    Attributes:
      records: Ranked paper records (already trimmed to ``limit``).
      total: Backend-reported total match count, or the post-filter count for
        backends that report none (SearXNG, Google Scholar). For a fused
        search, the largest of the two backend totals and the pre-trim fused
        count -- the two overlap unknowably, so this is a lower bound.
      complete: Whether every match was returned. False when a backend's cursor
        was cut short (by ``limit`` or a depth ceiling), and, for a fused
        search, also when a backend was lost to an error -- a partial result a
        caller may decline to cache.

    """

    records: list[PaperRecord]
    total: int
    complete: bool


def search(
    query: str,
    *,
    source: Source = "fused",
    limit: int | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    open_access_only: bool = False,
    transport: Transport = "auto",
) -> SearchResult:
    """Search the scholarly literature and return ranked paper records.

    Args:
      query: Free-text query, matched against title/abstract by every backend.
      source: Which backend (or ``"fused"``) to query.
      limit: Max hits, or ``None`` for the backend's default page.
      year_from: Inclusive lower publication-year bound.
      year_to: Inclusive upper publication-year bound.
      open_access_only: Restrict to papers with a known open-access PDF.
      transport: Retrieval transport forwarded to each selected provider.

    Returns:
      result: A :class:`SearchResult`.

    Raises:
      PaperError: On a total backend failure.

    """
    if source == "fused":
        return _fused(
            query,
            limit=limit,
            year_from=year_from,
            year_to=year_to,
            open_access_only=open_access_only,
            transport=transport,
        )
    records, total, complete = _single_backend(source)(
        query,
        limit=limit,
        year_from=year_from,
        year_to=year_to,
        open_access_only=open_access_only,
        transport=transport,
    )
    return SearchResult(records=records, total=total, complete=complete)


def _s2_search(
    query: str,
    *,
    limit: int | None,
    year_from: int | None,
    year_to: int | None,
    open_access_only: bool,
    transport: Transport = "auto",
) -> tuple[list[PaperRecord], int, bool]:
    """Query Semantic Scholar; return ``(records, total, complete)``."""
    params: dict[str, str | int] = {
        "query": query,
        "fields": s2.S2_PAPER_FIELDS_STR,
    }
    year_spec = _s2_year_param(year_from, year_to)
    if year_spec is not None:
        params["year"] = year_spec
    if open_access_only:
        params["openAccessPdf"] = ""  # S2 treats it as a presence flag.
    page, total = s2.search_paginate(params, limit=limit, transport=transport)
    records = [s2.paper_record_from(row) for row in page.entries]
    return records, total, page.complete


def _s2_year_param(year_from: int | None, year_to: int | None) -> str | None:
    """Translate year bounds to S2's ``year=FROM-TO`` query form."""
    if year_from is None and year_to is None:
        return None
    lo = str(year_from) if year_from is not None else ""
    hi = str(year_to) if year_to is not None else ""
    return f"{lo}-{hi}"


def _single_backend(source: Source) -> _SingleBackend:
    """Return the single-backend search function for ``source``."""
    table: dict[str, _SingleBackend] = {
        "s2": _s2_search,
        "openalex": openalex.search,
        "searxng": searxng.search,
    }
    backend = table.get(source)
    if backend is None:
        raise PaperError(f"Unknown search source: {source!r}")
    return backend


def _fused(
    query: str,
    *,
    limit: int | None,
    year_from: int | None,
    year_to: int | None,
    open_access_only: bool,
    transport: Transport = "auto",
) -> SearchResult:
    """Run S2 and OpenAlex, degrading gracefully when one fails."""
    s2_hits: list[PaperRecord] = []
    oa_hits: list[PaperRecord] = []
    s2_total = oa_total = 0
    exhausted = True
    errors: list[str] = []
    answered = 0

    try:
        s2_hits, s2_total, s2_complete = _s2_search(
            query,
            limit=limit,
            year_from=year_from,
            year_to=year_to,
            open_access_only=open_access_only,
            transport=transport,
        )
        exhausted = exhausted and s2_complete
        answered += 1
    except PaperError as e:
        errors.append(f"S2: {e}")
    try:
        oa_hits, oa_total, oa_complete = openalex.search(
            query,
            limit=limit,
            year_from=year_from,
            year_to=year_to,
            open_access_only=open_access_only,
            transport=transport,
        )
        exhausted = exhausted and oa_complete
        answered += 1
    except PaperError as e:
        errors.append(f"OpenAlex: {e}")

    # Only a TOTAL failure is an error: one backend returning cleanly -- even
    # with zero hits -- is a real result. ``complete`` is False when a backend
    # errored, so the caller can decline to cache the partial result.
    if not answered:
        raise PaperError("; ".join(errors))
    # Each backend already returned up to ``limit`` rows, so a disjoint fused
    # set holds up to twice that. Trim HERE, after fusion, so a paper both
    # backends ranked -- the hit fusion exists to surface -- outranks a lone
    # backend's spare rows rather than being cut with them. Trimming is this
    # function's job, not a consumer's: ``SearchResult.records`` promises a
    # trimmed list, and a caller that re-slices cannot restore what fusion knew.
    fused = fuse(s2_hits, oa_hits)
    records = fused if limit is None else fused[:limit]
    return SearchResult(
        records=records,
        # The two backend totals overlap unknowably, so `max` is the honest
        # lower bound -- but the fused set can hold papers unique to each, so
        # never report fewer than fusion actually found (pre-trim, since the
        # trimmed-away records are matches the caller may still page to).
        total=max(s2_total, oa_total, len(fused)),
        # Three ways to be incomplete: a backend errored, a backend's cursor was
        # cut short, or the trim above dropped a fused record.
        complete=not errors and exhausted and len(records) == len(fused),
    )
