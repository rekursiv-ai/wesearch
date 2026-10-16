"""OpenAlex backend for :mod:`wesearch.paper`: search and citation graph.

Maps OpenAlex ``/works`` results onto :class:`PaperRecord`. Set
``OPENALEX_EMAIL`` to join the polite pool and ``OPENALEX_API_KEY`` for a
higher daily budget.
"""

from __future__ import annotations

from typing import cast

import json
import os
import re

from treekle import MutablePlainTree, ReadError, from_plain, json_loads

from wesearch.fetch import (
    ContentParams,
    FetchSession,
    PolicyParams,
    RequestParams,
    RetryParams,
    Transport,
    fetch,
)
from wesearch.paper.errors import (
    BackendError,
    NotFoundError,
    translate_http_error,
)
from wesearch.paper.paginate import Cursor, Page, paginate
from wesearch.paper.types import IdType, PaperRecord
from wesearch.types.errors import FetchError


__all__ = [
    "citations",
    "references",
    "search",
]


def search(
    query: str,
    *,
    limit: int | None,
    year_from: int | None,
    year_to: int | None,
    open_access_only: bool,
    transport: Transport = "auto",
    session: FetchSession | None = None,
) -> tuple[list[PaperRecord], int, bool]:
    """Query OpenAlex via ``title_and_abstract.search``; every term must match.

    Args:
      query: Free-text query matched against title and abstract.
      limit: Maximum records to return, or ``None`` for one page.
      year_from: Inclusive lower publication-year bound, when set.
      year_to: Inclusive upper publication-year bound, when set.
      open_access_only: Restrict to works with an open-access location.
      transport: Retrieval transport forwarded to the HTTP layer.
      session: Connections to reuse; ``None`` uses the process's shared session.

    Returns:
      records: The mapped works, in OpenAlex rank order.
      total: OpenAlex's reported match count.
      complete: Whether the cursor was walked to exhaustion.

    Raises:
      PaperError: On an HTTP failure, timeout, or bad JSON.

    """
    base = _filter(
        year_from=year_from,
        year_to=year_to,
        open_access_only=open_access_only,
    )
    # Comma (filter separator) and pipe (OR) become spaces. Do not quote instead:
    # quoting switches to exact-phrase match and collapses recall.
    sanitized = query.replace(",", " ").replace("|", " ")
    terms = f"title_and_abstract.search:{sanitized}"
    flt = f"{base},{terms}" if base else terms
    page, total = _paginate_works(
        {"filter": flt},
        limit=limit,
        transport=transport,
        session=session,
    )
    return [_work_to_record(w) for w in page.entries], total, page.complete


def references(
    kind: IdType,
    canonical: str,
    *,
    limit: int | None,
    transport: Transport = "auto",
    session: FetchSession | None = None,
) -> tuple[list[PaperRecord], bool]:
    """Fetch the works a paper cites (outgoing edges).

    Args:
      kind: Seed identifier type (must be ``doi``).
      canonical: Bare seed DOI.
      limit: Maximum reference records to return, or ``None`` for all.
      transport: Retrieval transport forwarded to the HTTP layer.
      session: Connections to reuse; ``None`` uses the process's shared session.

    Returns:
      records: The seed's cited works, as resolved by OpenAlex.
      complete: Whether every referenced work was returned (False if
        curtailed by ``limit`` or if OpenAlex could not resolve an id).

    Raises:
      BackendError: For a non-DOI seed; use the S2 source for an arXiv id.
      NotFoundError: When OpenAlex has no work for the seed DOI.
      PaperError: On any HTTP failure.

    """
    work = _resolve_work(
        kind,
        canonical,
        extra_select="referenced_works",
        transport=transport,
        session=session,
    )
    ref_values = _array(work.get("referenced_works"))
    ref_urls = [value for value in ref_values if isinstance(value, str)]
    ids = [_work_id_tail(u) for u in ref_urls]
    if limit is None:
        capped = ids
    else:
        capped = ids[:limit]
    records = _resolve_works(capped, transport=transport, session=session)
    # The ``openalex:`` OR-filter silently drops unresolvable ids and de-dups
    # repeats, so compare against the distinct requested count.
    complete = (limit is None or len(ids) <= limit) and len(records) == len(set(capped))
    return records, complete


def citations(
    kind: IdType,
    canonical: str,
    *,
    limit: int | None,
    year_from: int | None = None,
    transport: Transport = "auto",
    session: FetchSession | None = None,
) -> tuple[list[PaperRecord], int, bool]:
    """Fetch the works that cite a paper (incoming edges).

    Args:
      kind: Seed identifier type (must be ``doi``).
      canonical: Bare seed DOI.
      limit: Maximum citing records to return, or ``None`` for one page.
      year_from: Inclusive lower publication-year bound, applied server-side.
      transport: Retrieval transport forwarded to the HTTP layer.
      session: Connections to reuse; ``None`` uses the process's shared session.

    Returns:
      records: Works citing the seed, in OpenAlex cursor order.
      total: OpenAlex's reported total number of citing works.
      complete: Whether the cursor was walked to exhaustion (False when
        ``limit`` curtailed the walk).

    Raises:
      BackendError: For a non-DOI seed; use the S2 source for an arXiv id.
      NotFoundError: When OpenAlex has no work for the seed DOI.
      PaperError: On any HTTP failure.

    """
    work = _resolve_work(
        kind,
        canonical,
        extra_select="id",
        transport=transport,
        session=session,
    )
    work_id = _work_id_tail(str(work.get("id") or ""))
    flt = f"cites:{work_id}"
    if year_from is not None:
        flt += f",from_publication_date:{year_from}-01-01"
    page, total = _paginate_works(
        {"filter": flt},
        limit=limit,
        transport=transport,
        session=session,
    )
    records = [_work_to_record(w) for w in page.entries]
    return records, total, page.complete


def _filter(
    *,
    year_from: int | None,
    year_to: int | None,
    open_access_only: bool,
) -> str:
    """Build an OpenAlex filter string from year bounds and OA flag."""
    parts: list[str] = []
    if year_from is not None:
        parts.append(f"from_publication_date:{year_from}-01-01")
    if year_to is not None:
        parts.append(f"to_publication_date:{year_to}-12-31")
    if open_access_only:
        parts.append("open_access.is_oa:true")
    return ",".join(parts)


def _paginate_works(
    extra_params: dict[str, str | int],
    *,
    limit: int | None,
    per_page_max: int = 200,
    transport: Transport = "auto",
    session: FetchSession | None = None,
) -> tuple[Page, int]:
    """Walk ``/works`` via the shared cursor; return (page, reported-total)."""
    total = 0
    if limit == 0:
        return Page(entries=[], complete=True), total

    def fetch_page(page_no: int, size: int) -> dict[str, MutablePlainTree]:
        nonlocal total
        params: dict[str, str | int] = {
            "select": _select(),
            **extra_params,
            "page": page_no,
            "per-page": size,
        }
        body = _get("/works", params, transport=transport, session=session)
        total = _count(_object(body.get("meta")).get("count"))
        return body

    cursor = Cursor(
        fetch=fetch_page,
        rows=lambda body: cast(
            list[dict[str, MutablePlainTree]],
            body.get("results") or [],
        ),
        advance=_works_page_advance,
        page_size_max=per_page_max,
        start=1,
    )
    return paginate(cursor, limit=limit), total


def _select(extra: str = "") -> str:
    """Return the comma-joined ``select`` field list, plus ``extra`` when set."""
    fields = (
        "id",
        "doi",
        "ids",
        "title",
        "display_name",
        "authorships",
        "publication_year",
        "primary_location",
        "cited_by_count",
        "referenced_works_count",
        "abstract_inverted_index",
        "open_access",
    )
    return ",".join((*fields, extra)) if extra else ",".join(fields)


def _get(
    path: str,
    params: dict[str, str | int],
    *,
    base: str = "https://api.openalex.org",
    timeout_sec: float = 10.0,
    transport: Transport = "auto",
    session: FetchSession | None = None,
) -> dict[str, MutablePlainTree]:
    """GET an OpenAlex path with polite UA + optional key; parse JSON."""
    api_key = os.environ.get("OPENALEX_API_KEY", "")
    if api_key:
        params = {**params, "api_key": api_key}
    try:
        raw = fetch(
            url=f"{base}{path}",
            request=RequestParams(
                content=ContentParams(params=params, headers=_headers()),
                retry=RetryParams(timeout_sec=timeout_sec),
                policy=PolicyParams(transport=transport),
            ),
            session=session,
        )
    except FetchError as e:
        detail = e.body[:200].decode(errors="replace")
        rate_limit_message = (
            "OpenAlex rate limit / daily credit budget exhausted. Set "
            "OPENALEX_API_KEY for a higher budget, or retry after the reset "
            f"(midnight UTC). {detail}"
        )
        backend = "OpenAlex"
        # OpenAlex signals real not-found semantically (200 + empty results);
        # an HTTP 404 here is a bad endpoint -> BackendError, not NotFound.
        raise translate_http_error(
            e,
            backend=backend,
            rate_limit_message=rate_limit_message,
            treat_404_as_missing=False,
        ) from e
    except (TimeoutError, OSError) as e:
        raise BackendError(
            f"OpenAlex request failed (timeout or connection error): {e}",
        ) from e
    try:
        body = json_loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise BackendError(f"OpenAlex returned invalid JSON: {e}") from e
    # Checked, not cast: a list or null must raise PaperError, not AttributeError,
    # so a fused search can still degrade to the other backend.
    if not isinstance(body, dict):
        raise BackendError(
            f"OpenAlex returned {type(body).__name__}, expected a JSON object.",
        )
    return body


def _headers() -> dict[str, str]:
    """Return request headers; a mailto in the UA joins OpenAlex's polite pool."""
    email = os.environ.get("OPENALEX_EMAIL", "")
    ua = f"loop-paper (mailto:{email})" if email else "loop-paper"
    return {"Accept": "application/json", "User-Agent": ua}


def _count(value: object) -> int:
    """Return an integer count, parsing ``"3"`` and ``3.0``, else 0."""
    try:
        return from_plain(value, int, strict=False, default=0)
    except ReadError:
        return 0


def _object(value: object) -> dict[str, object]:
    """Return ``value`` when it is a JSON object, else ``{}``."""
    try:
        return from_plain(value, dict[str, object])
    except ReadError:
        return {}


# Stops on ``meta.count``: ``len < size`` never fires for a full final page.
def _works_page_advance(
    body: dict[str, MutablePlainTree],
    page_no: int,
    size: int,
) -> int | None:
    """Next 1-based ``/works`` page, or None at the end."""
    rows = _array(body.get("results"))
    count = _count(_object(body.get("meta")).get("count"))
    seen = (page_no - 1) * size + len(rows)
    return page_no + 1 if rows and seen < count else None


def _array(value: object) -> list[object]:
    """Return ``value`` when it is a JSON array, else ``[]``."""
    try:
        return from_plain(value, list[object])
    except ReadError:
        return []


def _work_to_record(work: dict[str, MutablePlainTree]) -> PaperRecord:
    """Convert an OpenAlex work dict into a :class:`PaperRecord`."""
    authorships = _objects(work.get("authorships"))
    authors = tuple(
        str(_object(a.get("author")).get("display_name"))
        for a in authorships
        if _object(a.get("author")).get("display_name")
    )
    title = str(work.get("title") or work.get("display_name") or "")

    doi_raw = work.get("doi")
    doi = ""
    if isinstance(doi_raw, str):
        doi = (
            doi_raw.removeprefix("https://doi.org/")
            .removeprefix("http://doi.org/")
            .removeprefix("https://dx.doi.org/")
            .removeprefix("http://dx.doi.org/")
        )

    arxiv = ""
    ids = _object(work.get("ids"))
    arxiv_raw = ids.get("arxiv")
    if isinstance(arxiv_raw, str):
        m = re.search(
            r"(?:arxiv\.org/abs/|arxiv:)?([\w.-]+/\d+|\d{4}\.\d{4,5})",
            arxiv_raw,
        )
        if m:
            value = m.group(1)
            if isinstance(value, str):
                arxiv = value
    if not arxiv and doi:
        # An arXiv preprint work carries only arXiv's DataCite DOI; its suffix is
        # the id fusion joins on. Strip the version: S2 reports the bare id.
        m = re.match(r"10\.48550/arxiv\.(.+?)(?:v\d+)?$", doi, re.IGNORECASE)
        if m:
            value = m.group(1)
            if isinstance(value, str):
                arxiv = value

    primary = _object(work.get("primary_location"))
    source = _object(primary.get("source"))
    venue = source.get("display_name")
    oa = _object(work.get("open_access"))

    return PaperRecord(
        title=title,
        authors=authors,
        year=cast(int | None, work.get("publication_year")),
        venue=str(venue) if venue else "",
        doi=doi,
        arxiv_id=arxiv,
        abstract=_reconstruct_abstract(
            cast(dict[str, list[int]] | None, work.get("abstract_inverted_index")),
        ),
        citation_count=cast(int | None, work.get("cited_by_count")),
        reference_count=cast(int | None, work.get("referenced_works_count")),
        open_access_pdf=str(oa["oa_url"]) if oa.get("oa_url") else "",
        sources=("openalex",),
    )


def _objects(value: object) -> list[dict[str, object]]:
    """Return the nonempty JSON objects of an array, dropping other elements."""
    return [obj for item in _array(value) if (obj := _object(item))]


def _reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str:
    """Rebuild plain text from OpenAlex's ``{word: [positions]}`` abstract."""
    positions: dict[int, str] = {}
    for word, idxs in (inverted or {}).items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def _resolve_work(
    kind: IdType,
    canonical: str,
    *,
    extra_select: str,
    transport: Transport = "auto",
    session: FetchSession | None = None,
) -> dict[str, object]:
    """Resolve a seed DOI to its OpenAlex work."""
    if kind != "doi":
        raise BackendError(
            "OpenAlex citation graph resolves DOIs only; arXiv-id resolution is "
            "unreliable. Use the S2 source for an arXiv id, or supply the DOI.",
        )
    data = _get(
        "/works",
        {"filter": f"doi:{canonical}", "select": f"id,{extra_select}"},
        transport=transport,
        session=session,
    )
    results = _objects(data.get("results"))
    if not results:
        raise NotFoundError(f"OpenAlex has no work for doi:{canonical}.")
    return results[0]


def _work_id_tail(url_or_id: str) -> str:
    """Return the bare ``W...`` id from an OpenAlex work URL or id."""
    return url_or_id.rsplit("/", maxsplit=1)[-1]


def _resolve_works(
    work_ids: list[str],
    *,
    per_page_max: int = 200,
    transport: Transport = "auto",
    session: FetchSession | None = None,
) -> list[PaperRecord]:
    """Batch-resolve OpenAlex work ids to records, in no particular order."""
    records: list[PaperRecord] = []
    for start in range(0, len(work_ids), per_page_max):
        chunk = work_ids[start : start + per_page_max]
        page, _ = _paginate_works(
            {"filter": f"openalex:{'|'.join(chunk)}"},
            limit=per_page_max,
            per_page_max=per_page_max,
            transport=transport,
            session=session,
        )
        records.extend(_work_to_record(w) for w in page.entries)
    return records
