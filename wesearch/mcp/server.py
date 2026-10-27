#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
MCP server exposing wesearch to agent clients over stdio.

Run with ``wesearch-mcp`` (installed via the ``mcp`` extra: ``pip install
wesearch[mcp]``), as ``sh server.py``, or as ``python -m
wesearch.mcp.server``. Every tool wraps a public wesearch function and
returns token-lean output: abstracts are truncated, empty fields dropped, and
PDF bytes are written to the user cache directory rather than returned inline.

Each MCP client session may run its own server process; rate limits and
cookie profiles are shared on disk. Errors reach the client as MCP tool errors
carrying the underlying exception message.
'''
# fmt: on

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Literal

import functools
import hashlib
import inspect
import re


try:
    from mcp.server import MCPServer
    from mcp.server.mcpserver import Context
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError as e:  # pragma: no cover -- depends on the install's extras.
    # A bare ModuleNotFoundError names `mcp`, not the extra that supplies it.
    raise ImportError(
        "The wesearch MCP server requires the 'mcp' extra: pip install wesearch[mcp]",
    ) from e

from wesearch.fetch import FetchSession
from wesearch.lib.userdirs import cache_dir
from wesearch.mcp.types import FetchParamsSchema, SearchParamsSchema
from wesearch.paper import (
    authors,
    details,
    fetch,
    search as paper_search_module,
)
from wesearch.paper.ids import id_slug, normalize_id
from wesearch.paper.render import lean_author, lean_record
from wesearch.search.render import lean_result
from wesearch.search.search import search
from wesearch.search.types import (
    DEFAULT_SEARCH_CHAIN,
    SearchBackends,
    SearxngCategory,
)
from wesearch.types.params import (
    Extractor,
    PolicyParams,
    Transport,
)
from wesearch.web import fetch_web


@asynccontextmanager
async def _lifespan(_: MCPServer[FetchSession]) -> AsyncGenerator[FetchSession]:
    """Share one session -- connections, Chrome, the egress -- across every call."""
    async with FetchSession() as session:
        yield session


mcp = MCPServer[FetchSession](
    lifespan=_lifespan,
    name="wesearch",
    instructions=(
        "Scholarly-paper search and resilient web access. Paper ids may be "
        "DOIs or arXiv ids in any common form (bare, prefixed, or full URL). "
        "Prefer paper_search for literature discovery; results fuse Semantic "
        "Scholar and OpenAlex."
    ),
)


# The MCP SDK (2.x) hides any exception other than ``ToolError`` behind a generic
# ``Error executing tool <name>``. Must sit UNDER ``@mcp.tool()``.
def _surface_errors[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Re-raise a tool's failure as ``ToolError`` so its message reaches the client."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            raise ToolError(f"{type(e).__name__}: {e}") from e

    # The SDK publishes ``__doc__`` verbatim, and Python 3.12 keeps the
    # indentation 3.13+ strips. ``ctx`` is injected, so clients never see it.
    wrapper.__doc__ = re.sub(r"(?m)^  ctx: .*\n", "", inspect.getdoc(fn) or "")
    return wrapper


@mcp.tool()
@_surface_errors
def paper_search(
    query: str,
    *,
    source: Literal["fused", "s2", "openalex", "searxng"] = "fused",
    limit: int = 10,
    year_from: int | None = None,
    year_to: int | None = None,
    open_access_only: bool = False,
    ctx: Context[FetchSession] | None = None,
) -> dict[str, object]:
    """Search scholarly literature. The default "fused" source rank-fuses
    Semantic Scholar and OpenAlex and degrades gracefully if one is down
    (complete=false means a backend was lost or more matches remain).

    Args:
      query: Search term(s) for papers.
      source: Search backend: "fused" (rank-fused S2 + OpenAlex),
        "s2", "openalex", or "searxng".
      limit: Maximum number of results to return.
      year_from: Minimum publication year (inclusive), or None for all.
      year_to: Maximum publication year (inclusive), or None for all.
      open_access_only: Restrict to papers with free full-text PDFs.
      ctx: The request, injected by the server; not a client argument.

    Returns:
      result: Dict with "records" (list of papers), "total" (estimated
        result count), and "complete" (false when more matches remain or a
        backend was lost).

    """
    if limit < 1:
        raise ValueError(f"'limit' must be >= 1, got {limit}.")
    result = paper_search_module.search(
        query,
        source=source,
        limit=limit,
        year_from=year_from,
        year_to=year_to,
        open_access_only=open_access_only,
        session=_session(ctx),
    )
    return {
        "records": [lean_record(r) for r in result.records],
        "total": result.total,
        "complete": result.complete,
    }


@mcp.tool()
@_surface_errors
def paper_details(
    paper_id: str,
    *,
    ctx: Context[FetchSession] | None = None,
) -> dict[str, object]:
    """Full metadata for one paper. Accepts a DOI or arXiv id in any common
    form (bare, doi:/arxiv: prefixed, or full URL).

    Args:
      paper_id: DOI or arXiv id (bare, prefixed, or full URL).
      ctx: The request, injected by the server; not a client argument.

    Returns:
      metadata: Dict of paper metadata with extended abstract (1500 chars).

    """
    kind, canonical = normalize_id(paper_id)
    record = details.metadata(kind, canonical, session=_session(ctx))
    lean = lean_record(record, abstract_chars=1_500)
    lean["id"] = f"{kind}:{canonical}"
    return lean


@mcp.tool()
@_surface_errors
def paper_references(
    paper_id: str,
    limit: int = 20,
    source: Literal["s2", "openalex"] = "s2",
    *,
    ctx: Context[FetchSession] | None = None,
) -> dict[str, object]:
    """Papers this paper cites (its bibliography). source="openalex" reaches
    an independent quota when Semantic Scholar is throttled (DOI seeds only).

    Args:
      paper_id: DOI or arXiv id to fetch bibliography of.
      limit: Maximum citations to return.
      source: Search backend (s2 or openalex; openalex DOI-only).
      ctx: The request, injected by the server; not a client argument.

    Returns:
      references: Dict with "records" (papers this paper cites) and "complete"
        (false when more may exist beyond those returned).

    """
    if limit < 1:
        raise ValueError(f"'limit' must be >= 1, got {limit}.")
    kind, canonical = normalize_id(paper_id)
    listing = details.references(
        kind,
        canonical,
        limit=limit,
        source=source,
        session=_session(ctx),
    )
    return {
        "records": [lean_record(r) for r in listing.records],
        "complete": listing.complete,
    }


@mcp.tool()
@_surface_errors
def paper_citations(
    paper_id: str,
    limit: int = 20,
    influential_only: bool = False,
    year_from: int | None = None,
    source: Literal["s2", "openalex"] = "s2",
    *,
    ctx: Context[FetchSession] | None = None,
) -> dict[str, object]:
    """Papers that cite this paper. influential_only keeps only citations
    Semantic Scholar flags as influential (S2 only). source="openalex" reaches
    an independent quota when Semantic Scholar is throttled (DOI seeds only).

    Args:
      paper_id: DOI or arXiv id to fetch citations of.
      limit: Maximum papers to return.
      influential_only: Include only citations Semantic Scholar flags as
        influential (S2 backend only).
      year_from: Minimum publication year (inclusive), or None for all.
      source: Search backend (s2 or openalex; openalex DOI-only).
      ctx: The request, injected by the server; not a client argument.

    Returns:
      citations: Dict with "records" (papers citing this paper) and "complete"
        (false when more may exist beyond those returned).

    """
    if limit < 1:
        raise ValueError(f"'limit' must be >= 1, got {limit}.")
    kind, canonical = normalize_id(paper_id)
    listing = details.citations(
        kind,
        canonical,
        limit=limit,
        influential_only=influential_only,
        year_from=year_from,
        source=source,
        session=_session(ctx),
    )
    return {
        "records": [lean_record(r) for r in listing.records],
        "complete": listing.complete,
    }


@mcp.tool()
@_surface_errors
def paper_pdf(
    paper_id: str,
    *,
    ctx: Context[FetchSession] | None = None,
) -> dict[str, object]:
    """Download a paper's PDF (arXiv direct, then open-access lookup) into
    the local cache and return its filesystem path.

    Args:
      paper_id: DOI or arXiv id of the paper to download.
      ctx: The request, injected by the server; not a client argument.

    Returns:
      pdf: Dict with "path" (filesystem location in cache), "bytes" (size),
        and "source" (arXiv or open-access database name).

    """
    kind, canonical = normalize_id(paper_id)
    pdf_bytes, source = fetch.download(kind, canonical, session=_session(ctx))
    target_dir = cache_dir() / "rekursiv-ai" / "wesearch" / "papers"
    target_dir.mkdir(parents=True, exist_ok=True)
    # Slug for a human, digest for identity: ``id_slug`` maps ``10.1/a_b`` and
    # ``10.1/a/b`` to one name, and a long DOI exceeds the 255-byte filename limit.
    digest = hashlib.sha256(f"{kind}:{canonical}".encode()).hexdigest()[:16]
    target = target_dir / f"{id_slug(kind, canonical)[:80]}.{digest}.pdf"
    target.write_bytes(pdf_bytes)
    return {"path": str(target), "bytes": len(pdf_bytes), "source": source}


@mcp.tool()
@_surface_errors
def author_search(
    query: str,
    limit: int = 10,
    *,
    ctx: Context[FetchSession] | None = None,
) -> dict[str, object]:
    """Find scholars by name; results are ranked by h-index.

    Args:
      query: Scholar name or name fragment.
      limit: Maximum results to return (ranked by h-index).
      ctx: The request, injected by the server; not a client argument.

    Returns:
      authors: Dict with "records" (author profiles) and "total" (estimated
        result count).

    """
    if limit < 1:
        raise ValueError(f"'limit' must be >= 1, got {limit}.")
    result = authors.search_authors(query, limit=limit, session=_session(ctx))
    return {
        "records": [lean_author(r) for r in result.records],
        "total": result.total,
    }


@mcp.tool()
@_surface_errors
def author_papers(
    author_id: str,
    limit: int = 20,
    year_from: int | None = None,
    year_to: int | None = None,
    *,
    ctx: Context[FetchSession] | None = None,
) -> dict[str, object]:
    """Publications of one author (author_id from author_search).

    Args:
      author_id: Scholar id from author_search.
      limit: Maximum papers to return.
      year_from: Minimum publication year (inclusive), or None for all.
      year_to: Maximum publication year (inclusive), or None for all.
      ctx: The request, injected by the server; not a client argument.

    Returns:
      papers: Dict with "records" (authored papers) and "complete" (false when
        more may exist beyond those returned).

    """
    if limit < 1:
        raise ValueError(f"'limit' must be >= 1, got {limit}.")
    if year_from is not None and year_to is not None and year_from > year_to:
        raise ValueError(f"'year_from' ({year_from}) exceeds 'year_to' ({year_to}).")
    listing = authors.author_papers(
        author_id,
        limit=limit,
        year_from=year_from,
        year_to=year_to,
        session=_session(ctx),
    )
    return {
        "records": [lean_record(r) for r in listing.records],
        "complete": listing.complete,
    }


# Must sit UNDER ``@mcp.tool()``, which publishes ``__doc__`` at registration.
def _append_doc[F: Callable[..., object]](extra: str) -> Callable[[F], F]:
    """Append generated prose to a tool's docstring before the SDK reads it."""

    def decorate(fn: F) -> F:
        # `getdoc` dedents: Python 3.12 keeps the indentation 3.13+ strips from
        # __doc__, which would misalign the column-0 ``extra`` with the sections.
        fn.__doc__ = f"{inspect.getdoc(fn) or ''}\n\n{extra}\n"
        return fn

    return decorate


@mcp.tool()
@_surface_errors
@_append_doc(
    f"Omitting ``backend`` tries {', '.join(DEFAULT_SEARCH_CHAIN)} in order.\n\n"
    f"{SearchParamsSchema.asset_markdown()}",
)
def web_search(
    query: str,
    num_results: int = 10,
    backend: SearchBackends | None = None,
    categories: SearxngCategory = "general",
    transport: Transport = "auto",
    *,
    ctx: Context[FetchSession] | None = None,
) -> list[dict[str, object]]:
    """Web search.

    Omitting ``backend`` walks the fallback order stated below, skipping any backend
    that is blocked, unconfigured (SearXNG needs ``SEARXNG_URL``, Brave
    ``BRAVE_API_KEY``), or empty; repeat queries are served from a host-wide cache.
    ``categories`` selects a SearXNG result tab, which selects that backend when
    none is named and is rejected alongside an explicit non-SearXNG one. The list
    at the end describes each parameter, including every tab and what it returns.

    Args:
      query: Search terms.
      num_results: Maximum number of results to return.
      backend: Search provider; "searxng" requires the SEARXNG_URL environment
        variable.
      categories: SearXNG result tab; the tabs are listed at the end.
      transport: Retrieval path, as for web_fetch; see the list at the end.
      ctx: The request, injected by the server; not a client argument.

    Returns:
      results: One dict per hit with its url, title, and snippet, plus the
        tab's own fields (a paper's DOI and authors, an image's resolution).

    """
    results = search(
        query,
        backend=backend,
        num_results=num_results,
        categories=categories,
        transport=transport,
        session=_session(ctx),
    )
    # Not a url/title/snippet projection: that drops each tab's own fields.
    return [lean_result(r) for r in results]


@mcp.tool()
@_surface_errors
@_append_doc(FetchParamsSchema.asset_markdown())
def web_fetch(
    url: str,
    max_chars: int = 8000,
    transport: Transport = "auto",
    extractor: Extractor = "html2text",
    *,
    ctx: Context[FetchSession] | None = None,
) -> dict[str, object]:
    """Fetch a page and return its extracted text.

    GET only. ``url``, ``transport``, and ``extractor`` are described in the list at
    the end.

    Args:
      url: The page to fetch; see the list at the end.
      max_chars: Most characters of text to return; at least 1.
      transport: Retrieval path; see the list at the end.
      extractor: How the page becomes text; the values are listed at the end.
      ctx: The request, injected by the server; not a client argument.

    Returns:
      page: Dict with "url" (the URL as requested), "text" (the extracted text,
        possibly empty, cut to max_chars), "truncated" (whether that cut removed
        anything), and "kind" (how the body became text, such as 'html' through
        the extractor, 'markdown' from the reader proxy as-is, or 'rss' for a
        feed).

    """
    if max_chars < 1:
        raise ValueError(f"'max_chars' must be >= 1, got {max_chars}.")
    # The URL comes from a model, so it stays untrusted (the PolicyParams
    # default): the host must resolve to a public address, never loopback or
    # private ranges.
    result = fetch_web(
        url,
        max_chars=max_chars,
        policy=PolicyParams(transport=transport, extractor=extractor),
        session=_session(ctx),
    )
    return {
        "url": result.url,
        "text": result.text,
        "truncated": result.truncated,
        "kind": result.kind,
    }


def main() -> int:
    """Serve the MCP tools over stdio.

    Returns:
      status: Process exit status.

    """
    mcp.run()
    return 0


def _session(ctx: Context[FetchSession] | None) -> FetchSession | None:
    """Return the server's session; ``None`` (the process's shared one) outside a request."""
    if not isinstance(ctx, Context):
        return None
    try:
        return ctx.request_context.lifespan_context
    except ValueError:
        return None


# Not a package `__main__.py`: `mcp` is an optional extra, so `python -m
# wesearch` would fail to import on every install that did not ask for it.
if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
