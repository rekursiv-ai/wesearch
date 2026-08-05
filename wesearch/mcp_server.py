#!/bin/sh
# ruff: noqa: EXE003, D300 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
MCP server exposing wesearch to agent clients over stdio.

Run with ``wesearch-mcp`` (installed via the ``mcp`` extra: ``pip install
wesearch[mcp]``), as ``sh mcp_server.py``, or as ``python -m
wesearch.mcp_server``. Every tool is a thin wrapper over a public wesearch
function, with outputs reshaped to be token-lean for model consumption:
abstracts are truncated, empty fields dropped, and PDF bytes are written to
the user cache directory rather than returned inline.

Tools are synchronous by design -- wesearch's rate limiting and cookie/UA
profile state are cross-process safe on disk, so each MCP client session can
run its own server process without coordination. Errors surface to the
client as MCP tool errors carrying the underlying exception message
(``BotDetectionError`` includes its recovery guidance).
'''
# fmt: on

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import hashlib


try:
    from mcp.server import MCPServer
except ImportError as e:  # pragma: no cover -- depends on the install's extras.
    # The MCP SDK is an optional extra, so this module is the one place in the
    # package where a missing dependency is expected. A bare ModuleNotFoundError
    # names `mcp`, which tells a reader nothing about which extra supplies it.
    raise ImportError(
        "The wesearch MCP server requires the 'mcp' extra: pip install wesearch[mcp]"
    ) from e

from wesearch.fetch.common import public_host
from wesearch.fetch.fetch import RequestParams, fetch
from wesearch.lib.userdirs import cache_dir
from wesearch.paper import (
    authors as paper_authors_mod,
    details as paper_details_mod,
    fetch as paper_fetch_mod,
    search as paper_search_mod,
)
from wesearch.paper.ids import id_slug, normalize_id
from wesearch.search import search as web_search_fn


if TYPE_CHECKING:
    from bs4 import BeautifulSoup

    from wesearch.paper.custom_types import AuthorRecord, PaperRecord
else:
    from wrapt import lazy_import

    BeautifulSoup = lazy_import("bs4", "BeautifulSoup")  # 140ms

mcp = MCPServer(
    name="wesearch",
    instructions=(
        "Scholarly-paper search and resilient web access. Paper ids may be "
        "DOIs or arXiv ids in any common form (bare, prefixed, or full URL). "
        "Prefer paper_search for literature discovery; results fuse Semantic "
        "Scholar and OpenAlex."
    ),
)


def _clip(text: str | None, limit: int) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _lean_paper(record: PaperRecord, *, abstract_chars: int = 500) -> dict[str, object]:
    """Compact dict for one paper: empty fields dropped, abstract clipped."""
    authors: list[str] = list(record.authors[:5])
    if len(record.authors) > 5:
        authors.append("et al.")
    fields: dict[str, object] = {
        "title": record.title,
        "authors": authors,
        "year": record.year,
        "venue": record.venue,
        "doi": record.doi,
        "arxiv_id": record.arxiv_id,
        "citation_count": record.citation_count,
        "reference_count": record.reference_count,
        "open_access_pdf": record.open_access_pdf,
        "is_influential": record.is_influential,
        "abstract": _clip(record.abstract, abstract_chars),
    }
    return {k: v for k, v in fields.items() if v not in (None, [], "")}


def _lean_author(record: AuthorRecord) -> dict[str, object]:
    fields: dict[str, object] = {
        "author_id": record.author_id,
        "name": record.name,
        "affiliations": list(record.affiliations),
        "h_index": record.h_index,
        "citation_count": record.citation_count,
        "paper_count": record.paper_count,
    }
    return {k: v for k, v in fields.items() if v not in (None, [], "")}


@mcp.tool()
def paper_search(
    query: str,
    *,
    source: Literal["fused", "s2", "openalex", "searxng"] = "fused",
    limit: int = 10,
    year_from: int | None = None,
    year_to: int | None = None,
    open_access_only: bool = False,
) -> dict[str, object]:
    """Search scholarly literature. The default "fused" source rank-fuses
    Semantic Scholar and OpenAlex and degrades gracefully if one is down
    (complete=false means a backend was lost or more matches remain).
    """
    if limit < 1:
        raise ValueError(f"'limit' must be >= 1, got {limit}.")
    result = paper_search_mod.search(
        query,
        source=source,
        limit=limit,
        year_from=year_from,
        year_to=year_to,
        open_access_only=open_access_only,
    )
    return {
        "records": [_lean_paper(r) for r in result.records],
        "total": result.total,
        "complete": result.complete,
    }


@mcp.tool()
def paper_details(paper_id: str) -> dict[str, object]:
    """Full metadata for one paper. Accepts a DOI or arXiv id in any common
    form (bare, doi:/arxiv: prefixed, or full URL).
    """
    kind, canonical = normalize_id(paper_id)
    record = paper_details_mod.metadata(kind, canonical)
    # Three times the search clip: a detail lookup is one paper the caller
    # already chose, so the abstract is what they asked for.
    lean = _lean_paper(record, abstract_chars=1_500)
    lean["id"] = f"{kind}:{canonical}"
    return lean


@mcp.tool()
def paper_references(
    paper_id: str,
    limit: int = 20,
    source: Literal["s2", "openalex"] = "s2",
) -> dict[str, object]:
    """Papers this paper cites (its bibliography). source="openalex" reaches
    an independent quota when Semantic Scholar is throttled (DOI seeds only).
    """
    if limit < 1:
        raise ValueError(f"'limit' must be >= 1, got {limit}.")
    kind, canonical = normalize_id(paper_id)
    listing = paper_details_mod.references(kind, canonical, limit=limit, source=source)
    return {
        "records": [_lean_paper(r) for r in listing.records],
        "complete": listing.complete,
    }


@mcp.tool()
def paper_citations(
    paper_id: str,
    limit: int = 20,
    influential_only: bool = False,
    year_from: int | None = None,
    source: Literal["s2", "openalex"] = "s2",
) -> dict[str, object]:
    """Papers that cite this paper. influential_only keeps only citations
    Semantic Scholar flags as influential (S2 only). source="openalex" reaches
    an independent quota when Semantic Scholar is throttled (DOI seeds only).
    """
    if limit < 1:
        raise ValueError(f"'limit' must be >= 1, got {limit}.")
    kind, canonical = normalize_id(paper_id)
    listing = paper_details_mod.citations(
        kind,
        canonical,
        limit=limit,
        influential_only=influential_only,
        year_from=year_from,
        source=source,
    )
    return {
        "records": [_lean_paper(r) for r in listing.records],
        "complete": listing.complete,
    }


@mcp.tool()
def paper_pdf(paper_id: str) -> dict[str, object]:
    """Download a paper's PDF (arXiv direct, then open-access lookup) into
    the local cache and return its filesystem path.
    """
    kind, canonical = normalize_id(paper_id)
    pdf_bytes, source = paper_fetch_mod.download(kind, canonical)
    target_dir = cache_dir("wesearch") / "pdf"
    target_dir.mkdir(parents=True, exist_ok=True)
    # Slug for a human, digest for identity: ``id_slug`` maps every unsafe
    # character to ``_``, so ``10.1/a_b`` and ``10.1/a/b`` collide and one
    # paper overwrites the other. It is also unbounded, and a long DOI suffix
    # exceeds the 255-byte filename limit.
    digest = hashlib.sha256(f"{kind}:{canonical}".encode()).hexdigest()[:16]
    target = target_dir / f"{id_slug(kind, canonical)[:80]}.{digest}.pdf"
    target.write_bytes(pdf_bytes)
    return {"path": str(target), "bytes": len(pdf_bytes), "source": source}


@mcp.tool()
def author_search(query: str, limit: int = 10) -> dict[str, object]:
    """Find scholars by name; results are ranked by h-index."""
    result = paper_authors_mod.search_authors(query, limit=limit)
    return {
        "records": [_lean_author(r) for r in result.records],
        "total": result.total,
    }


@mcp.tool()
def author_papers(
    author_id: str,
    limit: int = 20,
    year_from: int | None = None,
    year_to: int | None = None,
) -> dict[str, object]:
    """Publications of one author (author_id from author_search)."""
    listing = paper_authors_mod.author_papers(
        author_id,
        limit=limit,
        year_from=year_from,
        year_to=year_to,
    )
    return {
        "records": [_lean_paper(r) for r in listing.records],
        "complete": listing.complete,
    }


@mcp.tool()
def web_search(
    query: str,
    num_results: int = 10,
    backend: Literal["duckduckgo", "searxng"] | None = None,
) -> list[dict[str, str]]:
    """Web search (DuckDuckGo by default; SearXNG when configured via
    SEARXNG_URL).
    """
    results = web_search_fn(query, backend=backend, num_results=num_results)
    return [{"url": r.url, "title": r.title, "snippet": r.snippet} for r in results]


@mcp.tool()
def web_fetch(
    url: str, max_chars: int = 8000, browser: bool = False
) -> dict[str, object]:
    """Fetch a page and return its extracted text. Set browser=true to route
    through headless Chrome when a site blocks plain HTTP clients (slower;
    needs a local Chrome/Chromium).
    """
    if max_chars < 1:
        raise ValueError(f"'max_chars' must be >= 1, got {max_chars}.")
    transport: Literal["auto", "curl-then-zendriver"] = (
        "curl-then-zendriver" if browser else "auto"
    )
    # The URL comes from a language model, so this server is the application
    # layer that must opt into SSRF pinning -- unpinned, the model reaches
    # loopback, the metadata endpoint, and every private-range host.
    body, _session = fetch(
        url,
        request=RequestParams(transport=transport, validated_hosts=public_host),
    )
    soup = BeautifulSoup(body, "html.parser")
    text = "\n".join(line for line in soup.get_text("\n").splitlines() if line.strip())
    truncated = len(text) > max_chars
    return {"url": url, "text": text[:max_chars], "truncated": truncated}


def main() -> int:
    """Serve the MCP tools over stdio.

    Returns:
      status: Process exit status.

    """
    mcp.run()
    return 0


# The executable lives here rather than in a package `__main__.py`: `mcp` is an
# optional extra, and `python -m wesearch` starting a server that a default
# install cannot import would make the whole package look broken. Named after
# the server, the failure explains itself.
if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
