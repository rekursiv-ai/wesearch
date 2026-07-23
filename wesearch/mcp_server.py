"""MCP server exposing wesearch to agent clients over stdio.

Run with ``wesearch-mcp`` (installed via the ``mcp`` extra: ``pip install
wesearch[mcp]``). Every tool is a thin wrapper over a public wesearch
function, with outputs reshaped to be token-lean for model consumption:
abstracts are truncated, empty fields dropped, and PDF bytes are written to
the user cache directory rather than returned inline.

Tools are synchronous by design -- wesearch's rate limiting and cookie/UA
profile state are cross-process safe on disk, so each MCP client session can
run its own server process without coordination. Errors surface to the
client as MCP tool errors carrying the underlying exception message
(``BotDetectionError`` includes its recovery guidance).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from mcp.server.fastmcp import FastMCP

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
    from wesearch.paper.custom_types import AuthorRecord, PaperRecord

_ABSTRACT_CHARS = 500
_DETAIL_ABSTRACT_CHARS = 1500
_MAX_AUTHORS = 5

mcp = FastMCP(
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


def _lean_paper(
    record: PaperRecord, *, abstract_chars: int = _ABSTRACT_CHARS
) -> dict[str, object]:
    """Compact dict for one paper: empty fields dropped, abstract clipped."""
    authors: list[str] = list(record.authors[:_MAX_AUTHORS])
    if len(record.authors) > _MAX_AUTHORS:
        authors.append("et al.")
    fields: dict[str, object] = {
        "title": record.title,
        "authors": authors,
        "year": record.year,
        "venue": record.venue,
        "doi": record.doi,
        "arxiv_id": record.arxiv_id,
        "citation_count": record.citation_count,
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
    source: Literal["fused", "s2", "openalex", "searxng"] = "fused",
    limit: int = 10,
    year_from: int | None = None,
    year_to: int | None = None,
    open_access_only: bool = False,
) -> dict[str, object]:
    """Search scholarly literature. The default "fused" source rank-fuses
    Semantic Scholar and OpenAlex and degrades gracefully if one is down
    (complete=false means a backend was lost or results were truncated).
    """
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
    lean = _lean_paper(record, abstract_chars=_DETAIL_ABSTRACT_CHARS)
    lean["id"] = f"{kind}:{canonical}"
    return lean


@mcp.tool()
def paper_references(paper_id: str, limit: int = 20) -> dict[str, object]:
    """Papers this paper cites (its bibliography)."""
    kind, canonical = normalize_id(paper_id)
    listing = paper_details_mod.references(kind, canonical, limit=limit)
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
) -> dict[str, object]:
    """Papers that cite this paper. influential_only keeps only citations
    Semantic Scholar flags as influential.
    """
    kind, canonical = normalize_id(paper_id)
    listing = paper_details_mod.citations(
        kind,
        canonical,
        limit=limit,
        influential_only=influential_only,
        year_from=year_from,
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
    target = target_dir / f"{id_slug(kind, canonical)}.pdf"
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
    from bs4 import (  # noqa: PLC0415 -- heavy import deferred to first use.
        BeautifulSoup,
    )

    from wesearch.fetch.fetch import (  # noqa: PLC0415 -- deferred with bs4.
        RequestParams,
        fetch,
    )

    transport: Literal["auto", "curl-then-zendriver"] = (
        "curl-then-zendriver" if browser else "auto"
    )
    body, _session = fetch(url, request=RequestParams(transport=transport))
    soup = BeautifulSoup(body, "html.parser")
    text = "\n".join(line for line in soup.get_text("\n").splitlines() if line.strip())
    truncated = len(text) > max_chars
    return {"url": url, "text": text[:max_chars], "truncated": truncated}


def main() -> None:
    """Console-script entry point: serve over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
