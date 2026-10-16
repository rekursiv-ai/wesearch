"""Render paper and author records as agent-readable text or compact dicts.

Text renderings are one greppable line per record, or a multi-line detail
block; ``lean_*`` renderings are JSON-ready dicts. Pure functions, no I/O.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from wesearch.paper.types import AuthorRecord, PaperRecord


__all__ = [
    "format_author_block",
    "format_author_line",
    "format_block",
    "format_record",
    "lean_author",
    "lean_record",
    "truncation_notice",
]


def format_record(rec: PaperRecord, abstract_chars: int | None = None) -> str:
    """Format a paper as one greppable line, followed by an indented abstract.

    Args:
      rec: Paper record to format.
      abstract_chars: Abstract character cap, or None for the full abstract.

    Returns:
      formatted: ``[ids] title - authors, year, venue - meta`` header, plus
        the abstract when present.

    Raises:
      ValueError: When ``abstract_chars`` < 1 and the abstract exceeds it.

    """
    id_block = _id_prefix(rec)
    year_str = str(rec.year) if rec.year is not None else "?"
    venue_str = rec.venue or "?"
    meta_parts: list[str] = []
    if rec.citation_count is not None:
        meta_parts.append(f"cites:{rec.citation_count}")
    if rec.reference_count is not None:
        meta_parts.append(f"refs:{rec.reference_count}")
    if rec.open_access_pdf:
        meta_parts.append("OA")
    if rec.sources:
        meta_parts.append("sources: " + ",".join(rec.sources))
    if rec.is_influential is True:
        meta_parts.append("influential")
    meta = " · ".join(meta_parts)

    header = (
        f"{id_block} {rec.title} - {_format_authors(rec.authors)}, "
        f"{year_str}, {venue_str}"
    )
    if meta:
        header += f" - {meta}"
    abstract = _trim_abstract(rec.abstract, abstract_chars)
    if abstract:
        body = "\n".join(f"    {line}" for line in abstract.splitlines())
        return f"{header}\n    abstract:\n{body}"
    return header


def format_block(rec: PaperRecord, abstract_chars: int | None = None) -> str:
    """Render a paper as ``key: value`` lines, omitting empty fields.

    Args:
      rec: Paper record to format.
      abstract_chars: Abstract character cap, or None for the full abstract.

    Returns:
      formatted: One line per present field (id, doi, title, authors, ...).

    Raises:
      ValueError: When ``abstract_chars`` < 1 and the abstract exceeds it.

    """
    lines: list[str] = []
    if rec.arxiv_id:
        lines.append(f"id: arXiv:{rec.arxiv_id}")
    if rec.doi:
        lines.append(f"doi: {rec.doi}")
    lines.append(f"title: {rec.title}")
    lines.append(f"authors: {', '.join(rec.authors) if rec.authors else 'unknown'}")
    if rec.year is not None:
        lines.append(f"year: {rec.year}")
    if rec.venue:
        lines.append(f"venue: {rec.venue}")
    if rec.citation_count is not None:
        lines.append(f"citation_count: {rec.citation_count}")
    if rec.reference_count is not None:
        lines.append(f"reference_count: {rec.reference_count}")
    if rec.open_access_pdf:
        lines.append(f"open_access_pdf: {rec.open_access_pdf}")
    if rec.sources:
        lines.append(f"sources: {','.join(rec.sources)}")
    abstract = _trim_abstract(rec.abstract, abstract_chars)
    if abstract:
        lines.append(f"abstract: {abstract}")
    return "\n".join(lines)


def format_author_line(rec: AuthorRecord) -> str:
    """Format an author as one greppable line.

    Args:
      rec: Author record to format.

    Returns:
      formatted: ``[author:id] name - metrics - first affiliation``.

    """
    id_block = f"[author:{rec.author_id}]"
    meta: list[str] = []
    if rec.h_index is not None:
        meta.append(f"h-index:{rec.h_index}")
    if rec.citation_count is not None:
        meta.append(f"cites:{rec.citation_count}")
    if rec.paper_count is not None:
        meta.append(f"papers:{rec.paper_count}")
    line = f"{id_block} {rec.name}"
    if meta:
        line += f" - {' '.join(meta)}"
    if rec.affiliations:
        line += f" - {rec.affiliations[0]}"
    return line


def format_author_block(rec: AuthorRecord) -> str:
    """Render an author as ``key: value`` lines, omitting empty fields.

    Args:
      rec: Author record to format.

    Returns:
      formatted: One line per present field (author_id, name, aliases, ...).

    """
    lines: list[str] = [
        f"author_id: {rec.author_id}",
        f"name: {rec.name}",
    ]
    if rec.aliases:
        lines.append(f"aliases: {', '.join(rec.aliases)}")
    if rec.affiliations:
        lines.append(f"affiliations: {', '.join(rec.affiliations)}")
    if rec.homepage:
        lines.append(f"homepage: {rec.homepage}")
    if rec.h_index is not None:
        lines.append(f"h_index: {rec.h_index}")
    if rec.citation_count is not None:
        lines.append(f"citation_count: {rec.citation_count}")
    if rec.paper_count is not None:
        lines.append(f"paper_count: {rec.paper_count}")
    return "\n".join(lines)


def truncation_notice(shown: int, total: int) -> str:
    """Build a ``... showing N of M`` suffix for paginated output.

    Args:
      shown: Number of records displayed.
      total: Number of records available.

    Returns:
      notice: The suffix, or ``""`` when nothing was withheld.

    """
    if total > shown and total > 0:
        return f"\n... (showing {shown} of {total}; tighten filters for more)"
    return ""


def lean_record(
    rec: PaperRecord,
    *,
    abstract_chars: int = 500,
    author_limit: int = 5,
) -> dict[str, object]:
    """Return one paper as a compact dict: empty fields dropped, abstract clipped.

    Args:
      rec: Paper record to format.
      abstract_chars: Abstract character cap.
      author_limit: Maximum authors to include; ``"et al."`` marks the rest.

    Returns:
      compact: The record's non-empty fields.

    Raises:
      ValueError: When ``abstract_chars`` or ``author_limit`` is < 1.

    """
    if author_limit < 1:
        raise ValueError(f"'author_limit' must be >= 1, got {author_limit}.")
    if abstract_chars < 1:
        raise ValueError(f"'abstract_chars' must be >= 1, got {abstract_chars}.")
    authors: list[str] = list(rec.authors[:author_limit])
    if len(rec.authors) > author_limit:
        authors.append("et al.")
    fields: dict[str, object] = {
        "title": rec.title,
        "authors": authors,
        "year": rec.year,
        "venue": rec.venue,
        "doi": rec.doi,
        "arxiv_id": rec.arxiv_id,
        "citation_count": rec.citation_count,
        "reference_count": rec.reference_count,
        "open_access_pdf": rec.open_access_pdf,
        "is_influential": rec.is_influential,
        "sources": list(rec.sources),
        "abstract": _trim_abstract(rec.abstract, abstract_chars),
    }
    return {k: v for k, v in fields.items() if v not in (None, [], "")}


def lean_author(rec: AuthorRecord) -> dict[str, object]:
    """Return one author as a compact dict, mirroring :func:`format_author_line`.

    Args:
      rec: Author record to format.

    Returns:
      compact: The record's non-empty fields.

    """
    fields: dict[str, object] = {
        "author_id": rec.author_id,
        "name": rec.name,
        "affiliations": list(rec.affiliations),
        "h_index": rec.h_index,
        "citation_count": rec.citation_count,
        "paper_count": rec.paper_count,
    }
    return {k: v for k, v in fields.items() if v not in (None, [], "")}


def _id_prefix(rec: PaperRecord) -> str:
    """Bracketed identifier prefix: ``[doi:... | arXiv:...]`` / subset."""
    parts: list[str] = []
    if rec.doi:
        parts.append(f"doi:{rec.doi}")
    if rec.arxiv_id:
        parts.append(f"arXiv:{rec.arxiv_id}")
    inner = " | ".join(parts) if parts else "no-id"
    return f"[{inner}]"


def _format_authors(authors: tuple[str, ...], limit: int = 3) -> str:
    """Render first-``limit`` authors with ``+N`` suffix for the rest."""
    if not authors:
        return "unknown"
    shown = ", ".join(authors[:limit])
    extra = len(authors) - limit
    if extra > 0:
        return f"{shown} +{extra}"
    return shown


def _trim_abstract(abstract: str, cap: int | None) -> str:
    """Cut an abstract to ``cap`` characters; ``None`` means uncapped."""
    if cap is None or len(abstract) <= cap:
        return abstract
    if cap < 1:
        raise ValueError(f"'abstract_chars' must be >= 1, got {cap}.")
    return abstract[:cap].rstrip() + "..."
