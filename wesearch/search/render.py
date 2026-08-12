"""Render search results as agent-readable text and as lean dicts.

The search counterpart to :mod:`wesearch.paper.render`, and here for the
same reason: every surface that shows a result to a model needs the SAME
rendering. The sagent tool grew a text renderer that reached into each
category's fields while the MCP server flattened every hit to
``url``/``title``/``snippet`` -- so the same query answered on one surface with
a paper's DOI and authors and on the other with neither.

Both renderings read one field table, so a category's extra fields cannot
reach one surface and not the other.

Pure functions over records: no tool framework, no MCP, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from wesearch.search.custom_types import (
    CodeResult,
    FileResult,
    ImageResult,
    MapResult,
    MediaResult,
    PackageResult,
    PaperResult,
    SearchResult,
    TorrentResult,
    VideoResult,
)


__all__ = ["detail_fields", "format_result", "lean_result"]


def format_result(result: SearchResult) -> str:
    """Render one result as a markdown link plus its structured fields.

    Args:
      result: The record to render.

    Returns:
      text: ``[title](url)`` followed by the snippet and, when the concrete
        subclass carries extra structure, an indented detail line.

    """
    head = f"[{result.title}]({result.url})"
    # Empty values dropped BEFORE joining: a field a given engine did not
    # report would otherwise render as a bare separator run (``· · ·``).
    # ``is not None`` and not ``if v``: a reported ZERO is a fact ("cites:0",
    # "seed:0"), and truthiness deleted it while keeping an empty string.
    kept = [
        _labelled(k, v)
        for k, v in detail_fields(result).items()
        if v is not None and v != ""
    ]
    detail = "  " + " · ".join(p for p in kept if p) if any(kept) else ""
    body = "\n".join(part for part in (result.snippet, detail) if part)
    return f"{head}\n{body}" if body else head


def _labelled(
    name: str,
    value: object,
    *,
    prefixes: Mapping[str, str] = MappingProxyType(
        {"doi": "doi:", "citations": "cites:", "seed": "seed:", "leech": "leech:"}
    ),
    suffixes: Mapping[str, str] = MappingProxyType({"views": " views"}),
) -> str:
    """Render one detail field for the TEXT surface, with its reading label.

    The labels live on this function rather than beside the data, because
    only the text rendering wants them: ``lean_result`` feeds a JSON protocol,
    where ``"doi": "doi:10.1/x"`` forces every client to strip a prefix to
    recover the identifier -- the mistake ``paper/render.py`` avoids by
    emitting ``rec.doi`` raw. Passed as defaults, not module state, so a
    caller can respell them without reaching through a global.
    """
    return f"{prefixes.get(name, '')}{value}{suffixes.get(name, '')}"


def lean_result(result: SearchResult) -> dict[str, object]:
    """Return one result as a compact dict, empty fields dropped.

    The structured counterpart to :func:`format_result`, for a caller whose
    protocol carries JSON rather than text -- mirroring
    :func:`wesearch.paper.render.lean_record`.

    Args:
      result: The record to render.

    Returns:
      fields: ``url``/``title``/``snippet`` plus whatever the concrete
        subclass carries; keys with empty values are omitted.

    """
    fields: dict[str, object] = {
        "url": result.url,
        "title": result.title,
        "snippet": result.snippet,
        **detail_fields(result),
    }
    # Enumerated emptiness, not truthiness: a reported ``0`` (zero citations,
    # zero seeders) is a fact the caller asked for, and ``if v`` deleted it.
    return {k: v for k, v in fields.items() if v not in (None, [], (), "")}


def detail_fields(result: SearchResult) -> Mapping[str, object]:
    """Return the category-specific fields of ``result``, keyed by name.

    Dispatches on the concrete :class:`SearchResult` subclass, so a category's
    extra structure (a paper's authors/DOI, an image's source URL, a place's
    coordinates) is named once for every renderer. Empty values are dropped by
    the callers, which differ in how they present them.

    Ordered most-derived first: :class:`VideoResult` subclasses
    :class:`MediaResult`, so testing the base first would render a video as a
    plain media result and silently lose its view count and channel.
    """
    if isinstance(result, PaperResult):
        authors = ", ".join(result.authors[:3]) + (
            " +" if len(result.authors) > 3 else ""
        )
        return {
            "authors": authors if result.authors else "",
            "journal": result.journal,
            "year": result.published.year if result.published else None,
            "doi": result.doi,
            "citations": result.citations,
            "pdf_url": result.pdf_url,
        }
    if isinstance(result, ImageResult):
        return {
            "image_url": result.image_url,
            "resolution": result.resolution,
            "img_format": result.img_format,
            "source": result.source,
        }
    if isinstance(result, VideoResult):
        return {
            "author": result.author,
            "length": result.length,
            "views": result.views,
            "iframe_url": result.iframe_url,
        }
    if isinstance(result, MediaResult):
        return {
            "published": str(result.published.date()) if result.published else "",
            "length": result.length,
            "url_media": result.audio_url or result.iframe_url,
        }
    if isinstance(result, MapResult):
        # BOTH or neither: guarding on latitude alone rendered "1.0,None",
        # which reads as a coordinate pair and is not one.
        coords = (
            f"{result.latitude},{result.longitude}"
            if result.latitude is not None and result.longitude is not None
            else ""
        )
        return {"coordinates": coords, "address": ", ".join(result.address.values())}
    if isinstance(result, PackageResult):
        return {
            "package_name": result.package_name,
            "version": result.version,
            "license_name": result.license_name,
            "homepage": result.homepage or result.source_code_url,
        }
    if isinstance(result, CodeResult):
        return {
            "repository": result.repository,
            "filename": result.filename,
            "code_language": result.code_language,
        }
    if isinstance(result, FileResult):
        return {
            "filename": result.filename,
            "size": result.size,
            "mimetype": result.mimetype,
        }
    if isinstance(result, TorrentResult):
        return {
            "filesize": result.filesize,
            "seed": result.seed,
            "leech": result.leech,
            "magnet_url": result.magnet_url,
        }
    return {}
