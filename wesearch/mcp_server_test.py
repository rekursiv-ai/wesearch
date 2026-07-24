"""Hermetic tests for the MCP server's tool wrappers (no network)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import asyncio
import importlib

import pytest


# The MCP server needs the optional [mcp] extra. Skip the whole module when it is
# absent (e.g. a plain `uv run pytest` without --all-extras) instead of erroring
# on collection; CI installs the extra and exercises these tests.
pytest.importorskip("mcp.server.fastmcp")

from wesearch import mcp_server
from wesearch.paper import (
    authors as paper_authors_mod,
    details as paper_details_mod,
    fetch as paper_fetch_mod,
    search as paper_search_mod,
)
from wesearch.paper.custom_types import AuthorRecord, PaperRecord
from wesearch.paper.search import SearchResult as PaperSearchResult
from wesearch.search import SearchResult as WebSearchResult


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _returns[T](value: T) -> Callable[..., T]:
    """Return a typed stub callable for monkeypatching.

    A bare ``lambda *_a, **_k: value`` loses its signature to the type checker
    (reportUnknownLambdaType). This preserves the return type so patched calls
    stay fully typed.
    """
    return lambda *_args, **_kwargs: value


_RECORD = PaperRecord(
    title="Microcanonical Sampling",
    authors=("A", "B", "C", "D", "E", "F", "G"),
    year=2025,
    doi="10.1000/x",
    abstract="a" * 900,
)


def test_lean_paper_caps_authors_and_clips_abstract() -> None:
    lean = mcp_server._lean_paper(_RECORD)
    assert lean["authors"] == ["A", "B", "C", "D", "E", "et al."]
    abstract = lean["abstract"]
    assert isinstance(abstract, str)
    assert len(abstract) == mcp_server._ABSTRACT_CHARS
    assert abstract.endswith("…")


def test_lean_paper_drops_empty_fields() -> None:
    lean = mcp_server._lean_paper(PaperRecord(title="T"))
    assert lean == {"title": "T"}


def test_clip_passes_short_text_through() -> None:
    assert mcp_server._clip("short", 100) == "short"
    assert mcp_server._clip(None, 100) is None


def test_paper_search_shapes_result(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = PaperSearchResult(records=[_RECORD], total=41, complete=False)
    monkeypatch.setattr(paper_search_mod, "search", _returns(fake))
    out = mcp_server.paper_search("mclmc")
    assert out["total"] == 41
    assert out["complete"] is False
    records = out["records"]
    assert isinstance(records, list)
    assert records[0]["title"] == "Microcanonical Sampling"


def test_paper_details_normalizes_id(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_metadata(kind: str, canonical: str) -> PaperRecord:
        seen["id"] = (kind, canonical)
        return _RECORD

    monkeypatch.setattr(paper_details_mod, "metadata", fake_metadata)
    out = mcp_server.paper_details("https://arxiv.org/abs/2503.01234v2")
    assert seen["id"] == ("arxiv", "2503.01234v2")
    assert out["id"] == "arxiv:2503.01234v2"


def test_paper_pdf_writes_cache_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        paper_fetch_mod,
        "download",
        _returns((b"%PDF-fake", "arxiv")),
    )
    monkeypatch.setattr(mcp_server, "cache_dir", _returns(tmp_path))
    out = mcp_server.paper_pdf("arxiv:2503.01234")
    path = out["path"]
    assert isinstance(path, str)
    assert path.endswith(".pdf")
    with open(path, "rb") as handle:  # noqa: PTH123 -- symmetry with write path is irrelevant here.
        assert handle.read() == b"%PDF-fake"
    assert out == {"path": path, "bytes": 9, "source": "arxiv"}


def test_author_search_shapes_result(monkeypatch: pytest.MonkeyPatch) -> None:
    record = AuthorRecord(author_id="123", name="Ada", h_index=40)
    fake = paper_authors_mod.AuthorSearchResult(records=[record], total=1)
    monkeypatch.setattr(paper_authors_mod, "search_authors", _returns(fake))
    out = mcp_server.author_search("ada")
    assert out["records"] == [{"author_id": "123", "name": "Ada", "h_index": 40}]


def test_web_search_returns_lean_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [WebSearchResult(url="https://e.co", title="E", snippet="s")]
    monkeypatch.setattr(mcp_server, "web_search_fn", _returns(rows))
    out = mcp_server.web_search("q")
    assert out == [{"url": "https://e.co", "title": "E", "snippet": "s"}]


def test_web_fetch_extracts_and_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    # importlib, not attribute traversal: the package's ``fetch`` re-export
    # shadows the submodule of the same name.
    fetch_module = importlib.import_module("wesearch.fetch.fetch")
    html = b"<html><body><p>Hello</p><script>no</script><p>World</p></body></html>"
    monkeypatch.setattr(fetch_module, "fetch", _returns((html, object())))
    out = mcp_server.web_fetch("https://e.co", max_chars=7)
    assert out["truncated"] is True
    text = out["text"]
    assert isinstance(text, str)
    assert text.startswith("Hello")


def test_dedupe_drops_fusion_duplicates() -> None:
    first = PaperRecord(title="CogToM", arxiv_id="2601.15628")
    by_title = PaperRecord(title="cogtom ")
    by_id = PaperRecord(title="CogToM: a benchmark", arxiv_id="2601.15628")
    distinct = PaperRecord(title="MuMA-ToM")
    unique = mcp_server._dedupe([first, by_title, by_id, distinct])
    assert unique == [first, distinct]


def test_paper_search_dedupes_records(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = PaperSearchResult(records=[_RECORD, _RECORD], total=2, complete=True)
    monkeypatch.setattr(paper_search_mod, "search", _returns(fake))
    out = mcp_server.paper_search("dupes")
    records = out["records"]
    assert isinstance(records, list)
    assert len(cast("list[object]", records)) == 1


def test_all_tools_registered() -> None:
    tools = asyncio.run(mcp_server.mcp.list_tools())
    names = {tool.name for tool in tools}
    assert names == {
        "paper_search",
        "paper_details",
        "paper_references",
        "paper_citations",
        "paper_pdf",
        "author_search",
        "author_papers",
        "web_search",
        "web_fetch",
    }
