"""Hermetic tests for the MCP server's tool wrappers (no network)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import asyncio

import pytest


# The MCP server needs the optional [mcp] extra. Skip the whole module when it is
# absent (e.g. a plain `uv run pytest` without --all-extras) instead of erroring
# on collection; CI installs the extra and exercises these tests.
pytest.importorskip("mcp.server")

from wesearch.mcp import server as mcp_server
from wesearch.paper import (
    authors as paper_authors_mod,
    details as paper_details_mod,
    fetch as paper_fetch_mod,
    search as paper_search_mod,
)
from wesearch.paper.custom_types import AuthorRecord, PaperRecord
from wesearch.paper.search import SearchResult as PaperSearchResult
from wesearch.search.custom_types import SearchResult as WebSearchResult
from wesearch.web import _KIND_HTML, WebFetchResult, _extract_text


if TYPE_CHECKING:
    from collections.abc import Callable


def _returns[T](value: T) -> Callable[..., T]:
    """Return a typed stub callable for monkeypatching.

    A bare ``lambda *_a, **_k: value`` loses its signature to the type checker
    (reportUnknownLambdaType). This preserves the return type so patched calls
    stay fully typed.
    """
    return lambda *_args, **_kwargs: value


def _fetch_web_returning(
    html: bytes, *, max_chars: int
) -> Callable[..., WebFetchResult]:
    """Return a ``fetch_web`` stub that renders ``html`` through the real extractor."""

    def _stub(
        url: str, *, max_chars: int = max_chars, policy: object = None
    ) -> WebFetchResult:
        del policy
        text = _extract_text(html, kind=_KIND_HTML, url=url)
        return WebFetchResult(
            text=text[:max_chars],
            url=url,
            kind=_KIND_HTML,
            truncated=len(text) > max_chars,
        )

    return _stub


_RECORD = PaperRecord(
    title="Microcanonical Sampling",
    authors=("A", "B", "C", "D", "E", "F", "G"),
    year=2025,
    doi="10.1000/x",
    abstract="a" * 900,
)


def test_paper_search_shapes_result(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = PaperSearchResult(records=[_RECORD], total=41, complete=False)
    monkeypatch.setattr(paper_search_mod, "search", _returns(fake))
    out = mcp_server.paper_search("mclmc")
    assert out["total"] == 41
    assert out["complete"] is False
    records = out["records"]
    assert isinstance(records, list)
    first = cast("dict[str, object]", records[0])
    assert first["title"] == "Microcanonical Sampling"


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
    html = b"<html><body><p>Hello</p><script>no</script><p>World</p></body></html>"
    monkeypatch.setattr(
        mcp_server, "fetch_web", _fetch_web_returning(html, max_chars=7)
    )
    out = mcp_server.web_fetch("https://e.co", max_chars=7)
    assert out["truncated"] is True
    text = out["text"]
    assert isinstance(text, str)
    assert "Hello" in text


def test_web_fetch_routes_through_the_shared_render_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The MCP tool renders via ``fetch_web``, not a private HTML dump.

    It hand-rolled a BeautifulSoup text dump for a while, so the MCP surface
    silently lacked the provider dispatch, feed rendering, and extractor
    selection every other caller got. Pinned by asserting the ``kind`` only
    ``fetch_web`` can report.
    """
    captured: dict[str, object] = {}

    def _fake(
        url: str, *, max_chars: int | None = None, policy: object = None
    ) -> object:
        del max_chars
        captured["extractor"] = getattr(policy, "extractor", None)
        return WebFetchResult(text="body", url=url, kind="html", truncated=False)

    monkeypatch.setattr(mcp_server, "fetch_web", _fake)
    out = mcp_server.web_fetch("https://e.co", extractor="trafilatura")
    assert out["kind"] == "html"
    assert captured["extractor"] == "trafilatura"


def test_web_fetch_treats_the_model_url_as_untrusted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An agent-supplied URL is the SSRF trust boundary. The server no longer has
    # to opt in -- "untrusted" is the default -- so what it owes is simply not
    # opting OUT, which this pins.
    captured: dict[str, object] = {}

    def _fake_fetch(
        url: str, *, max_chars: int | None = None, policy: object = None
    ) -> WebFetchResult:
        del max_chars
        captured["trust"] = getattr(policy, "trust", None)
        return WebFetchResult(text="ok", url=url, kind="html", truncated=False)

    monkeypatch.setattr(mcp_server, "fetch_web", _fake_fetch)
    mcp_server.web_fetch("https://e.co")
    # The validation itself is pinned in fetch/common_test.py; what this server
    # owes is leaving the safe default alone.
    assert captured["trust"] == "untrusted"


def test_web_fetch_rejects_a_nonpositive_max_chars() -> None:
    # A negative bound slices from the END, returning nearly the whole page
    # while still reporting truncated=True.
    with pytest.raises(ValueError, match="max_chars"):
        mcp_server.web_fetch("https://e.co", max_chars=-1)


def test_paper_search_rejects_a_nonpositive_limit() -> None:
    # Unbounded, this reaches paginate's own ValueError, which escapes the
    # PaperError contract every caller catches.
    with pytest.raises(ValueError, match="limit"):
        mcp_server.paper_search("q", limit=0)


def test_paper_pdf_gives_colliding_slugs_distinct_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # ``id_slug`` maps every unsafe character to ``_``, so these two valid DOIs
    # slug identically and one paper's bytes overwrote the other's.
    monkeypatch.setattr(mcp_server, "cache_dir", _returns(tmp_path))
    monkeypatch.setattr(paper_fetch_mod, "download", _returns((b"%PDF-a", "oa")))
    first = mcp_server.paper_pdf("10.1234/a_b")["path"]
    monkeypatch.setattr(paper_fetch_mod, "download", _returns((b"%PDF-b", "oa")))
    second = mcp_server.paper_pdf("10.1234/a/b")["path"]
    assert first != second
    assert isinstance(first, str)
    assert Path(first).read_bytes() == b"%PDF-a"


def test_paper_pdf_survives_a_long_doi_suffix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An unbounded slug exceeds the 255-byte filename limit and raises OSError.
    monkeypatch.setattr(mcp_server, "cache_dir", _returns(tmp_path))
    monkeypatch.setattr(paper_fetch_mod, "download", _returns((b"%PDF-x", "oa")))
    path = mcp_server.paper_pdf("10.1234/" + "a" * 300)["path"]
    assert isinstance(path, str)
    assert Path(path).read_bytes() == b"%PDF-x"


def test_paper_search_emits_library_records_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Deduplication and the ``limit`` trim belong to the library, which owns the
    # identifier graph and the ranking. A server-side pass can only re-decide
    # identity from the lean fields it kept -- which is how the prior casefolded
    # -title dedup destroyed distinct papers that share a title ("Discussion",
    # "Editorial introduction"). Emit exactly what the library returned.
    distinct = PaperRecord(title="Discussion", doi="10.1/a", year=1998)
    namesake = PaperRecord(title="Discussion", doi="10.1/b", year=1997)
    fake = PaperSearchResult(records=[distinct, namesake], total=2, complete=True)
    monkeypatch.setattr(paper_search_mod, "search", _returns(fake))
    out = mcp_server.paper_search("discussion")
    records = out["records"]
    assert isinstance(records, list)
    assert len(cast("list[object]", records)) == 2


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
