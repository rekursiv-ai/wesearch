"""Tests for record rendering, text and structured."""

from __future__ import annotations

import pytest

from wesearch.paper.custom_types import AuthorRecord, PaperRecord
from wesearch.paper.render import (
    format_author_block,
    format_author_line,
    format_block,
    format_record,
    lean_author,
    lean_record,
    truncation_notice,
)


_RECORD = PaperRecord(
    title="Microcanonical Sampling",
    authors=("A", "B", "C", "D", "E", "F", "G"),
    year=2025,
    doi="10.1000/x",
    abstract="a" * 900,
)


def test_lean_record_caps_authors_and_clips_abstract() -> None:
    # Defaults, not re-supplied arguments: passing the declared default back in
    # asserts nothing, and moves with the code instead of pinning it.
    lean = lean_record(_RECORD)
    assert lean["authors"] == ["A", "B", "C", "D", "E", "et al."]
    abstract = lean["abstract"]
    assert isinstance(abstract, str)
    assert len(abstract) == 503  # 500 chars plus the "..." marker.
    assert abstract.endswith("...")


def test_format_block_and_author_block_include_all_present_fields() -> None:
    record = PaperRecord(
        title="T",
        authors=("A", "B"),
        year=2024,
        venue="V",
        doi="10.1/x",
        arxiv_id="2401.00001",
        abstract="abstract",
        citation_count=4,
        reference_count=5,
        open_access_pdf="https://x/pdf",
        sources=("s2", "openalex"),
    )
    assert format_block(record, abstract_chars=3) == (
        "id: arXiv:2401.00001\n"
        "doi: 10.1/x\n"
        "title: T\n"
        "authors: A, B\n"
        "year: 2024\n"
        "venue: V\n"
        "citation_count: 4\n"
        "reference_count: 5\n"
        "open_access_pdf: https://x/pdf\n"
        "sources: s2,openalex\n"
        "abstract: abs..."
    )
    author = AuthorRecord(
        author_id="1",
        name="Ada",
        aliases=("A", "Ally"),
        affiliations=("MIT", "Stanford"),
        homepage="https://ada",
        h_index=42,
        citation_count=100,
        paper_count=7,
    )
    assert format_author_block(author) == (
        "author_id: 1\nname: Ada\naliases: A, Ally\naffiliations: MIT, Stanford\n"
        "homepage: https://ada\nh_index: 42\ncitation_count: 100\n"
        "paper_count: 7"
    )
    assert lean_author(author) == {
        "author_id": "1",
        "name": "Ada",
        "affiliations": ["MIT", "Stanford"],
        "h_index": 42,
        "citation_count": 100,
        "paper_count": 7,
    }


def test_lean_record_drops_empty_fields() -> None:
    assert lean_record(PaperRecord(title="T")) == {"title": "T"}


def test_lean_author_drops_empty_fields() -> None:
    assert lean_author(AuthorRecord(author_id="1", name="N")) == {
        "author_id": "1",
        "name": "N",
    }


def test_lean_record_rejects_non_positive_caps() -> None:
    """A zero/negative cap is a caller bug, not a request for everything.

    ``abstract_chars=0`` -- the plainest way to ask for no abstract -- used to
    return the full one, and ``author_limit=-1`` sliced an author off and then
    appended "et al." claiming there were more.
    """
    for kwargs in ({"author_limit": 0}, {"author_limit": -1}, {"abstract_chars": 0}):
        with pytest.raises(ValueError, match="must be >= 1"):
            lean_record(_RECORD, **kwargs)


def test_format_record_without_id_and_authors() -> None:
    assert format_record(PaperRecord(title="T")) == "[no-id] T - unknown, ?, ?"


def test_format_record_exactly_three_authors_has_no_suffix() -> None:
    record = PaperRecord(title="T", authors=("A", "B", "C"))
    assert "A, B, C, ?, ?" in format_record(record)


def test_format_record_emits_all_metadata_and_author_suffix() -> None:
    record = PaperRecord(
        title="T",
        authors=("A", "B", "C", "D"),
        year=2024,
        venue="V",
        doi="10.1/x",
        arxiv_id="2401.00001",
        abstract="line one\nline two",
        citation_count=4,
        reference_count=5,
        open_access_pdf="https://x/pdf",
        sources=("s2", "openalex"),
        is_influential=True,
    )
    assert format_record(record) == (
        "[doi:10.1/x | arXiv:2401.00001] T - A, B, C +1, 2024, V - "
        "cites:4 · refs:5 · OA · sources: s2,openalex · influential\n"
        "    abstract:\n    line one\n    line two"
    )


def test_text_and_lean_renderings_agree_on_identity() -> None:
    """Both renderings surface the same identifiers for the same record.

    The two used to live in different packages -- the tools rendered text, the
    MCP server built dicts -- and drifted: different author truncation, and the
    structured form silently omitted fields the text form showed. Anything that
    identifies a paper must appear in both.
    """
    text = format_record(_RECORD)
    lean = lean_record(_RECORD)
    assert _RECORD.title in text
    assert lean["title"] == _RECORD.title
    assert "doi:10.1000/x" in text
    assert lean["doi"] == "10.1000/x"
    assert str(_RECORD.year) in text
    assert lean["year"] == _RECORD.year


def test_both_renderings_emit_sources() -> None:
    """``sources`` is the field the two renderings most recently drifted on."""
    rec = PaperRecord(title="T", sources=("s2", "openalex"))
    assert "sources: s2,openalex" in format_record(rec)
    assert lean_record(rec)["sources"] == ["s2", "openalex"]


def test_format_author_line_is_one_greppable_line() -> None:
    line = format_author_line(
        AuthorRecord(author_id="7", name="Ada", h_index=42, affiliations=("MIT",)),
    )
    assert "\n" not in line
    assert "[author:7]" in line
    assert "h-index:42" in line
    assert "MIT" in line


def test_truncation_notice_only_when_truncated() -> None:
    assert (
        truncation_notice(10, 100)
        == "\n... (showing 10 of 100; tighten filters for more)"
    )
    assert truncation_notice(10, 10) == ""
    assert truncation_notice(10, 0) == ""


def test_format_record_abstract_cap_boundaries() -> None:
    record = PaperRecord(title="T", abstract=" abc ")
    assert format_record(record, abstract_chars=5) == (
        "[no-id] T - unknown, ?, ?\n    abstract:\n     abc "
    )
    with pytest.raises(ValueError, match="must be >= 1"):
        format_record(record, abstract_chars=0)


def test_render_boundaries_and_all_optional_fields() -> None:
    assert lean_record(PaperRecord(title="T", authors=("A",)), author_limit=1)[
        "authors"
    ] == ["A"]
    assert (
        lean_record(PaperRecord(title="T", abstract="abcdef"), abstract_chars=1)[
            "abstract"
        ]
        == "a..."
    )
    assert (
        lean_record(PaperRecord(title="T", abstract=" abc"), abstract_chars=2)[
            "abstract"
        ]
        == " a..."
    )
    assert truncation_notice(-1, 0) == ""
    assert truncation_notice(0, 1) == "\n... (showing 0 of 1; tighten filters for more)"

    record = PaperRecord(
        title="XXXX",
        authors=("A",),
        year=2024,
        venue="V",
        doi="10.1/x",
        arxiv_id="2401.00001",
        abstract="a",
        citation_count=4,
        reference_count=5,
        open_access_pdf="https://x/pdf",
        sources=("s2", "openalex"),
        is_influential=True,
    )
    assert lean_record(record) == {
        "title": "XXXX",
        "authors": ["A"],
        "year": 2024,
        "venue": "V",
        "doi": "10.1/x",
        "arxiv_id": "2401.00001",
        "citation_count": 4,
        "reference_count": 5,
        "open_access_pdf": "https://x/pdf",
        "is_influential": True,
        "sources": ["s2", "openalex"],
        "abstract": "a",
    }


def test_author_line_includes_each_metric() -> None:
    assert (
        format_author_line(
            AuthorRecord(
                author_id="7",
                name="Ada",
                h_index=42,
                citation_count=100,
                paper_count=7,
                affiliations=("MIT",),
            ),
        )
        == "[author:7] Ada - h-index:42 cites:100 papers:7 - MIT"
    )


def test_format_block_uses_unknown_for_empty_authors() -> None:
    assert "authors: unknown" in format_block(PaperRecord(title="T"))


def test_lean_author_keeps_xxxx_values() -> None:
    assert lean_author(AuthorRecord(author_id="XXXX", name="XXXX")) == {
        "author_id": "XXXX",
        "name": "XXXX",
    }


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
