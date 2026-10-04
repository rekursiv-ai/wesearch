"""Tests for backend-agnostic paper and author records."""

from __future__ import annotations

from wesearch.paper.custom_types import AuthorRecord, PaperRecord


def test_paper_merge_fills_empty_primary_values() -> None:
    primary = PaperRecord(title="", sources=("s2",))
    other = PaperRecord(
        title="T",
        authors=("A",),
        year=2024,
        venue="V",
        doi="10.1/x",
        arxiv_id="2401.00001",
        abstract="A",
        citation_count=4,
        reference_count=5,
        open_access_pdf="https://x/pdf",
        is_influential=True,
        sources=("openalex",),
    )
    merged = primary.merge(other)
    assert merged.title == "T"
    assert merged.authors == ("A",)
    assert merged.year == 2024
    assert merged.venue == "V"
    assert merged.doi == "10.1/x"
    assert merged.arxiv_id == "2401.00001"
    assert merged.abstract == "A"
    assert merged.citation_count == 4
    assert merged.reference_count == 5
    assert merged.open_access_pdf == "https://x/pdf"
    assert merged.is_influential is True


def test_author_record_defaults_are_sparse() -> None:
    assert AuthorRecord(author_id="1", name="Ada") == AuthorRecord(
        author_id="1",
        name="Ada",
    )


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
