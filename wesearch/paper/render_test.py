"""Tests for record rendering, text and structured."""

from __future__ import annotations

import pytest

from wesearch.paper.custom_types import AuthorRecord, PaperRecord
from wesearch.paper.render import (
    format_author_line,
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
        AuthorRecord(author_id="7", name="Ada", h_index=42, affiliations=("MIT",))
    )
    assert "\n" not in line
    assert "[author:7]" in line
    assert "h-index:42" in line
    assert "MIT" in line


def test_truncation_notice_only_when_truncated() -> None:
    assert truncation_notice(10, 100) != ""
    assert truncation_notice(10, 10) == ""


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
