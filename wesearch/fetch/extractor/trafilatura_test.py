"""Tests for the trafilatura extractor."""

from __future__ import annotations

from wesearch.fetch.extractor.trafilatura import extract_trafilatura


def test_extracts_article_prose() -> None:
    """An article-shaped page yields its body."""
    body = "This is a paragraph of article prose long enough to be scored as content."
    text = extract_trafilatura(
        f"<html><body><article><p>{body}</p></article></body></html>"
    )
    assert body in text


def test_empty_document_yields_no_text() -> None:
    """A document with no article yields empty text, not ``None``."""
    assert extract_trafilatura("") == ""


if __name__ == "__main__":
    from wesearch.lib.testing import test_main

    test_main(__file__)
