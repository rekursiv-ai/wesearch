"""Tests for the html2text extractor."""

from __future__ import annotations

from wesearch.fetch.extractor.html2text import extract_html2text


def test_keeps_text_an_article_scorer_would_discard() -> None:
    """Text inside a link-only container survives.

    The shape that motivated this extractor: a dictionary entry's pronunciation
    sits inside an audio-playback anchor, so its container is 100% anchor text
    and an article extractor scores it as navigation.
    """
    text = extract_html2text(
        '<html><body><div class="prons"><a href="/audio">\u02c8\u0101-j\u0259nt</a>'
        "</div></body></html>"
    )
    assert "\u02c8\u0101-j\u0259nt" in text


def test_renders_structure_as_markdown() -> None:
    """Headings and list items keep their structure."""
    text = extract_html2text(
        "<html><body><h1>Title</h1><ul><li>one</li><li>two</li></ul></body></html>"
    )
    assert "# Title" in text
    assert "* one" in text


def test_url_resolves_relative_links() -> None:
    """A relative href becomes absolute, so a reader can follow it."""
    text = extract_html2text(
        '<html><body><a href="/login">Log In</a></body></html>',
        url="https://example.com/page",
    )
    assert "https://example.com/login" in text


def test_long_line_is_not_rewrapped() -> None:
    """A sentence past the default 78-column wrap stays on one line.

    Rewrapping splits a quoted phrase across a newline, which breaks any
    downstream search for it -- including this package's own probe tests.
    """
    sentence = "the quick brown fox jumps over the lazy dog and keeps on running " * 2
    text = extract_html2text(f"<html><body><p>{sentence}</p></body></html>")
    assert sentence.strip() in text


def test_empty_document_yields_no_text() -> None:
    """A document with no body yields empty text, not an exception."""
    assert extract_html2text("").strip() == ""


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
