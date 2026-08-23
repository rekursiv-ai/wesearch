"""Tests for the markdownify extractor."""

from __future__ import annotations

from wesearch.fetch.extractor.markdownify import extract_markdownify


def test_drops_stylesheet_and_script_text() -> None:
    """A page's CSS and JS never reach the output.

    ``strip=["script", "style"]`` reads like the way to ensure this and does the
    opposite -- ``strip`` drops a tag's markup and KEEPS its text. Passing it
    turned one dictionary entry into 567_708 characters, 516_564 of them a
    single CSS rule; the bare call yields 19_650.
    """
    text = extract_markdownify(
        "<html><head><style>.a{color:#375c71;font-size:1.3em}</style>"
        "<script>var tracking=1;</script></head>"
        "<body><p>real content</p></body></html>"
    )
    assert "real content" in text
    assert "color" not in text
    assert "tracking" not in text


def test_keeps_nested_structure() -> None:
    """Nested lists and tables survive, which is this extractor's reason to exist."""
    text = extract_markdownify(
        "<html><body><ul><li>outer<ul><li>inner</li></ul></li></ul>"
        "<table><tr><th>h</th></tr><tr><td>cell</td></tr></table></body></html>"
    )
    assert "outer" in text
    assert "inner" in text
    assert "cell" in text


def test_keeps_text_an_article_scorer_would_discard() -> None:
    """Text inside a link-only container survives, as with html2text."""
    text = extract_markdownify(
        '<html><body><div class="prons"><a href="/audio">\u02c8\u0101-j\u0259nt</a>'
        "</div></body></html>"
    )
    assert "\u02c8\u0101-j\u0259nt" in text


def test_empty_document_yields_no_text() -> None:
    """An empty document yields empty text, not an exception."""
    assert extract_markdownify("").strip() == ""


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
