"""Tests for wesearch.fetch.html_to_text."""

from __future__ import annotations

from pathlib import Path
from typing import cast, get_args
from unittest.mock import Mock

import pytest

from wesearch.fetch.html_to_text import html_to_text
from wesearch.types.params import Extractor


# The corpus page the default was chosen on; populate it with
# ``python -m wesearch.scripts.compare_extractors --url.
# https://www.merriam-webster.com/dictionary/agent``.
_MW_PAGE = (
    Path("/opt/scratch/caches/wesearch-extractors")
    / "www-merriam-webster-com-dictionary-agent-5db0e54b.html"
)
_PRONUNCIATION = "\u02c8\u0101-j\u0259nt"


@pytest.mark.compute_large_fixture
@pytest.mark.parametrize("extractor", get_args(Extractor))
def test_every_extractor_returns_text(extractor: Extractor) -> None:
    out = html_to_text(
        "<html><body>hi</body></html>",
        extractor=extractor,
        url="http://x/",
    )
    assert "hi" in out


@pytest.mark.parametrize("name", ["nope"])
def test_an_unknown_extractor_raises(name: str) -> None:
    with pytest.raises(AssertionError, match="'nope'"):
        html_to_text("<p>x</p>", extractor=cast(Extractor, name))


def test_raw_returns_the_page_unchanged() -> None:
    page = "<html><body><p>Hi</p></body></html>"
    assert html_to_text(page, extractor="raw", url="https://example.com") == page


def test_html2text_keeps_text_inside_a_link_only_container() -> None:
    page = (
        '<html><body><div class="prons"><a href="/audio">'
        f"{_PRONUNCIATION}</a></div></body></html>"
    )
    assert _PRONUNCIATION in html_to_text(page)


def test_html2text_renders_structure_as_markdown() -> None:
    text = html_to_text(
        "<html><body><h1>Title</h1><ul><li>one</li><li>two</li></ul></body></html>",
    )
    assert "# Title" in text
    assert "* one" in text


def test_html2text_resolves_relative_links() -> None:
    text = html_to_text(
        '<html><body><a href="/login">Log In</a></body></html>',
        url="https://example.com/page",
    )
    assert "https://example.com/login" in text


def test_html2text_does_not_rewrap_long_lines() -> None:
    sentence = "the quick brown fox jumps over the lazy dog and keeps on running " * 2
    assert sentence.strip() in html_to_text(
        f"<html><body><p>{sentence}</p></body></html>",
    )


def test_html2text_empty_page_yields_no_text() -> None:
    assert html_to_text("").strip() == ""


def test_html2text_omits_images_with_alt_text() -> None:
    page = '<p>before<img src="data:image/png;base64,abc" alt="pixel">after</p>'
    assert html_to_text(page).strip() == "beforeafter"


def test_html2text_configures_its_converter(monkeypatch: pytest.MonkeyPatch) -> None:
    converter = Mock()
    converter.handle.return_value = "converted"
    backend = Mock()
    backend.HTML2Text.return_value = converter
    monkeypatch.setattr("wesearch.fetch.html_to_text.html2text", backend)
    assert html_to_text("<p>x</p>", url="https://example.com") == "converted"
    backend.HTML2Text.assert_called_once_with(baseurl="https://example.com")
    assert converter.ignore_images is True
    assert converter.body_width == 0
    converter.handle.assert_called_once_with("<p>x</p>")


def test_markdownify_drops_stylesheet_and_script_text() -> None:
    text = html_to_text(
        "<html><head><style>.a{color:#375c71;font-size:1.3em}</style>"
        "<script>var tracking=1;</script></head>"
        "<body><p>real content</p></body></html>",
        extractor="markdownify",
    )
    assert "real content" in text
    assert "color" not in text
    assert "tracking" not in text


def test_markdownify_keeps_nested_structure() -> None:
    text = html_to_text(
        "<html><body><ul><li>outer<ul><li>inner</li></ul></li></ul>"
        "<table><tr><th>h</th></tr><tr><td>cell</td></tr></table></body></html>",
        extractor="markdownify",
    )
    assert "outer" in text
    assert "inner" in text
    assert "cell" in text


def test_markdownify_empty_page_yields_no_text() -> None:
    assert html_to_text("", extractor="markdownify").strip() == ""


@pytest.mark.compute_large_fixture
def test_trafilatura_extracts_article_prose() -> None:
    body = "This is a paragraph of article prose long enough to be scored as content."
    text = html_to_text(
        f"<html><body><article><p>{body}</p></article></body></html>",
        extractor="trafilatura",
    )
    assert body in text


@pytest.mark.compute_large_fixture
def test_trafilatura_empty_page_yields_no_text() -> None:
    assert html_to_text("", extractor="trafilatura") == ""


def test_trafilatura_passes_its_options(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = Mock()
    backend.extract.return_value = "article"
    monkeypatch.setattr("wesearch.fetch.html_to_text.trafilatura", backend)
    assert (
        html_to_text("<p>x</p>", extractor="trafilatura", url="https://example.com")
        == "article"
    )
    backend.extract.assert_called_once_with(
        "<p>x</p>",
        url="https://example.com",
        output_format="markdown",
        include_links=True,
        include_tables=True,
        with_metadata=True,
    )


def test_trafilatura_gets_none_for_no_url(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = Mock()
    backend.extract.return_value = None
    monkeypatch.setattr("wesearch.fetch.html_to_text.trafilatura", backend)
    assert html_to_text("<p>x</p>", extractor="trafilatura") == ""
    assert backend.extract.call_args.kwargs["url"] is None


@pytest.mark.cli_python_subprocess
@pytest.mark.skipif(not _MW_PAGE.exists(), reason="corpus page not cached")
def test_trafilatura_scores_away_content_html2text_keeps() -> None:
    """Why ``html2text`` is the default: trafilatura drops this pronunciation."""
    page = _MW_PAGE.read_text(errors="replace")
    assert _PRONUNCIATION in html_to_text(page)
    assert _PRONUNCIATION not in html_to_text(page, extractor="trafilatura")


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
