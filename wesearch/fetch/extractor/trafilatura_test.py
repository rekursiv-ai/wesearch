"""Tests for the trafilatura extractor."""

from __future__ import annotations

from pathlib import Path

import pytest

from wesearch.fetch.extractor.html2text import extract_html2text
from wesearch.fetch.extractor.trafilatura import extract_trafilatura


# The corpus page this behavior was measured on, replayed from the cache
# ``scripts/compare_extractors.py`` already writes -- the same bytes the
# benchmark scored, rather than a second copy of a third-party page checked
# into git. Populate it with:
#
#   python -m wesearch.scripts.compare_extractors \
#       --url https://www.merriam-webster.com/dictionary/agent
_CORPUS_CACHE = Path("/opt/scratch/caches/wesearch-extractors")
_MW_PAGE = _CORPUS_CACHE / "www-merriam-webster-com-dictionary-agent-5db0e54b.html"

# The pronunciation, ``ˈā-jənt`` -- probe #1 of the merriam-webster entry in
# that script's corpus.
_PRONUNCIATION = "\u02c8\u0101-j\u0259nt"


@pytest.mark.cli_python_subprocess
@pytest.mark.skipif(not _MW_PAGE.exists(), reason="corpus page not cached")
def test_scores_away_content_html2text_keeps() -> None:
    """The measured reason ``html2text`` is the default, not ``trafilatura``.

    Across that script's 11-page corpus ``html2text`` loses 0 of 37 content
    probes and ``trafilatura`` loses 12. This pins one of the 12 so a future
    default flip has to delete a failing test rather than a paragraph of
    prose.

    Needs the REAL page, not a fragment: block scoring is a relative judgement,
    so it needs surrounding boilerplate to score the pronunciation away. The
    same markup alone loses nothing -- which is why the synthetic fixture this
    test started as proved the opposite of what it claimed.
    """
    html = _MW_PAGE.read_text(errors="replace")
    assert _PRONUNCIATION in extract_html2text(html)
    assert _PRONUNCIATION not in extract_trafilatura(html)


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
