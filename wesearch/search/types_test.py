"""Tests for shared search result types and helpers."""

from __future__ import annotations

from bs4 import BeautifulSoup

from wesearch.search.types import (
    ImageResult,
    clean_text,
    strip_scripts,
)


def test_clean_text_collapses_whitespace_and_prepunctuation_spaces() -> None:
    assert clean_text("  a   b  , c ; d !  ") == "a b, c; d!"


def test_strip_scripts_removes_every_script() -> None:

    soup = BeautifulSoup(
        "<p>keep</p><script>one</script><script>two</script>",
        "html.parser",
    )
    strip_scripts(soup)
    assert str(soup) == "<p>keep</p>"


def test_result_shapes_preserve_base_and_category_fields() -> None:
    result = ImageResult(url="u", title="t", snippet="s", image_url="i")
    assert result.url == "u"
    assert result.image_url == "i"


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
