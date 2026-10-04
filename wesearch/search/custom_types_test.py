"""Tests for shared search result types and helpers."""

from __future__ import annotations

from unittest.mock import patch

import hashlib

from bs4 import BeautifulSoup

from wesearch.search.custom_types import (
    ImageResult,
    clean_text,
    gsa_headers_for_query,
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


def test_gsa_headers_are_query_stable_and_have_suffix() -> None:
    headers = gsa_headers_for_query("same query")
    assert headers == gsa_headers_for_query("same query")
    assert set(headers) == {"User-Agent"}
    assert headers["User-Agent"].endswith(" NSTNWV")


def test_gsa_headers_use_the_first_eight_digest_bytes() -> None:
    pool = tuple(f"ua{i}" for i in range(7))
    with patch("wesearch.search.custom_types.user_agent_pool", return_value=pool):
        for query in ("a", "b", "cats", "a longer query"):
            digest_index = int.from_bytes(
                hashlib.sha256(query.encode()).digest()[:8],
            ) % len(pool)
            assert gsa_headers_for_query(query) == {
                "User-Agent": f"{pool[digest_index]} NSTNWV",
            }


def test_result_shapes_preserve_base_and_category_fields() -> None:
    result = ImageResult(url="u", title="t", snippet="s", image_url="i")
    assert result.url == "u"
    assert result.image_url == "i"


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
