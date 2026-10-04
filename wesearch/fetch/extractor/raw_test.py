"""Tests for the raw extractor."""

from __future__ import annotations

import inspect

from wesearch.fetch.extractor.raw import extract_raw


def test_returns_the_source_untouched() -> None:
    """Markup is preserved exactly, which is the point of this extractor."""
    html = "<html><body><p>Hi</p></body></html>"
    assert extract_raw(html) == html
    assert extract_raw(html, url="https://example.com") == html


def test_url_default_is_empty_string() -> None:
    assert (
        str(inspect.signature(extract_raw))
        == "(html: 'str', *, url: 'str' = '') -> 'str'"
    )


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
