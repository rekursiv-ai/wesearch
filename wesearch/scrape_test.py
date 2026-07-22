"""Tests for wesearch.scrape."""

from __future__ import annotations

from wesearch.scrape import get_element_content


class TestGetElementContent:
    def test_found(self) -> None:
        html = '<div class="test">Hello World</div>'
        assert get_element_content(html, "div.test") == "Hello World"

    def test_whitespace_stripped(self) -> None:
        html = '<div class="test">\n  Hello World\n</div>'
        assert get_element_content(html, "div.test") == "Hello World"

    def test_not_found(self) -> None:
        html = '<div class="test">Hello</div>'
        assert get_element_content(html, "div.missing") is None

    def test_first_match_only(self) -> None:
        html = '<div class="t">First</div><div class="t">Second</div>'
        assert get_element_content(html, "div.t") == "First"

    def test_complex_selector(self) -> None:
        html = '<div id="c"><p class="m">Important</p></div>'
        assert get_element_content(html, "#c p.m") == "Important"

    def test_nested_elements(self) -> None:
        html = '<div class="o"><span>Hello</span> <b>World</b></div>'
        assert get_element_content(html, "div.o") == "HelloWorld"

    def test_empty_html(self) -> None:
        assert get_element_content("", "div") is None

    def test_empty_element(self) -> None:
        html = "<div class='t'></div>"
        assert get_element_content(html, "div.t") == ""


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
