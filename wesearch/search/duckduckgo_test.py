"""Tests for the DuckDuckGo search backend."""

from __future__ import annotations

from unittest.mock import patch

import logging

import pytest

from wesearch.fetch import FetchSession, RequestParams
from wesearch.search.custom_types import SearchError, SearchResult
from wesearch.search.duckduckgo import (
    _duckduckgo_check_captcha,
    _duckduckgo_extract_url,
    _duckduckgo_parse,
    _duckduckgo_quote_bangs,
    _duckduckgo_validate_body,
    duckduckgo,
)
from wesearch.types.errors import PuzzleChallengeError


_HTML = """
<div id="links">
  <div class="result web-result">
    <h2><a href="https://example.com/a"> First </a></h2>
    <a class="result__snippet">A   snippet.</a>
  </div>
  <div class="result web-result">
    <h2><a href="//example.com/b">Second</a></h2>
  </div>
</div>
"""


def test_quote_bangs_only_quotes_bang_tokens() -> None:
    assert _duckduckgo_quote_bangs("!w python !gh") == "'!w' python '!gh'"
    assert _duckduckgo_quote_bangs("plain query") == "plain query"


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        ("", None),
        ("//example.com/p", "https://example.com/p"),
        ("https://example.com/p", "https://example.com/p"),
        ("HTTP://example.com/p", "HTTP://example.com/p"),
        (
            "https://SUB.DUCKDUCKGO.COM/l/?uddg=https%3A%2F%2Fexample.com%2Fp",
            "https://example.com/p",
        ),
        ("ftp://example.com/p", None),
        (
            "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fp",
            "https://example.com/p",
        ),
        (
            "https://example.com/l/?uddg=https%3A%2F%2Fbad",
            "https://example.com/l/?uddg=https%3A%2F%2Fbad",
        ),
    ],
)
def test_extract_url(href: str, expected: str | None) -> None:
    assert _duckduckgo_extract_url(href) == expected


def test_parse_uses_exact_text_extraction_options() -> None:
    html = '<div id="links"><div class="web-result"><h2><a href="https://x"> <span>A</span> <span>B</span> </a></h2><a class="result__snippet"> <span>S</span> <span>T</span> </a></div></div>'

    def identity(text: str) -> str:
        return text

    with patch(
        "wesearch.search.duckduckgo.clean_text",
        side_effect=identity,
    ):
        result = _duckduckgo_parse(html, 1)[0]
    assert result.title == "A B"
    assert result.snippet == "S T"


def test_parse_skips_empty_title_and_keeps_following_results() -> None:
    html = '<div id="links"><div class="web-result"><h2><a href="https://empty"> </a></h2></div><div class="web-result"><h2><a href="https://x">valid</a></h2></div></div>'
    assert _duckduckgo_parse(html, 1) == [
        SearchResult(url="https://x", title="valid", snippet=""),
    ]


def test_parse_continues_after_invalid_results() -> None:
    html = '<div id="links"><div class="web-result"><p>missing</p></div><div class="web-result"><h2><a href="https://x">x</a></h2></div></div>'
    assert _duckduckgo_parse(html, 1) == [
        SearchResult(url="https://x", title="x", snippet=""),
    ]


def test_parse_continues_after_non_string_href() -> None:
    class Link:
        def __init__(self, href: object, title: str) -> None:
            self.href = href
            self.title = title

        def get(self, name: str) -> object:
            assert name == "href"
            return self.href

        def get_text(self, *, separator: str, strip: bool) -> str:
            del separator, strip
            return self.title

    class Container:
        def __init__(self, link: Link) -> None:
            self.link = link

        def select_one(self, selector: str) -> Link | None:
            if selector == "h2 a[href]":
                return self.link
            assert selector == "a.result__snippet"
            return None

    class Soup:
        def select(self, selector: str) -> list[Container]:
            assert selector == "div#links > div.web-result"
            return [Container(Link(object(), "bad")), Container(Link("https://x", "x"))]

        def select_one(self, selector: str) -> None:
            assert selector == "form#challenge-form"

    def make_soup(page: str, parser: str) -> Soup:
        del page, parser
        return Soup()

    def ignore_scripts(soup: Soup) -> None:
        del soup

    with (
        patch(
            "wesearch.search.duckduckgo.bs4.BeautifulSoup",
            side_effect=make_soup,
        ),
        patch(
            "wesearch.search.duckduckgo.strip_scripts",
            side_effect=ignore_scripts,
        ),
    ):
        assert _duckduckgo_parse("ignored", 1) == [
            SearchResult(url="https://x", title="x", snippet=""),
        ]


def test_parse_extracts_clean_results_and_caps() -> None:
    results = _duckduckgo_parse(_HTML, 1)
    assert results[0].url == "https://example.com/a"
    assert results[0].title == "First"
    assert results[0].snippet == "A snippet."
    assert _duckduckgo_parse(_HTML, 0) == []


def test_parse_skips_invalid_and_reports_empty_shapes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    invalid = '<div id="links"><div class="web-result"><h2><a href="/bad">x</a></h2></div></div>'
    followed_by_valid = '<div id="links"><div class="web-result"><h2><a href="/bad">bad</a></h2></div><div class="web-result"><h2><a href="https://x">valid</a></h2></div></div>'
    assert _duckduckgo_parse(followed_by_valid, 1) == [
        SearchResult(url="https://x", title="valid", snippet=""),
    ]
    with caplog.at_level(logging.WARNING):
        assert _duckduckgo_parse(invalid, 10) == []
    assert "empty result list" in caplog.text
    with caplog.at_level(logging.WARNING):
        assert _duckduckgo_parse("<p>changed</p>", 10) == []
    assert "changed markup" in caplog.text


def test_challenge_detection_and_validator() -> None:
    html = '<form id="challenge-form"></form>'
    with pytest.raises(PuzzleChallengeError) as error:
        _duckduckgo_check_captcha(html)
    assert str(error.value) == "DuckDuckGo returned a challenge form."
    with pytest.raises(PuzzleChallengeError):
        _duckduckgo_validate_body(html.encode())
    assert _duckduckgo_validate_body(b"\xff ordinary body") is None
    _duckduckgo_check_captcha("<p>ok</p>")


def test_duckduckgo_count_boundaries_and_query_limit() -> None:
    with patch("wesearch.search.duckduckgo.fetch") as fetch_mock:
        assert duckduckgo("q", num_results=0) == []
        fetch_mock.assert_not_called()
    with pytest.raises(ValueError, match="got -1"):
        duckduckgo("q", num_results=-1)
    with pytest.raises(SearchError, match="exceeds 3"):
        duckduckgo("abcd", max_query_chars=3)
    with patch(
        "wesearch.search.duckduckgo.fetch",
        return_value=(b'<div id="links"></div>', FetchSession()),
    ):
        duckduckgo("abc", max_query_chars=3)
        duckduckgo("x" * 499)
    with pytest.raises(SearchError, match="exceeds 499"):
        duckduckgo("x" * 500)


def test_duckduckgo_default_result_limit_is_ten() -> None:
    html = (
        '<div id="links">'
        + "".join(
            f'<div class="web-result"><h2><a href="https://x/{i}">x{i}</a></h2></div>'
            for i in range(11)
        )
        + "</div>"
    )
    with patch(
        "wesearch.search.duckduckgo.fetch",
        return_value=(html.encode(), FetchSession()),
    ):
        assert len(duckduckgo("q")) == 10


def test_duckduckgo_request_forwards_override_headers() -> None:
    with patch(
        "wesearch.search.duckduckgo.fetch",
        return_value=(b'<div id="links"></div>', FetchSession()),
    ) as fetch_mock:
        duckduckgo("q", headers={"User-Agent": "custom"})
    request = fetch_mock.call_args.kwargs["request"]
    assert isinstance(request, RequestParams)
    assert request.content.headers is not None
    assert request.content.headers == {
        "User-Agent": "custom",
        "Accept": "*/*",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Accept-Language": "all,all-ALL;q=0.7",
        "Referer": "https://html.duckduckgo.com/html/",
    }
    assert request.content.raw_headers is True
    assert request.policy.transport == "auto"
    assert request.retry.retries == 2
    assert request.retry.timeout_sec == 30.0
    assert request.retry.connect_timeout_sec == 3.0


def test_duckduckgo_parse_preserves_missing_snippet_and_logs_challenge(
    caplog: pytest.LogCaptureFixture,
) -> None:
    html = '<div id="links"><div class="web-result"><h2><a href="https://x"> T </a></h2></div></div>'
    assert _duckduckgo_parse(html, 10) == [
        SearchResult(url="https://x", title="T", snippet=""),
    ]
    with caplog.at_level(logging.WARNING):
        assert _duckduckgo_parse('<form id="challenge-form"></form>', 10) == []
    assert caplog.messages[-1] == "No results parsed -- DDG served a bot challenge."
    with caplog.at_level(logging.WARNING):
        assert _duckduckgo_parse('<div id="links"></div>', 10) == []
    assert (
        caplog.messages[-1] == "No results parsed -- DDG returned an empty result list."
    )
    with caplog.at_level(logging.WARNING):
        assert _duckduckgo_parse("<p>none</p>", 10) == []
    assert caplog.messages[-1] == "No results parsed -- DDG may have changed markup."


def test_duckduckgo_public_request_and_validation() -> None:
    with patch(
        "wesearch.search.duckduckgo.fetch",
        return_value=(
            b'<div id="links"><div class="web-result"><h2><a href="https://x">x</a></h2></div></div>',
            FetchSession(),
        ),
    ) as fetch_mock:
        assert duckduckgo(
            "!w x",
            retries=4,
            timeout_sec=2,
            connect_timeout_sec=1,
            transport="curl",
        )
    url = fetch_mock.call_args.args[0]
    assert str(url) == ("https://html.duckduckgo.com/html/?q=%27%21w%27+x&kl=wt-wt")
    request = fetch_mock.call_args.kwargs["request"]
    assert isinstance(request, RequestParams)
    assert request.retry.retries == 4
    assert request.retry.timeout_sec == 2
    assert request.retry.connect_timeout_sec == 1
    assert request.policy.transport == "curl"
    assert request.observe.body_validator is not None
    with pytest.raises(PuzzleChallengeError):
        _duckduckgo_validate_body(b"<form id='challenge-form'></form>")


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
