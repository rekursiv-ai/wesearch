"""Tests for SearXNG category parsing and configuration."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import json

import pytest

from wesearch.fetch import FetchSession, RequestParams
from wesearch.search.custom_types import (
    CodeResult,
    FileResult,
    ImageResult,
    MapResult,
    MediaResult,
    PackageResult,
    PaperResult,
    SearchError,
    SearchResult,
    TorrentResult,
    VideoResult,
)
from wesearch.search.searxng import (
    _coordinate,
    _describe_non_json,
    _searxng_files,
    _searxng_it,
    _searxng_media,
    _searxng_paper,
    _searxng_url,
    _searxng_web,
    category_gloss,
    category_parser,
    searxng,
)


def test_category_gloss_lists_every_category() -> None:
    gloss = category_gloss()
    assert "`general`" in gloss
    assert "`social media`" in gloss
    assert "`science`" in gloss


def test_missing_url_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    with pytest.raises(SearchError, match="SEARXNG_URL"):
        _searxng_url()
    monkeypatch.setenv("SEARXNG_URL", "https://search.example///")
    assert _searxng_url() == "https://search.example"
    monkeypatch.setenv("SEARXNG_URL", "https://search.exampleX")
    assert _searxng_url() == "https://search.exampleX"


def test_coordinate_accepts_finite_numbers_only() -> None:
    assert _coordinate(1.5) == 1.5
    assert _coordinate(True) is None
    assert _coordinate("nan") is None
    assert _coordinate("inf") is None
    assert _coordinate("bad") is None


def test_media_parser_maps_and_cleans_fields() -> None:
    result = _searxng_media(
        {
            "url": "u",
            "title": " A  title ",
            "content": " A  body ",
            "publishedDate": "2020-01-02",
            "audio_src": "a",
            "iframe_src": "i",
            "length": "3:00",
            "thumbnail": "t",
        },
    )
    assert result == MediaResult(
        url="u",
        title="A title",
        snippet="A body",
        published=datetime(2020, 1, 2),  # noqa: DTZ001 -- Parser preserves naive wire timestamps.
        audio_url="a",
        iframe_url="i",
        length="3:00",
        thumbnail_url="t",
    )


def test_paper_parser_recovers_bounded_citations() -> None:
    result = _searxng_paper(
        {
            "title": " T ",
            "content": " C ",
            "authors": ["A"],
            "journal": " J ",
            "doi": "d",
            "pdf_url": "p",
            "publishedDate": "2020-01-02",
            "tags": ["tag"],
            "comments": "1,234 citations",
        },
    )
    assert result == PaperResult(
        url="",
        title="T",
        snippet="C",
        authors=("A",),
        journal="J",
        doi="d",
        pdf_url="p",
        published=datetime(2020, 1, 2),  # noqa: DTZ001 -- parser preserves wire timestamp.
        tags=("tag",),
        citations=1234,
    )
    assert _searxng_paper({"comments": "many"}).citations is None


def test_it_parser_dispatches_package_code_and_web() -> None:
    package = _searxng_it(
        {
            "template": "packages.html",
            "url": "u",
            "title": " P ",
            "content": " C ",
            "package_name": "p",
            "version": "v",
            "maintainer": "m",
            "license_name": "l",
            "homepage": "h",
            "source_code_url": "s",
            "popularity": "10",
            "tags": ["a"],
        },
    )
    code = _searxng_it(
        {
            "template": "code.html",
            "url": "u",
            "title": " C ",
            "content": " D ",
            "repository": "r",
            "filename": "f",
            "code_language": "py",
        },
    )
    web = _searxng_it({"template": "default.html", "title": "t"})
    assert package == PackageResult(
        url="u",
        title="P",
        snippet="C",
        package_name="p",
        version="v",
        maintainer="m",
        license_name="l",
        homepage="h",
        source_code_url="s",
        popularity="10",
        tags=("a",),
    )
    assert code == CodeResult(
        url="u",
        title="C",
        snippet="D",
        repository="r",
        filename="f",
        code_language="py",
    )
    assert web == SearchResult(url="", title="t", snippet="")


def test_package_tags_reject_non_strings() -> None:
    result = _searxng_it({"template": "packages.html", "tags": [1, "ok"]})
    assert isinstance(result, PackageResult)
    assert result.tags == ("ok",)


def test_files_parser_dispatches_torrent_file_and_web() -> None:
    torrent = _searxng_files(
        {
            "template": "torrent.html",
            "url": "u",
            "title": " T ",
            "content": " C ",
            "magnetlink": "m",
            "torrentfile": "tf",
            "seed": 2,
            "leech": 1,
            "filesize": "10 MB",
        },
    )
    file_result = _searxng_files(
        {
            "template": "file.html",
            "url": "u",
            "title": " T ",
            "filename": "f",
            "abstract": "a",
            "content": "c",
            "size": "2 KB",
            "mimetype": "text/plain",
            "author": "A",
        },
    )
    web = _searxng_files({"template": "default.html", "title": "t"})
    assert torrent == TorrentResult(
        url="u",
        title="T",
        snippet="C",
        magnet_url="m",
        torrent_url="tf",
        seed=2,
        leech=1,
        filesize="10 MB",
    )
    assert file_result == FileResult(
        url="u",
        title="T",
        snippet="a",
        filename="f",
        size="2 KB",
        mimetype="text/plain",
        author="A",
    )
    assert web == SearchResult(url="", title="t", snippet="")


def test_category_parser_falls_back_to_web() -> None:
    assert category_parser("general")({"title": "t"}) == SearchResult(
        url="",
        title="t",
        snippet="",
    )
    assert isinstance(category_parser("images")({}), ImageResult)
    assert isinstance(category_parser("videos")({}), VideoResult)
    assert isinstance(category_parser("map")({}), MapResult)


def test_non_json_description_distinguishes_rate_limit_html_and_plain() -> None:
    assert _describe_non_json("rate limited") == (
        "a rate-limit page instead of JSON (Cloudflare error 1015). The edge in front "
        "of the instance is throttling this egress IP -- slow the query rate or retry "
        "later; the instance itself is healthy and never saw the request."
    )
    assert _describe_non_json("  <HTML>bad") == (
        "an HTML page instead of JSON, so an intermediary answered rather than "
        "SearXNG: '<HTML>bad'"
    )
    assert _describe_non_json("plain") == "a body that is not JSON: 'plain'"


def test_searxng_public_request_decodes_and_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEARXNG_URL", "https://search.example/")
    payload = b'{"results": [{"title": " t ", "content": " c "}, {"title": "u"}]}'
    with patch(
        "wesearch.search.searxng.fetch",
        return_value=(payload, FetchSession()),
    ) as fetch_mock:
        results = searxng("q", 1, categories="general")
    assert results == [SearchResult(url="", title="t", snippet="c")]
    assert fetch_mock.call_args.args[0] == (
        "https://search.example/search?q=q&format=json&pageno=1&categories=general"
    )


def test_searxng_public_request_forwards_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEARXNG_URL", "https://search.example")
    with patch(
        "wesearch.search.searxng.fetch",
        return_value=(b'{"results": []}', FetchSession()),
    ) as fetch_mock:
        searxng(
            "q",
            headers={"X": "y"},
            categories="science",
            timeout_sec=4,
            connect_timeout_sec=2,
            retries=3,
            transport="stdlib",
        )
    request = fetch_mock.call_args.kwargs["request"]
    assert isinstance(request, RequestParams)
    assert request.content.headers == {"X": "y"}
    assert request.retry.retries == 3
    assert request.retry.timeout_sec == 4
    assert request.retry.connect_timeout_sec == 2
    assert request.policy.transport == "stdlib"
    assert request.policy.trust == "internal"


def test_searxng_defaults_and_count_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARXNG_URL", "https://search.example")
    with patch(
        "wesearch.search.searxng.fetch",
        return_value=(b'{"results": []}', FetchSession()),
    ) as fetch_mock:
        assert searxng("q") == []
    request = fetch_mock.call_args.kwargs["request"]
    assert isinstance(request, RequestParams)
    assert request.content.headers is None
    assert request.retry.retries == 1
    assert request.retry.timeout_sec == 15.0
    assert request.retry.connect_timeout_sec == 3.0
    with pytest.raises(ValueError, match="got -1"):
        searxng("q", -1)


def test_coordinate_rejects_nonfinite_and_preserves_finite_values() -> None:
    assert _coordinate(2.0) == 2.0
    assert _coordinate(float("nan")) is None
    assert _coordinate(float("inf")) is None


def test_non_json_description_recognizes_all_html_prefixes() -> None:
    expected = "an HTML page instead of JSON, so an intermediary answered rather than SearXNG: "
    assert _describe_non_json(" <!DOCTYPE html>") == expected + repr("<!DOCTYPE html>")
    assert _describe_non_json(" <?xml version='1.0'>") == expected + repr(
        "<?xml version='1.0'>",
    )
    assert _describe_non_json("error code: 1015") == (
        "a rate-limit page instead of JSON (Cloudflare error 1015). The edge in front "
        "of the instance is throttling this egress IP -- slow the query rate or retry "
        "later; the instance itself is healthy and never saw the request."
    )


def test_non_json_description_caps_each_body_at_200_characters() -> None:
    body = "plain " + "x" * 250
    description = _describe_non_json(body)
    assert description.endswith(repr(body.strip()[:200]))
    assert not description.endswith(repr(body.strip()[:201]))
    html_body = "<html>" + "x" * 250
    html_description = _describe_non_json(html_body)
    assert html_description.endswith(repr(html_body.strip()[:200]))
    assert not html_description.endswith(repr(html_body.strip()[:201]))


def test_files_parser_falls_back_to_content_and_decodes_counts() -> None:
    result = _searxng_files(
        {"template": "torrent.html", "seed": 2, "leech": 3},
    )
    assert isinstance(result, TorrentResult)
    assert result.seed == 2
    assert result.leech == 3
    invalid = _searxng_files(
        {"template": "torrent.html", "seed": "2", "leech": "3"},
    )
    assert isinstance(invalid, TorrentResult)
    assert invalid.seed is None
    assert invalid.leech is None
    file_result = _searxng_files(
        {"template": "file.html", "content": " fallback "},
    )
    assert isinstance(file_result, FileResult)
    assert file_result.snippet == "fallback"


def test_paper_parser_preserves_url_and_filters_typed_collections() -> None:
    result = _searxng_paper(
        {"url": "u", "authors": ["A", 1], "tags": ["t", 2]},
    )
    assert result.url == "u"
    assert result.authors == ("A",)
    assert result.tags == ("t",)


def test_web_parser_preserves_url() -> None:
    assert _searxng_web({"url": "u"}).url == "u"


def test_category_gloss_has_exact_order_and_separator() -> None:
    lines = category_gloss().split("\n")
    assert lines[0] == "  - `general` -- web results."
    assert lines[-1] == "  - `social media` -- posts from Mastodon/Lemmy (web results)."
    assert len(lines) == 10


def test_searxng_defaults_to_ten_results_and_auto_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEARXNG_URL", "https://search.example")
    payload = {"results": [{"title": str(index)} for index in range(11)]}
    with patch(
        "wesearch.search.searxng.fetch",
        return_value=(json.dumps(payload).encode(), FetchSession()),
    ) as fetch_mock:
        results = searxng("q")
    assert len(results) == 10
    request = fetch_mock.call_args.kwargs["request"]
    assert isinstance(request, RequestParams)
    assert request.policy.transport == "auto"


def test_searxng_zero_skips_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARXNG_URL", "https://search.example")
    with patch("wesearch.search.searxng.fetch") as fetch_mock:
        assert searxng("q", 0) == []
    fetch_mock.assert_not_called()


def test_searxng_reports_decode_and_json_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEARXNG_URL", "https://search.example")
    with (
        patch(
            "wesearch.search.searxng.fetch",
            return_value=(b"\xff", FetchSession()),
        ),
        pytest.raises(SearchError, match=r"^searxng returned undecodable bytes:"),
    ):
        searxng("q")
    with (
        patch(
            "wesearch.search.searxng.fetch",
            return_value=(b"not json", FetchSession()),
        ),
        pytest.raises(
            SearchError,
            match=r"^searxng returned a body that is not JSON:",
        ),
    ):
        searxng("q")


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
