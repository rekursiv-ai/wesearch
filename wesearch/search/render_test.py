"""Tests for search-result rendering."""

from __future__ import annotations

from datetime import datetime

from wesearch.search.custom_types import (
    ImageResult,
    MapResult,
    PaperResult,
    SearchResult,
    TorrentResult,
    VideoResult,
)
from wesearch.search.render import format_result, lean_result


def test_format_paper_result() -> None:
    out = format_result(
        PaperResult(
            url="https://doi.org/10.1/x",
            title="Attn",
            snippet="abstract",
            authors=("A", "B"),
            journal="NeurIPS",
            doi="10.1/x",
            published=datetime(2017, 6, 1),  # noqa: DTZ001 -- naive ok in test
            citations=42,
        )
    )
    assert "[Attn](https://doi.org/10.1/x)" in out
    assert "abstract" in out
    assert "doi:10.1/x" in out
    assert "cites:42" in out
    assert "2017" in out


def test_format_image_result() -> None:
    out = format_result(
        ImageResult(
            url="https://p",
            title="Cat",
            snippet="",
            image_url="https://img",
            resolution="1x1",
        )
    )
    assert out.count("https://img") >= 1
    assert "1x1" in out


def test_format_map_result() -> None:
    out = format_result(
        MapResult(
            url="https://m", title="Tower", snippet="", latitude=48.8, longitude=2.3
        )
    )
    assert "48.8,2.3" in out


def test_format_torrent_result() -> None:
    out = format_result(
        TorrentResult(
            url="https://t",
            title="ISO",
            snippet="",
            magnet_url="magnet:?xt=1",
            seed=10,
            leech=2,
        )
    )
    assert "seed:10" in out
    assert "leech:2" in out
    assert "magnet:?xt=1" in out


def test_format_plain_search_result_has_no_detail() -> None:
    out = format_result(SearchResult(url="https://w", title="W", snippet="s"))
    assert out == "[W](https://w)\ns"


def test_video_detail_precedes_media_dispatch() -> None:
    """``VideoResult`` is-a ``MediaResult``; the video branch must win.

    Testing the base first renders a video as plain media and silently loses
    its view count and channel.
    """
    out = format_result(
        VideoResult(url="https://v", title="V", snippet="", views="1M", author="C")
    )
    assert "1M views" in out
    assert "C" in out


def test_format_omits_empty_fields_rather_than_separators() -> None:
    """An unreported field must not render as a bare separator run."""
    out = format_result(
        PaperResult(url="https://p", title="P", snippet="", doi="10.1/y")
    )
    assert "·  ·" not in out
    assert out.endswith("doi:10.1/y")


def test_lean_result_keeps_category_fields() -> None:
    """The dict rendering carries the same structure the text one does.

    The MCP server projected every hit to ``url``/``title``/``snippet``, so a
    client could ask for the science tab and receive none of what makes it
    one.
    """
    lean = lean_result(
        PaperResult(
            url="https://p",
            title="P",
            snippet="s",
            doi="10.1/x",
            citations=42,
            authors=("A",),
        )
    )
    # RAW values, not the text surface's reading labels: a JSON client would
    # otherwise have to strip "doi:" to recover the identifier, and parse
    # "cites:42" to recover a number. ``paper/render.lean_record`` emits raw
    # for the same reason.
    assert lean["doi"] == "10.1/x"
    assert lean["citations"] == 42
    assert lean["authors"] == "A"


def test_text_labels_live_only_in_the_text_rendering() -> None:
    """The label belongs to presentation; the value belongs to both."""
    paper = PaperResult(url="https://p", title="P", snippet="", doi="10.1/x")
    assert "doi:10.1/x" in format_result(paper)
    assert lean_result(paper)["doi"] == "10.1/x"


def test_a_reported_zero_survives_both_renderings() -> None:
    """Zero citations and zero seeders are facts, not absences.

    Both renderings filtered on truthiness, so a reported ``0`` vanished while
    an empty string did not.
    """
    paper = PaperResult(url="https://p", title="P", snippet="", citations=0)
    assert "cites:0" in format_result(paper)
    assert lean_result(paper)["citations"] == 0

    torrent = TorrentResult(url="https://t", title="T", snippet="", seed=0, leech=0)
    text = format_result(torrent)
    assert "seed:0" in text
    assert "leech:0" in text


def test_lean_result_drops_empty_fields() -> None:
    lean = lean_result(SearchResult(url="https://w", title="W", snippet=""))
    assert lean == {"url": "https://w", "title": "W"}


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
