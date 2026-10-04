"""Tests for extractor comparison helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import argparse
import time

from lxml import html

import html2text
import pytest

from wesearch.fetch import PolicyParams, RequestParams
from wesearch.scripts import compare_extractors


if TYPE_CHECKING:
    from collections.abc import Callable


def _uppercase(value: str) -> str:
    return value.upper()


def _mark(value: str) -> str:
    return value + "!"


def _no_output(value: str) -> str:
    del value
    return "no"


def _output(value: str) -> str:
    del value
    return "output"


def _empty_output(value: str) -> str:
    del value
    return ""


def _identity(value: str) -> str:
    return value


def _reference_five(value: str) -> str:
    del value
    return "one two three four five"


def _reference_six(value: str) -> str:
    del value
    return "one two three four five six"


def _converter_five(value: str) -> str:
    del value
    return "one two three four five"


def _one_converter() -> dict[str, Callable[[str], str]]:
    return {"x": _identity}


def _two_converters() -> dict[str, Callable[[str], str]]:
    return {"a": _output, "b": _empty_output}


def test_page_slug_is_stable_and_safe() -> None:
    page = compare_extractors.Page(url="https://Example.com/a path")
    assert page.slug.startswith("example-com-a-path-")
    assert len(page.slug.rsplit("-", 1)[-1]) == 8


def test_reference_text_contains_visible_nodes() -> None:
    assert "Title" in compare_extractors.reference_text(
        "<html><body><h1>Title</h1><p>Body</p></body></html>",
    )


def test_converters_have_expected_names() -> None:
    assert list(compare_extractors.converters()) == [
        "traf",
        "traf-txt",
        "h2t",
        "mdfy",
        "read-md",
    ]


def test_score_page_filters_names_and_reports_probes() -> None:
    page = compare_extractors.Page(
        url="https://example.com",
        probes=("Body", "Missing"),
    )
    reference, scores = compare_extractors.score_page(
        page,
        "<html><body><p>Body</p></body></html>",
        names=("traf-txt",),
    )
    assert reference > 0
    assert [score.name for score in scores] == ["traf-txt"]
    assert scores[0].missing_probes == ("Missing",)
    assert scores[0].chars > 0


def test_word_grams_ignore_markdown_links() -> None:
    assert compare_extractors._word_grams("one two three four five") == frozenset(
        {("one", "two", "three", "four", "five")},
    )
    assert compare_extractors._word_grams("one [two](https://two) three") == frozenset()


def test_gram_distortion_boundaries() -> None:
    grams = compare_extractors._word_grams("one two three four five")
    assert compare_extractors._gram_distortion(frozenset(), "anything") == 1.0
    assert compare_extractors._gram_distortion(grams, "") == 1.0
    assert compare_extractors._gram_distortion(grams, "unrelated words here now") == 1.0
    assert compare_extractors._gram_distortion(grams, "one two three four five") == 0.0


def test_page_label_removes_common_suffixes() -> None:
    assert (
        compare_extractors._page_label(
            compare_extractors.Page(url="https://www.example.com/path"),
        )
        == "example"
    )
    assert (
        compare_extractors._page_label(
            compare_extractors.Page(url="https://example.org/path"),
        )
        == "example"
    )


def test_parse_args_defaults_and_overrides(tmp_path: Path) -> None:
    flags = compare_extractors._parse_args(
        ["--cache-dir", str(tmp_path), "--refresh", "--converter", "h2t"],
    )
    assert flags.cache_dir == tmp_path
    assert flags.refresh
    assert flags.converter == ["h2t"]
    assert flags.url == []


def test_main_rejects_unknown_converter(capsys: pytest.CaptureFixture[str]) -> None:
    assert compare_extractors.main(["--converter", "unknown"]) == 2
    assert "Unknown converter(s): unknown." in capsys.readouterr().out


def test_main_rejects_url_outside_corpus(capsys: pytest.CaptureFixture[str]) -> None:
    assert compare_extractors.main(["--url", "https://not-in-corpus.test"]) == 2
    assert "URL(s) not in the corpus" in capsys.readouterr().out


def test_main_scores_cached_page_and_writes_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    page = compare_extractors.CORPUS[0]
    with (
        patch.object(
            compare_extractors,
            "cached_html",
            return_value="<html><body>Body</body></html>",
        ),
        patch.object(compare_extractors, "score_page") as score,
        patch.object(compare_extractors, "_write_samples"),
        patch.object(compare_extractors, "_print_table") as table,
    ):
        score.return_value = (1, [])
        assert (
            compare_extractors.main(
                ["--cache-dir", str(tmp_path), "--url", page.url],
            )
            == 0
        )
    score.assert_called_once()
    table.assert_called_once()
    assert capsys.readouterr().out == ""


def test_main_returns_one_when_all_pages_unavailable(tmp_path: Path) -> None:
    with patch.object(
        compare_extractors,
        "cached_html",
        side_effect=OSError("offline"),
    ):
        assert compare_extractors.main(["--cache-dir", str(tmp_path)]) == 1


def test_html2text_configures_unwrapped_converter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConverter:
        ignore_images = False
        body_width = 80

        def handle(self, value: str) -> str:
            return value

    monkeypatch.setattr(html2text, "HTML2Text", FakeConverter)
    assert compare_extractors._html2text("<p>x</p>") == "<p>x</p>"


def test_readability_markdown_extracts_article(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def fromstring(value: str) -> str:
        calls.append(("fromstring", value))
        return value

    def readability(value: str) -> str:
        calls.append(("readability", value))
        return value

    def tostring(value: str, *, encoding: str) -> str:
        calls.append(("tostring", value, encoding))
        return value

    monkeypatch.setattr(html, "fromstring", fromstring)
    monkeypatch.setattr(compare_extractors, "try_readability", readability)
    monkeypatch.setattr(html, "tostring", tostring)
    monkeypatch.setattr(compare_extractors, "_html2text", _uppercase)
    assert compare_extractors._readability_markdown("<p>x</p>") == "<P>X</P>"
    assert calls == [
        ("fromstring", "<p>x</p>"),
        ("readability", "<p>x</p>"),
        ("tostring", "<p>x</p>", "unicode"),
    ]


def test_cached_html_reads_cache_without_fetch(tmp_path: Path) -> None:
    page = compare_extractors.Page(url="https://example.com")
    (tmp_path / f"{page.slug}.html").write_text("cached")
    with patch.object(compare_extractors, "fetch") as fetch:
        assert compare_extractors.cached_html(page, cache_dir=tmp_path) == "cached"
    fetch.assert_not_called()


def test_cached_html_fetches_and_writes_bytes(tmp_path: Path) -> None:
    page = compare_extractors.Page(url="https://example.com")
    with patch.object(
        compare_extractors,
        "fetch",
        return_value=(b"fresh", object()),
    ) as fetch:
        assert (
            compare_extractors.cached_html(page, cache_dir=tmp_path, refresh=True)
            == "fresh"
        )
    fetch.assert_called_once()
    assert (tmp_path / f"{page.slug}.html").read_bytes() == b"fresh"


def test_write_samples_writes_selected_converter_outputs(tmp_path: Path) -> None:
    page = compare_extractors.Page(url="https://example.com")
    with patch.object(
        compare_extractors,
        "converters",
        return_value={"first": _mark, "second": _no_output},
    ):
        compare_extractors._write_samples(
            page,
            html="body",
            out_dir=tmp_path,
            names=("first",),
        )
    assert (tmp_path / f"{page.slug}.first.txt").read_text() == "body!"
    assert not (tmp_path / f"{page.slug}.second.txt").exists()


def test_print_table_reports_exact_cells(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    page = compare_extractors.Page(url="https://example.com/path", probes=("keep",))
    score = compare_extractors.Score(
        name="x",
        chars=4,
        compression=0.25,
        distortion=0.5,
        missing_probes=("keep",),
    )
    compare_extractors._print_table([(page, [score])], names=("x",), samples=tmp_path)
    output = capsys.readouterr().out
    assert "| page | x c | x d | x p |" in output
    assert "| example | 0.25 | 0.50 | **1/1** |" in output
    assert f"Every converter's text is in `{tmp_path}`" in output


def test_score_page_runs_all_converters_and_empty_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compare_extractors, "converters", _two_converters)
    reference, scores = compare_extractors.score_page(
        compare_extractors.Page(url="https://example.com", probes=("output",)),
        "",
    )
    assert reference == 0
    assert [(score.name, score.chars, score.compression) for score in scores] == [
        ("a", 6, 0.0),
        ("b", 0, 0.0),
    ]


def test_reference_text_forwards_clean_false(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []

    def html2txt(value: str, *, clean: bool) -> str:
        calls.append((value, clean))
        return "text"

    monkeypatch.setattr(compare_extractors, "html2txt", html2txt)
    assert compare_extractors.reference_text("raw") == "text"
    assert calls == [("raw", False)]


def test_html2text_sets_exact_options(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[bool, int]] = []

    class FakeConverter:
        ignore_images = False
        body_width = 80

        def handle(self, value: str) -> str:
            seen.append((self.ignore_images, self.body_width))
            return value

    monkeypatch.setattr(html2text, "HTML2Text", FakeConverter)
    assert compare_extractors._html2text("x") == "x"
    assert seen == [(True, 0)]


def test_gram_distortion_uses_f1_not_one_sided_score() -> None:
    reference = compare_extractors._word_grams("one two three four five six")
    assert compare_extractors._gram_distortion(
        reference,
        "one two three four five",
    ) == pytest.approx(
        1 - 2 * 1 * (1 / 2) / (1 + 1 / 2),
    )


def test_page_label_handles_double_slashes_in_path() -> None:
    page = compare_extractors.Page(url="https://example.com//path/item")
    assert compare_extractors._page_label(page) == "example"


def test_parse_args_defaults_are_exact(capsys: pytest.CaptureFixture[str]) -> None:
    flags = compare_extractors._parse_args([])
    assert flags.cache_dir == Path("/opt/scratch/caches/wesearch-extractors")
    assert flags.samples == Path("/opt/scratch/artifacts/wesearch-extractors")
    with pytest.raises(SystemExit):
        compare_extractors._parse_args(["--help"])
    output = capsys.readouterr().out
    assert "XXCompare" not in output
    assert "Compare HTML-to-text converters on real pages" in output


def test_parse_args_description_is_docstring_after_shebang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str | None] = []
    real_init = argparse.ArgumentParser.__init__

    def init(
        self: argparse.ArgumentParser,
        *,
        description: str | None = None,
    ) -> None:
        seen.append(description)
        real_init(self, description=description)

    monkeypatch.setattr(argparse.ArgumentParser, "__init__", init)
    compare_extractors._parse_args([])
    doc = compare_extractors.__doc__
    assert doc is not None
    first, second, rest = doc.split("\n", 2)
    assert first == "' 2>/dev/null #"
    assert second.startswith("exec uv")
    assert seen == [rest]
    assert "\n\n" in rest


def test_print_table_includes_separator_and_empty_probe_cell(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    page = compare_extractors.Page(url="https://example.com")
    score = compare_extractors.Score(name="x", chars=1, compression=0.1, distortion=0.2)
    compare_extractors._print_table([(page, [score])], names=("x",), samples=tmp_path)
    output = capsys.readouterr().out
    assert output.splitlines()[1] == "|---|---|---|---|"
    assert "| example | 0.10 | 0.20 | 0/0 |" in output


def test_write_samples_creates_nested_directory(tmp_path: Path) -> None:
    page = compare_extractors.Page(url="https://example.com")
    out_dir = tmp_path / "nested" / "samples"
    compare_extractors._write_samples(page, html="body", out_dir=out_dir, names=())
    assert out_dir.is_dir()


def test_cached_html_forwards_url_and_request_and_replaces_invalid_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = compare_extractors.Page(url="https://example.com", transport="zendriver")
    cache_dir = tmp_path / "nested" / "cache"
    calls: list[tuple[str, object]] = []

    def fake_fetch(url: str, *, request: object) -> tuple[bytes, object]:
        calls.append((url, request))
        return b"\xff", object()

    monkeypatch.setattr(compare_extractors, "fetch", fake_fetch)
    assert compare_extractors.cached_html(page, cache_dir=cache_dir) == "\ufffd"
    assert calls[0][0] == page.url
    assert calls[0][1] == RequestParams(
        policy=PolicyParams(transport="zendriver"),
    )
    assert cache_dir.joinpath(f"{page.slug}.html").read_bytes() == b"\xff"


def test_score_page_records_exact_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compare_extractors, "reference_text", _reference_five)
    monkeypatch.setattr(compare_extractors, "converters", _one_converter)
    clocks = iter((1.0, 1.25))
    monkeypatch.setattr(time, "perf_counter", clocks.__next__)
    _, scores = compare_extractors.score_page(
        compare_extractors.Page(url="https://example.com", probes=("two", "missing")),
        "one two three four five six",
    )
    assert scores[0].name == "x"
    assert scores[0].chars == 27
    assert scores[0].compression == 1.0
    assert scores[0].seconds == 0.25
    assert scores[0].missing_probes == ("missing",)


def test_main_success_forwards_exact_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pages = (
        compare_extractors.Page(url="https://first.example", probes=("a",)),
        compare_extractors.Page(url="https://second.example", probes=("b",)),
    )
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(compare_extractors, "CORPUS", pages)

    def cached(page: compare_extractors.Page, *, cache_dir: Path, refresh: bool):
        calls.append(("cache", (page.url, cache_dir, refresh)))
        return page.url

    def score(page: compare_extractors.Page, html: str, *, names: tuple[str, ...]):
        calls.append(("score", (page.url, html, tuple(names))))
        return 1, [
            compare_extractors.Score(
                name="h2t",
                chars=1,
                compression=1.0,
                distortion=0.0,
            ),
        ]

    def write(
        page: compare_extractors.Page,
        *,
        html: str,
        out_dir: Path,
        names: tuple[str, ...],
    ):
        calls.append(("write", (page.url, html, out_dir, tuple(names))))

    def table(
        rows: list[tuple[compare_extractors.Page, list[compare_extractors.Score]]],
        *,
        names: tuple[str, ...],
        samples: Path,
    ):
        calls.append(("table", (len(rows), tuple(names), samples)))

    monkeypatch.setattr(compare_extractors, "cached_html", cached)
    monkeypatch.setattr(compare_extractors, "score_page", score)
    monkeypatch.setattr(compare_extractors, "_write_samples", write)
    monkeypatch.setattr(compare_extractors, "_print_table", table)
    assert (
        compare_extractors.main(
            [
                "--cache-dir",
                str(tmp_path),
                "--samples",
                str(tmp_path / "samples"),
                "--converter",
                "h2t",
                "--refresh",
            ],
        )
        == 0
    )
    assert calls == [
        ("cache", (pages[0].url, tmp_path, True)),
        ("score", (pages[0].url, pages[0].url, ("h2t",))),
        ("write", (pages[0].url, pages[0].url, tmp_path / "samples", ("h2t",))),
        ("cache", (pages[1].url, tmp_path, True)),
        ("score", (pages[1].url, pages[1].url, ("h2t",))),
        ("write", (pages[1].url, pages[1].url, tmp_path / "samples", ("h2t",))),
        ("table", (2, ("h2t",), tmp_path / "samples")),
    ]


def test_word_grams_strip_link_target_without_new_words() -> None:
    text = "one two three four five [six](https://six.example) seven"
    assert compare_extractors._word_grams(text) == frozenset(
        {
            ("one", "two", "three", "four", "five"),
            ("two", "three", "four", "five", "six"),
            ("three", "four", "five", "six", "seven"),
        },
    )


def test_gram_distortion_exact_partial_f1() -> None:
    reference = compare_extractors._word_grams("one two three four five six seven")
    assert compare_extractors._gram_distortion(
        reference,
        "one two three four five six seven eight",
    ) == pytest.approx(1 / 7)


def test_page_label_protocol_and_path_delimiters() -> None:
    assert (
        compare_extractors._page_label(
            compare_extractors.Page(url="https://foo.example//bar"),
        )
        == "foo.example"
    )


def test_cached_html_request_and_cached_decode_errors(tmp_path: Path) -> None:
    page = compare_extractors.Page(url="https://example.com", transport="zendriver")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    path = cache_dir / f"{page.slug}.html"
    path.write_bytes(b"\xff")
    assert compare_extractors.cached_html(page, cache_dir=cache_dir) == "�"


def test_main_reports_all_unknown_values(capsys: pytest.CaptureFixture[str]) -> None:
    assert compare_extractors.main(["--converter", "bad", "--converter", "worse"]) == 2
    output = capsys.readouterr().out
    assert "Unknown converter(s): bad, worse." in output
    assert "Available: traf, traf-txt, h2t, mdfy, read-md." in output


def test_main_reports_all_missing_urls(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        compare_extractors.main(
            ["--url", "https://bad.one", "--url", "https://bad.two"],
        )
        == 2
    )
    assert (
        "URL(s) not in the corpus: https://bad.one, https://bad.two."
        in capsys.readouterr().out
    )


def test_main_reports_unavailable_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pages = (
        compare_extractors.Page(url="https://first.example"),
        compare_extractors.Page(url="https://second.example"),
    )
    monkeypatch.setattr(compare_extractors, "CORPUS", pages)
    calls: list[str] = []

    def cached(page: compare_extractors.Page, *, cache_dir: Path, refresh: bool):
        del cache_dir, refresh
        calls.append(page.url)
        if page is pages[0]:
            raise OSError("offline")
        return "html"

    def score(
        page: compare_extractors.Page,
        html: str,
        *,
        names: tuple[str, ...],
    ) -> tuple[int, list[compare_extractors.Score]]:
        del page, html, names
        return 1, []

    def ignore(*args: object, **kwargs: object) -> None:
        del args, kwargs

    monkeypatch.setattr(compare_extractors, "cached_html", cached)
    monkeypatch.setattr(compare_extractors, "score_page", score)
    monkeypatch.setattr(compare_extractors, "_write_samples", ignore)
    monkeypatch.setattr(compare_extractors, "_print_table", ignore)
    assert compare_extractors.main(["--cache-dir", str(tmp_path)]) == 0
    assert calls == [pages[0].url, pages[1].url]
    assert "UNAVAILABLE: offline" in capsys.readouterr().out


def test_score_page_preserves_distortion_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compare_extractors, "reference_text", _reference_six)

    def one_converter() -> dict[str, Callable[[str], str]]:
        return {"x": _converter_five}

    monkeypatch.setattr(compare_extractors, "converters", one_converter)
    _, scores = compare_extractors.score_page(
        compare_extractors.Page(url="https://example.com"),
        "html",
    )
    assert scores[0].distortion == pytest.approx(1 / 3)


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
