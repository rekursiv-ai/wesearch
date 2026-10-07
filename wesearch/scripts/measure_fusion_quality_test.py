"""Tests for fused-search quality measurements."""

from __future__ import annotations

from typing import TYPE_CHECKING

import time

from wesearch.paper.custom_types import PaperRecord
from wesearch.paper.errors import PaperError
from wesearch.paper.search import SearchResult
from wesearch.scripts import measure_fusion_quality


if TYPE_CHECKING:
    import pytest


def _record(title: str, *, doi: str = "", arxiv_id: str = "") -> PaperRecord:
    return PaperRecord(title=title, doi=doi, arxiv_id=arxiv_id)


def test_residual_duplicates_use_identifier_identity() -> None:
    records = [
        _record("Same", doi="10.1/a"),
        _record("Same", doi="10.1/A"),
        _record("Same", doi="10.1/b"),
        _record("Same", doi="10.1/A"),
    ]
    assert measure_fusion_quality._residual_duplicates(records) == 2


def test_residual_duplicates_use_arxiv_identifier_identity() -> None:
    records = [
        _record("Same", arxiv_id="AbC123"),
        _record("Same", arxiv_id="abc123"),
        _record("Same", arxiv_id="def456"),
        _record("Same", arxiv_id="abc123"),
    ]
    assert measure_fusion_quality._residual_duplicates(records) == 2


def test_residual_duplicates_fall_back_to_normalized_title() -> None:
    records = [_record("A Discussion"), _record("a discussion")]
    assert measure_fusion_quality._residual_duplicates(records) == 1


def test_distinct_identifiers_do_not_duplicate_same_title() -> None:
    records = [_record("Discussion", doi="10.1/a"), _record("Discussion", doi="10.1/b")]
    assert measure_fusion_quality._residual_duplicates(records) == 0


def test_sampled_forwards_query_and_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    result = SearchResult(records=[_record("x")], total=1, complete=True)
    calls: list[tuple[str, int]] = []

    def fake_search(query: str, *, limit: int) -> SearchResult:
        calls.append((query, limit))
        return result

    monkeypatch.setattr(measure_fusion_quality, "search", fake_search)
    assert measure_fusion_quality._sampled("x", limit=40) is result
    assert calls == [("x", 40)]


def test_sampled_retries_and_accepts_incomplete_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = SearchResult(records=[_record("x")], total=1, complete=False)
    calls = 0

    def fake_search(query: str, *, limit: int) -> SearchResult:
        del query, limit
        nonlocal calls
        calls += 1
        return result

    monkeypatch.setattr(measure_fusion_quality, "search", fake_search)
    assert measure_fusion_quality._sampled("x", limit=40) is result
    assert calls == 1


def test_sampled_retries_paper_errors_and_sleeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = SearchResult(records=[_record("x")], total=1, complete=True)
    calls = 0
    sleeps: list[float] = []

    def fake_search(query: str, *, limit: int) -> SearchResult:
        del query, limit
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PaperError("offline")
        return result

    monkeypatch.setattr(measure_fusion_quality, "search", fake_search)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    assert measure_fusion_quality._sampled("x", limit=40, attempts=2) is result
    assert calls == 2
    assert sleeps == [6.0]


def test_sampled_default_attempts_are_four(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fail_search(query: str, *, limit: int) -> SearchResult:
        del query, limit
        nonlocal calls
        calls += 1
        raise PaperError("offline")

    def no_sleep(seconds: float) -> None:
        del seconds

    monkeypatch.setattr(measure_fusion_quality, "search", fail_search)
    monkeypatch.setattr(time, "sleep", no_sleep)
    assert measure_fusion_quality._sampled("x", limit=40) is None
    assert calls == 4


def test_sampled_rejects_empty_incomplete_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def empty_search(query: str, *, limit: int) -> SearchResult:
        del query, limit
        return SearchResult(records=[], total=0, complete=False)

    monkeypatch.setattr(measure_fusion_quality, "search", empty_search)
    assert measure_fusion_quality._sampled("x", limit=40, attempts=1) is None


def test_main_reports_samples_and_skips(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    samples = iter(
        (
            SearchResult(records=[_record("x")], total=1, complete=True),
            SearchResult(
                records=[
                    _record("a", doi="10.1/same"),
                    _record("b", doi="10.1/same"),
                    _record("c"),
                ],
                total=3,
                complete=True,
            ),
            None,
            SearchResult(records=[_record("d")], total=1, complete=True),
        ),
    )
    seen_queries: list[str] = []
    seen_limits: list[int] = []
    sleeps: list[float] = []

    def fake_sampled(query: str, *, limit: int) -> SearchResult | None:
        seen_queries.append(query)
        seen_limits.append(limit)
        return next(samples)

    monkeypatch.setattr(measure_fusion_quality, "_sampled", fake_sampled)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    assert measure_fusion_quality.main(("one", "two", "three", "four"), limit=1) == 0
    output = capsys.readouterr().out
    assert "records=  1" in output
    assert "records=  3" in output
    assert "'three'" in output
    assert "queries=3/4 records=5" in output
    assert "residual-duplicates=1" in output
    assert "limit-overruns=2" in output
    assert seen_queries == ["one", "two", "three", "four"]
    assert seen_limits == [1, 1, 1, 1]
    assert sleeps == [1.0, 1.0, 1.0]


def test_main_default_limit_is_forty(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_limits: list[int] = []

    def fake_sampled(query: str, *, limit: int) -> SearchResult | None:
        del query
        seen_limits.append(limit)
        return None

    monkeypatch.setattr(measure_fusion_quality, "_sampled", fake_sampled)
    assert measure_fusion_quality.main(("one",)) == 1
    assert seen_limits == [40]


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
