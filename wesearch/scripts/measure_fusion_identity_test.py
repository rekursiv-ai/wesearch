"""Tests for fusion identity measurements."""

from __future__ import annotations

from unittest.mock import patch

import time

import pytest

from wesearch.paper.custom_types import PaperRecord
from wesearch.paper.errors import PaperError
from wesearch.paper.providers import openalex, s2
from wesearch.paper.search import SearchResult
from wesearch.scripts import measure_fusion_identity


def _record(*, doi: str = "", arxiv_id: str = "") -> PaperRecord:
    return PaperRecord(title="Title", doi=doi, arxiv_id=arxiv_id)


def _ignore_sleep(seconds: float) -> None:
    del seconds


def _empty_mag(query: str) -> dict[str, str]:
    del query
    return {}


def test_identity_of_doi_and_arxiv_doi() -> None:
    assert measure_fusion_identity._identity_of("10.1/X") == {"doi:10.1/x"}
    assert measure_fusion_identity._identity_of("10.48550/arxiv.AbC123v2") == {
        "doi:10.48550/arxiv.abc123v2",
        "arxiv:abc123",
    }
    assert measure_fusion_identity._identity_of("") == set()


def test_doi_and_arxiv_keys() -> None:
    assert measure_fusion_identity._doi_keys(_record(doi="10.1/X")) == ["doi:10.1/x"]
    assert measure_fusion_identity._arxiv_keys(_record(arxiv_id="AbC123")) == [
        "arxiv:abc123",
    ]
    assert measure_fusion_identity._arxiv_keys(
        _record(doi="10.48550/arxiv.AbC123v2"),
    ) == [
        "doi:10.48550/arxiv.abc123v2",
        "arxiv:abc123",
    ]


def test_cross_backend_joins_use_key_intersection() -> None:
    left = [_record(doi="10.1/shared"), _record(doi="10.1/left")]
    right = [_record(doi="10.1/SHARED"), _record(doi="10.1/right")]
    assert measure_fusion_identity._cross_backend_joins(
        left,
        right,
        measure_fusion_identity._doi_keys,
    ) == {"doi:10.1/shared"}


def test_raw_mag_s2_extracts_mag_and_doi() -> None:
    body = {
        "data": [
            {"externalIds": {"MAG": "mag-1", "DOI": "10.1/a"}},
            {"externalIds": {"MAG": "mag-2"}},
            {"externalIds": {"DOI": "10.1/no-mag"}},
        ],
    }
    with patch.object(s2, "get", return_value=body):
        assert measure_fusion_identity._raw_mag_s2("query") == {
            "mag-1": "10.1/a",
            "mag-2": "",
        }


def test_raw_mag_probes_skip_wrong_typed_rows() -> None:
    s2_body = {
        "data": [
            "stray",
            {"externalIds": "oops"},
            {"externalIds": {"MAG": 5}},
            {"externalIds": {"MAG": "m", "DOI": 3}},
        ],
    }
    with patch.object(s2, "get", return_value=s2_body):
        assert measure_fusion_identity._raw_mag_s2("query") == {"m": ""}
    oa_rows: list[object] = [7, {"ids": []}, {"ids": {"mag": "9"}, "doi": 1}]
    oa_body = {"results": oa_rows}
    with patch.object(openalex, "_get", return_value=oa_body):
        assert measure_fusion_identity._raw_mag_openalex("query") == {"9": ""}


def test_raw_mag_s2_retries_paper_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def get(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PaperError("busy")
        return {"data": []}

    sleeps: list[float] = []
    monkeypatch.setattr(s2, "get", get)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    assert measure_fusion_identity._raw_mag_s2("query", attempts=2) == {}
    assert sleeps == [6.0]


def test_raw_mag_s2_exhaustion_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def busy(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise PaperError("busy")

    monkeypatch.setattr(s2, "get", busy)
    monkeypatch.setattr(time, "sleep", _ignore_sleep)
    with pytest.raises(PaperError, match="Semantic Scholar unavailable"):
        measure_fusion_identity._raw_mag_s2("query", attempts=1)


def test_raw_mag_openalex_extracts_ids_and_strips_doi_url() -> None:
    body = {
        "results": [
            {
                "ids": {"mag": "https://mag.example/123"},
                "doi": "https://doi.org/10.1/a",
            },
            {
                "ids": {"mag": "https://mag.example/789"},
                "doi": "https://doi.org/first/doi.org/second",
            },
            {"ids": {"mag": "456"}, "doi": None},
        ],
    }
    with patch.object(openalex, "_get", return_value=body):
        assert measure_fusion_identity._raw_mag_openalex("query") == {
            "123": "10.1/a",
            "789": "second",
            "456": "",
        }


def test_raw_mag_openalex_sends_exact_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def get(path: str, params: dict[str, object]) -> dict[str, object]:
        calls.append((path, params))
        return {"results": []}

    monkeypatch.setattr(openalex, "_get", get)
    assert measure_fusion_identity._raw_mag_openalex("graph neural network") == {}
    assert calls == [
        (
            "/works",
            {
                "filter": "title_and_abstract.search:graph neural network",
                "select": "id,doi,ids,title",
                "per-page": 40,
                "page": 1,
            },
        ),
    ]


def test_raw_mag_s2_sends_exact_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def get(path: str, params: dict[str, object]) -> dict[str, object]:
        calls.append((path, params))
        return {"data": []}

    monkeypatch.setattr(s2, "get", get)
    assert measure_fusion_identity._raw_mag_s2("query") == {}
    assert calls == [
        (
            "/paper/search",
            {
                "query": "query",
                "fields": s2.S2_PAPER_FIELDS_STR,
                "limit": 40,
            },
        ),
    ]


def test_raw_mag_s2_default_attempts_and_retry_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def get(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        nonlocal calls
        calls += 1
        raise PaperError("busy")

    monkeypatch.setattr(s2, "get", get)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    with pytest.raises(PaperError, match=r"^Semantic Scholar unavailable for 'query'$"):
        measure_fusion_identity._raw_mag_s2("query")
    assert calls == 4
    assert sleeps == [6.0, 6.0, 6.0, 6.0]


def test_raw_records_forwards_searches() -> None:
    s2_result = SearchResult(records=[_record(doi="10.1/s2")], total=1, complete=True)
    oa_records = [_record(doi="10.1/oa")]
    with (
        patch.object(measure_fusion_identity, "search", return_value=s2_result),
        patch.object(
            openalex,
            "search",
            return_value=(oa_records, 1, True),
        ),
    ):
        assert measure_fusion_identity._raw_records("query") == (
            s2_result.records,
            oa_records,
        )


def test_raw_records_sends_exact_backend_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s2_calls: list[tuple[str, str, int]] = []
    oa_calls: list[tuple[str, int, object, object, bool]] = []

    def s2_search(query: str, *, source: str, limit: int) -> SearchResult:
        s2_calls.append((query, source, limit))
        return SearchResult(records=[], total=0, complete=True)

    def oa_search(
        query: str,
        *,
        limit: int,
        year_from: object,
        year_to: object,
        open_access_only: bool,
    ) -> tuple[list[PaperRecord], int, bool]:
        oa_calls.append((query, limit, year_from, year_to, open_access_only))
        return [], 0, True

    monkeypatch.setattr(measure_fusion_identity, "search", s2_search)
    monkeypatch.setattr(openalex, "search", oa_search)
    assert measure_fusion_identity._raw_records("query") == ([], [])
    assert s2_calls == [("query", "s2", 40)]
    assert oa_calls == [("query", 40, None, None, False)]


def test_raw_records_default_retries_and_preserves_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def unavailable(query: str, *, source: str, limit: int) -> SearchResult:
        del query, source, limit
        nonlocal calls
        calls += 1
        raise PaperError("busy")

    monkeypatch.setattr(measure_fusion_identity, "search", unavailable)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    with pytest.raises(PaperError, match=r"^Semantic Scholar unavailable for 'query'$"):
        measure_fusion_identity._raw_records("query")
    assert calls == 4
    assert sleeps == [6.0, 6.0, 6.0, 6.0]


def test_main_continues_after_skipped_query(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = [_record(doi="10.1/ok")]
    calls: list[str] = []

    def raw_records(query: str) -> tuple[list[PaperRecord], list[PaperRecord]]:
        calls.append(query)
        if query == "first":
            raise PaperError("offline")
        return records, records

    monkeypatch.setattr(measure_fusion_identity, "_raw_records", raw_records)
    monkeypatch.setattr(measure_fusion_identity, "_raw_mag_s2", _empty_mag)
    monkeypatch.setattr(measure_fusion_identity, "_raw_mag_openalex", _empty_mag)
    monkeypatch.setattr(time, "sleep", _ignore_sleep)
    assert measure_fusion_identity.main(("first", "second")) == 0
    assert calls == ["first", "second"]
    assert "TOTAL queries=1/2" in capsys.readouterr().out


def test_main_reports_identifier_totals(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paper_records = [_record(doi="10.1/shared", arxiv_id="abc")]

    def records(query: str) -> tuple[list[PaperRecord], list[PaperRecord]]:
        del query
        return paper_records, paper_records

    def mag_s2(query: str) -> dict[str, str]:
        del query
        return {"same": "10.1/shared", "s2-only": "10.1/s2"}

    def mag_openalex(query: str) -> dict[str, str]:
        del query
        return {"same": "10.1/shared", "oa-only": "10.1/oa"}

    monkeypatch.setattr(measure_fusion_identity, "_raw_records", records)
    monkeypatch.setattr(measure_fusion_identity, "_raw_mag_s2", mag_s2)
    monkeypatch.setattr(measure_fusion_identity, "_raw_mag_openalex", mag_openalex)
    monkeypatch.setattr(time, "sleep", _ignore_sleep)
    assert measure_fusion_identity.main(("query",)) == 0
    output = capsys.readouterr().out
    assert "doi=  1" in output
    assert "doi+arxiv=  2" in output
    assert "mag_pairs=  1" in output
    assert "mag_beyond_doi= 0" in output
    assert "TOTAL queries=1/1" in output


def test_arxiv_keys_ignore_non_arxiv_doi() -> None:
    assert measure_fusion_identity._arxiv_keys(_record(doi="10.48549/not-arxiv")) == [
        "doi:10.48549/not-arxiv",
    ]


def test_main_accumulates_each_query_and_sleeps(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = [_record(doi="10.1/shared", arxiv_id="abc")]
    calls: list[tuple[str, str]] = []
    sleeps: list[float] = []

    def raw_records(query: str) -> tuple[list[PaperRecord], list[PaperRecord]]:
        calls.append(("records", query))
        return records, records

    def raw_mag_s2(query: str) -> dict[str, str]:
        calls.append(("s2", query))
        return {"mag": "10.1/s2"}

    def raw_mag_openalex(query: str) -> dict[str, str]:
        calls.append(("openalex", query))
        return {"mag": "10.1/oa"}

    monkeypatch.setattr(measure_fusion_identity, "_raw_records", raw_records)
    monkeypatch.setattr(measure_fusion_identity, "_raw_mag_s2", raw_mag_s2)
    monkeypatch.setattr(measure_fusion_identity, "_raw_mag_openalex", raw_mag_openalex)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    assert measure_fusion_identity.main(("first", "second")) == 0
    output = capsys.readouterr().out
    assert (
        "TOTAL queries=2/2 doi=2 doi+arxiv=4 (+2 from arXiv) mag_beyond_doi=2" in output
    )
    assert calls == [
        ("records", "first"),
        ("s2", "first"),
        ("openalex", "first"),
        ("records", "second"),
        ("s2", "second"),
        ("openalex", "second"),
    ]
    assert sleeps == [1.0, 1.0]


def test_main_returns_one_when_no_samples(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unavailable(query: str) -> tuple[list[PaperRecord], list[PaperRecord]]:
        raise PaperError(f"offline {query}")

    monkeypatch.setattr(measure_fusion_identity, "_raw_records", unavailable)
    assert measure_fusion_identity.main(("query",)) == 1
    assert "SKIPPED (offline query)" in capsys.readouterr().out


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
