"""Tests for the wesearch canary."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import argparse
import sys

import pytest

from wesearch.search.types import SearchError, SearchResult
from wesearch.tools import canary


if TYPE_CHECKING:
    from collections.abc import Sequence


_HIT = SearchResult(url="https://hit.example/", title="hit", snippet="")


def test_probe_reports_results_and_bypasses_the_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_search(query: str, **kwargs: object) -> Sequence[SearchResult]:
        seen.update(kwargs, query=query)
        return [_HIT]

    monkeypatch.setattr(canary, "search", fake_search)
    ok, detail, _ = canary._probe_backend("duckduckgo", "q")

    assert ok
    assert "https://hit.example/" in detail
    assert seen["backend"] == "duckduckgo"
    assert seen["cache_ttl_sec"] == 0


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [([], "no results"), (SearchError("blocked"), "SearchError: blocked")],
)
def test_probe_reports_a_failure(
    monkeypatch: pytest.MonkeyPatch,
    outcome: object,
    expected: str,
) -> None:
    def fake_search(query: str, **kwargs: object) -> object:
        del query, kwargs
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(canary, "search", fake_search)
    ok, detail, _ = canary._probe_backend("searxng", "q")

    assert not ok
    assert expected in detail


def test_main_exits_with_the_failure_count(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outcomes = {"brave": (False, "no key", 0.0), "duckduckgo": (True, "3 results", 0.1)}

    def probe(backend: str, query: str) -> tuple[bool, str, float]:
        del query
        return outcomes[backend]

    monkeypatch.setattr(canary, "_probe_backend", probe)
    monkeypatch.setattr(sys, "argv", ["canary", "--backends", "brave", "duckduckgo"])

    assert canary.main() == 1
    out = capsys.readouterr().out
    assert "FAIL brave" in out
    assert "ok   duckduckgo" in out


def test_processes_flag_defaults_to_skipping_the_browser() -> None:
    parser = argparse.ArgumentParser()
    canary._add_arguments(parser)
    assert cast(canary.Flags, parser.parse_args([])).processes == 0


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
