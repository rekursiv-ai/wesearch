"""Tests for the wesearch canary."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, cast

import argparse
import sys

import pytest

from wesearch.fetch.transport import zendriver
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


@pytest.mark.parametrize(
    ("landed", "failures"),
    [((True, True), 0), ((True, False), 1)],
)
def test_main_probes_the_shared_browser_from_every_process(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    landed: tuple[bool, ...],
    failures: int,
) -> None:
    fetched: list[str] = []
    outcomes = iter(landed)

    def fetch(url: str) -> bool:
        fetched.append(url)
        return next(outcomes)

    monkeypatch.setattr(canary, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(canary, "_browser_fetch", fetch)
    argv = ["canary", "--backends", "--processes", "2", "--url", "https://u.example/"]
    monkeypatch.setattr(sys, "argv", argv)

    assert canary.main() == failures
    assert fetched == ["https://u.example/"] * 2
    out = capsys.readouterr().out
    assert f"{2 - failures}/2 processes fetched https://u.example/" in out
    assert out.startswith("FAIL browser" if failures else "ok   browser")


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [("<html>", True), ("", False), (OSError("evicted"), False)],
)
def test_browser_fetch_reports_whether_a_page_landed(
    monkeypatch: pytest.MonkeyPatch,
    outcome: object,
    *,
    expected: bool,
) -> None:
    seen: list[str] = []

    def fake_fetch(url: str) -> object:
        seen.append(url)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(zendriver, "fetch_zendriver", fake_fetch)
    assert canary._browser_fetch("https://u.example/") is expected
    assert seen == ["https://u.example/"]


def test_processes_flag_defaults_to_skipping_the_browser() -> None:
    parser = argparse.ArgumentParser()
    canary._add_arguments(parser)
    assert cast(canary.Flags, parser.parse_args([])).processes == 0


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
