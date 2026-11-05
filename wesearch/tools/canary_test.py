"""Tests for the wesearch canary."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, cast

import argparse
import sys

import pytest

from wesearch.browser import BrowserAction, BrowserResult, BrowserTab
from wesearch.fetch.transport import zendriver
from wesearch.search.types import SearchError, SearchResult
from wesearch.tools import canary


if TYPE_CHECKING:
    from collections.abc import Sequence

    from wesearch.fetch.types import FetchSession


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


@pytest.mark.parametrize("landed", [(True, True), (True, False)])
def test_main_probes_concurrent_threads_in_one_browser_session(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    landed: tuple[bool, ...],
) -> None:
    fetched: list[str] = []
    outcomes = iter(landed)

    def fetch(url: str) -> bool:
        fetched.append(url)
        return next(outcomes)

    monkeypatch.setattr(canary, "_browser_fetch", fetch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["canary", "--backends", "--threads", "2", "--url", "https://u.example/"],
    )
    assert canary.main() == int(not all(landed))
    assert fetched == ["https://u.example/"] * 2
    out = capsys.readouterr().out
    assert f"{sum(landed)}/2 threads fetched https://u.example/" in out
    assert out.startswith(f"{'ok  ' if all(landed) else 'FAIL'} {'browser':<11} ")


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
    flags = cast(canary.Flags, parser.parse_args([]))
    assert flags.processes == 0
    assert flags.threads == 0
    assert flags.backends == ["brave", "duckduckgo", "searxng"]
    assert flags.query == "python asyncio timeout"
    assert flags.url == "https://example.com/"
    with pytest.raises(SystemExit):
        parser.parse_args(["--backends", "unknown"])


@pytest.mark.parametrize("image", [b"\x89PNG\r\n\x1a\nimage", b""])
def test_browser_action_probe_checks_form_values_and_native_screenshot(
    monkeypatch: pytest.MonkeyPatch,
    image: bytes,
) -> None:
    calls: list[tuple[BrowserAction, dict[str, object]]] = []
    sessions: list[FetchSession] = []
    user_tab = BrowserTab(tab_id="user", url="https://example.com/", title="User")
    probe_tab = BrowserTab(
        tab_id="probe",
        url="https://httpbin.org/forms/post",
        title="Form",
    )
    opened = False

    def act(
        action: BrowserAction,
        *,
        session: FetchSession,
        **kwargs: object,
    ) -> BrowserResult:
        nonlocal opened
        sessions.append(session)
        calls.append((action, kwargs))
        if action == "tabs":
            return BrowserResult(tabs=(user_tab, probe_tab) if opened else (user_tab,))
        if action == "open":
            opened = True
        if action == "close":
            opened = False
        if action == "screenshot":
            return BrowserResult(tab=probe_tab, screenshot=image)
        return BrowserResult(tab=probe_tab, text='{"custname": "wesearch-canary"}')

    monkeypatch.setattr(canary, "browser", act, raising=False)
    ok, detail, _ = canary._probe_browser_actions()
    assert ok is bool(image), detail
    assert any(
        action == "fill" and args["text"] == "wesearch-canary" for action, args in calls
    )
    assert any(action == "press" for action, _ in calls)
    assert not opened
    assert len({id(session) for session in sessions}) == 2


@pytest.mark.parametrize("ok", [False, True])
def test_main_reports_browser_action_probe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    ok: bool,
) -> None:
    monkeypatch.setattr(
        canary,
        "_probe_browser_actions",
        lambda: (ok, "form probe", 0.1),
    )
    monkeypatch.setattr(sys, "argv", ["canary", "--backends", "--browser-actions"])
    assert canary.main() == int(not ok)
    status = "ok  " if ok else "FAIL"
    assert (
        capsys.readouterr().out
        == f"{status} {'browser-actions':<11} {0.1:5.1f}s  form probe\n"
    )


@pytest.mark.parametrize("failure", ["open", "echo", "lost-user-tab", "unclosed-tab"])
def test_browser_action_probe_reports_incomplete_or_destructive_results(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    user_tab = BrowserTab(tab_id="user", url="https://example.com/", title="User")
    probe_tab = BrowserTab(
        tab_id="probe",
        url="https://httpbin.org/forms/post",
        title="Form",
    )
    opened = False
    closed = False

    def act(action: BrowserAction, **kwargs: object) -> BrowserResult:
        nonlocal opened, closed
        del kwargs
        if action == "tabs":
            if not opened:
                return BrowserResult(tabs=(user_tab,))
            return BrowserResult(
                tabs=(() if failure == "lost-user-tab" else (user_tab,))
                + ((probe_tab,) if failure == "unclosed-tab" else ()),
            )
        if action == "open":
            opened = True
            if failure == "open":
                return BrowserResult()
        if action == "close":
            closed = True
        return BrowserResult(
            tab=probe_tab,
            text="no echo" if failure == "echo" else "wesearch-canary",
            screenshot=b"\x89PNG\r\n\x1a\nimage",
        )

    monkeypatch.setattr(canary, "browser", act)
    ok, detail, _ = canary._probe_browser_actions()
    assert not ok
    assert "ValueError" in detail
    assert closed is (failure != "open")


@pytest.mark.parametrize("doc", [canary.__doc__, None])
def test_canary_help_preserves_the_module_description(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    doc: str | None,
) -> None:
    monkeypatch.setattr(canary, "__doc__", doc)
    monkeypatch.setattr(sys, "argv", ["canary", "--help"])
    with pytest.raises(SystemExit) as caught:
        canary.main()
    assert caught.value.code == 0
    shown = capsys.readouterr().out
    assert "XXXX" not in shown
    if doc is not None:
        assert "Probe every search backend and the shared browser once" in shown


def test_canary_accumulates_failures_and_forwards_probe_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def search_probe(backend: str, query: str) -> tuple[bool, str, float]:
        assert backend == "brave"
        assert query == "probe query"
        return False, "search failed", 0.1

    modes: list[bool] = []

    def browser_probe(
        workers: int,
        url: str,
        *,
        threads: bool = False,
    ) -> tuple[bool, str, float]:
        assert workers == 2
        assert url == "https://u.example/"
        modes.append(threads)
        return False, "browser failed", 0.1

    monkeypatch.setattr(canary, "_probe_backend", search_probe)
    monkeypatch.setattr(canary, "_probe_shared_browser", browser_probe)
    monkeypatch.setattr(
        canary,
        "_probe_browser_actions",
        lambda: (False, "actions failed", 0.1),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "canary",
            "--backends",
            "brave",
            "--query",
            "probe query",
            "--processes",
            "2",
            "--threads",
            "2",
            "--url",
            "https://u.example/",
            "--browser-actions",
        ],
    )
    assert canary.main() == 4
    assert modes == [False, True]
    assert len(capsys.readouterr().out.splitlines()) == 4


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
