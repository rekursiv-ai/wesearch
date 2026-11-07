"""Tests for the Brave Search API backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import json

import pytest

from wesearch.fetch import FetchSession, RequestParams
from wesearch.search.brave import brave
from wesearch.search.types import SearchError, SearchResult


if TYPE_CHECKING:
    from contextlib import AbstractContextManager


_PAYLOAD = {
    "web": {
        "results": [
            {
                "url": "https://a.example/",
                "title": "<strong>Python</strong> &amp; you",
                "description": "Learn <strong>python</strong>.",
            },
            {"url": "javascript:alert(1)", "title": "bad", "description": ""},
            {"url": "https://b.example/", "title": "B", "description": ""},
            {"url": "http://c.example/"},
            {"title": "No URL", "description": "Drop this row."},
        ],
    },
}


@pytest.fixture(autouse=True)
def brave_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAVE_API_KEY", "test-key")


def _fetching(payload: object) -> AbstractContextManager[MagicMock | AsyncMock]:
    return patch(
        "wesearch.search.brave.fetch",
        return_value=json.dumps(payload).encode(),
    )


def test_parses_and_cleans_results() -> None:
    with _fetching(_PAYLOAD):
        results = brave("python", 10)

    assert results == [
        SearchResult(
            url="https://a.example/",
            title="Python & you",
            snippet="Learn python.",
        ),
        SearchResult(url="https://b.example/", title="B", snippet=""),
        SearchResult(url="http://c.example/", title="", snippet=""),
    ]


@pytest.mark.parametrize(("num_results", "want_count"), [(2, "2"), (50, "3")])
def test_sends_the_key_and_caps_count(
    num_results: int,
    want_count: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRAVE_API_KEY", "  test-key  ")
    with _fetching({"web": {"results": []}}) as fetch_mock:
        brave(
            "q&lang=es",
            num_results,
            headers={"X-Probe": "yes"},
            max_count=3,
            timeout_sec=2.5,
            connect_timeout_sec=1.25,
            retries=2,
            transport="stdlib",
        )

    url = urlparse(cast(str, fetch_mock.call_args.args[0]))
    request = cast(RequestParams, fetch_mock.call_args.kwargs["request"])
    assert (url.scheme, url.netloc, url.path) == (
        "https",
        "api.search.brave.com",
        "/res/v1/web/search",
    )
    assert parse_qs(url.query) == {"q": ["q&lang=es"], "count": [want_count]}
    headers = {name.lower(): value for name, value in request.content.headers.items()}
    assert (headers.pop("accept") or "").lower() == "application/json"
    assert headers == {"x-subscription-token": "test-key", "x-probe": "yes"}
    assert request.retry.retries == 2
    assert request.retry.timeout_sec == 2.5
    assert request.retry.connect_timeout_sec == 1.25
    assert request.policy.transport == "stdlib"


def test_truncates_to_num_results() -> None:
    with _fetching(_PAYLOAD):
        assert len(brave("q", 1)) == 1


def test_missing_key_is_a_search_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRAVE_API_KEY")
    with (
        patch("wesearch.search.brave.fetch") as fetch_mock,
        pytest.raises(SearchError, match="BRAVE_API_KEY"),
    ):
        brave("q")
    fetch_mock.assert_not_called()


def test_count_boundaries() -> None:
    with patch("wesearch.search.brave.fetch") as fetch_mock:
        assert brave("q", 0) == []
    fetch_mock.assert_not_called()
    with pytest.raises(ValueError, match="got -1"):
        brave("q", -1)


def test_the_session_reaches_fetch() -> None:
    session = FetchSession()
    with _fetching({"web": {"results": []}}) as fetch_mock:
        brave("q", session=session)
    assert fetch_mock.call_args.kwargs["session"] is session


def test_a_payload_without_web_results_is_empty() -> None:
    with _fetching({"query": {"original": "q"}}):
        assert brave("q") == []


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
