"""Tests for the Brave Search API backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import json

import pytest

from wesearch.fetch import FetchSession, RequestParams
from wesearch.search.brave import brave
from wesearch.search.custom_types import SearchError, SearchResult


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
        ],
    },
}


@pytest.fixture(autouse=True)
def brave_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAVE_API_KEY", "test-key")


def _fetching(payload: object) -> AbstractContextManager[MagicMock | AsyncMock]:
    return patch(
        "wesearch.search.brave.fetch",
        return_value=(json.dumps(payload).encode(), FetchSession()),
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
    ]


def test_sends_the_key_and_caps_count() -> None:
    with _fetching({"web": {"results": []}}) as fetch_mock:
        brave("q", 50)

    url = cast(str, fetch_mock.call_args.args[0])
    request = fetch_mock.call_args.kwargs["request"]
    assert isinstance(request, RequestParams)
    assert url.startswith("https://api.search.brave.com/res/v1/web/search?")
    assert "count=20" in url
    assert request.content.headers is not None
    assert request.content.headers["X-Subscription-Token"] == "test-key"


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


def test_a_payload_without_web_results_is_empty() -> None:
    with _fetching({"query": {"original": "q"}}):
        assert brave("q") == []


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
