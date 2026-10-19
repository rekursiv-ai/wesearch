"""Unit tests for the Reddit provider."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from wesearch.fetch import FetchSession, PolicyParams, RequestParams
from wesearch.fetch.providers import reddit


class TestMatches:
    @pytest.mark.parametrize(
        "url",
        [
            "https://reddit.com/r/x",
            "https://old.reddit.com/r/x",
            "https://www.reddit.com/r/x/comments/abc/t/",
        ],
    )
    def test_matches_reddit_hosts(self, url: str) -> None:
        assert reddit.matches(url)

    @pytest.mark.parametrize(
        "url",
        ["https://example.com", "https://notreddit.com", "not a url"],
    )
    def test_rejects_non_reddit(self, url: str) -> None:
        assert not reddit.matches(url)


class TestRssUrl:
    def test_appends_rss(self) -> None:
        assert reddit.rss_url("https://reddit.com/r/x").endswith("/r/x/.rss")

    def test_preserves_query(self) -> None:
        got = reddit.rss_url("https://reddit.com/r/x?limit=100")
        assert got == "https://reddit.com/r/x/.rss?limit=100"

    def test_removes_all_trailing_slashes_before_rss(self) -> None:
        assert reddit.rss_url("https://reddit.com/r/x///") == (
            "https://reddit.com/r/x/.rss"
        )

    def test_strips_trailing_json(self) -> None:
        assert reddit.rss_url("https://reddit.com/r/x/.json") == (
            "https://reddit.com/r/x/.rss"
        )

    def test_idempotent_on_rss(self) -> None:
        url = "https://reddit.com/r/x/.rss"
        assert reddit.rss_url(url) == url


def _rss_fetch(
    url: str,
    *,
    request: RequestParams,
    session: FetchSession | None = None,
) -> bytes:
    assert request.policy == PolicyParams()
    assert session is not None
    return url.encode()


class TestPublicFetch:
    def test_fetches_the_rss_feed_on_the_shared_session(self) -> None:
        # The public build has no OAuth path, so every fetch reads the feed.
        with patch.object(reddit, "fetch", _rss_fetch):
            body, payload = reddit.fetch_reddit(
                "https://reddit.com/r/x",
                policy=PolicyParams(),
            )
        assert body == b"https://reddit.com/r/x/.rss"
        assert payload == "rss"


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
