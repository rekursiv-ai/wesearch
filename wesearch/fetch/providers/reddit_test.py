"""Unit tests for the Reddit provider."""

from __future__ import annotations

import pytest

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

    @pytest.mark.parametrize("url", ["https://example.com", "https://notreddit.com"])
    def test_rejects_non_reddit(self, url: str) -> None:
        assert not reddit.matches(url)


class TestRssUrl:
    def test_appends_rss(self) -> None:
        assert reddit.rss_url("https://reddit.com/r/x").endswith("/r/x/.rss")

    def test_preserves_query(self) -> None:
        got = reddit.rss_url("https://reddit.com/r/x?limit=100")
        assert got == "https://reddit.com/r/x/.rss?limit=100"

    def test_strips_trailing_json(self) -> None:
        assert reddit.rss_url("https://reddit.com/r/x/.json") == (
            "https://reddit.com/r/x/.rss"
        )

    def test_idempotent_on_rss(self) -> None:
        url = "https://reddit.com/r/x/.rss"
        assert reddit.rss_url(url) == url


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
