"""Unit tests for the Google News provider."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from wesearch.fetch import PolicyParams, RequestParams
from wesearch.fetch.providers import google_news


class TestMatches:
    def test_matches_exact_host(self) -> None:
        assert google_news.matches("https://news.google.com/home")

    @pytest.mark.parametrize(
        "url",
        ["https://google.com", "https://mail.news.google.com", "https://example.com"],
    )
    def test_rejects_others(self, url: str) -> None:
        assert not google_news.matches(url)


class TestFetch:
    @pytest.mark.parametrize(
        ("url", "want_target", "want_payload"),
        [
            ("https://news.google.com/", "https://news.google.com/rss", "rss"),
            ("https://news.google.com/home", "https://news.google.com/rss", "rss"),
            ("https://news.google.com/rss/", "https://news.google.com/rss/", "rss"),
            ("https://news.google.com/RSS", "https://news.google.com/RSS", "html"),
            ("https://news.google.com/homeX", "https://news.google.com/homeX", "html"),
            (
                "https://news.google.com/search?q=ai",
                "https://news.google.com/rss/search?q=ai",
                "rss",
            ),
            (
                "https://news.google.com/articles/abc",
                "https://news.google.com/articles/abc",
                "html",
            ),
        ],
    )
    def test_rewrites_and_tags(
        self,
        url: str,
        want_target: str,
        want_payload: str,
    ) -> None:
        seen: dict[str, str] = {}

        def fake_fetch(target: str, *, request: object) -> tuple[bytes, None]:
            del request
            seen["target"] = target
            return b"<xml/>", None

        with patch.object(google_news, "fetch", fake_fetch):
            _body, payload = google_news.fetch_google_news(url, policy=PolicyParams())
        assert seen["target"] == want_target
        assert payload == want_payload

    @pytest.mark.parametrize(
        "url",
        [
            "https://news.google.com/rss",
            "https://news.google.com/rss/",
            "https://news.google.com/rss/topics/abc",
            "https://news.google.com/topics/abc",
        ],
    )
    def test_rewrite_boundary_paths(self, url: str) -> None:
        assert google_news._rewrite(url) is None

    def test_rewrite_preserves_query_and_fragment(self) -> None:
        assert (
            google_news._rewrite(
                "https://news.google.com/home/?hl=en#top",
            )
            == "https://news.google.com/rss?hl=en#top"
        )

    def test_fetch_forwards_policy_request(self) -> None:
        policy = PolicyParams()
        seen: list[object] = []

        def fake_fetch(url: str, *, request: object) -> tuple[bytes, None]:
            seen.extend((url, request))
            return b"body", None

        with patch.object(google_news, "fetch", fake_fetch):
            assert google_news.fetch_google_news(
                "https://news.google.com/articles/a",
                policy=policy,
            ) == (b"body", "html")
        assert isinstance(seen[1], RequestParams)
        assert seen[1].policy is policy


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
