"""Unit tests for the X (Twitter) provider."""

from __future__ import annotations

import pytest

from wesearch.fetch.providers import x


class TestMatches:
    @pytest.mark.parametrize(
        "url",
        [
            "https://x.com/user",
            "https://twitter.com/user",
            "https://mobile.x.com/user",
            "https://www.twitter.com/user",
        ],
    )
    def test_matches_x_hosts(self, url: str) -> None:
        assert x.matches(url)

    @pytest.mark.parametrize("url", ["https://example.com", "https://notx.com"])
    def test_rejects_others(self, url: str) -> None:
        assert not x.matches(url)


class TestMatchesExactDomain:
    @pytest.mark.parametrize(
        "url",
        ["x.com", "https://", "https://x.com.evil", "not-a-url"],
    )
    def test_rejects_host_without_exact_domain(self, url: str) -> None:
        assert not x.matches(url)


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
