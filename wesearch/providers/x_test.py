"""Unit tests for the X (Twitter) provider."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from wesearch.providers import x


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


class TestFetch:
    def test_delegates_to_reader_proxy(self) -> None:
        seen: dict[str, str] = {}

        def fake_proxy(
            url: str, *, transport: str, validated_hosts: Any = None
        ) -> bytes:
            del transport, validated_hosts
            seen["url"] = url
            return b"# tweet md"

        with patch.object(x, "fetch_reader_proxy", fake_proxy):
            body = x.fetch_x("https://x.com/user")
        assert body == b"# tweet md"
        assert seen["url"] == "https://x.com/user"


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
