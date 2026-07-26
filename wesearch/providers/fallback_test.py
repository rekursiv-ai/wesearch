"""Unit tests for the reader-proxy fallback ladder."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from wesearch.errors import FetchError
from wesearch.providers import fallback


def _fetch_raising(status: int) -> Any:
    def _fake(url: str, *, request: Any) -> tuple[bytes, None]:
        del url, request
        raise FetchError("https://s.example/", status, {}, b"blocked")

    return _fake


class TestFallback:
    def test_primary_success_skips_proxy(self) -> None:
        def ok(url: str, *, request: Any) -> tuple[bytes, None]:
            del url, request
            return b"<html>ok</html>", None

        with patch.object(fallback, "fetch", ok):
            body, via = fallback.fetch_with_reader_fallback("https://s.example/")
        assert body == b"<html>ok</html>"
        assert via is False

    @pytest.mark.parametrize("status", [403, 429, 503])
    def test_bot_wall_falls_through_to_proxy(self, status: int) -> None:
        def proxy_ok(url: str, *, transport: str, validated_hosts: Any = None) -> bytes:
            del url, transport, validated_hosts
            return b"# md"

        with (
            patch.object(fallback, "fetch", _fetch_raising(status)),
            patch.object(fallback, "fetch_reader_proxy", proxy_ok),
        ):
            body, via = fallback.fetch_with_reader_fallback("https://s.example/")
        assert body == b"# md"
        assert via is True

    @pytest.mark.parametrize("status", [404, 500, 401])
    def test_non_bot_wall_raises_immediately(self, status: int) -> None:
        with (
            patch.object(fallback, "fetch", _fetch_raising(status)),
            pytest.raises(FetchError) as exc,
        ):
            fallback.fetch_with_reader_fallback("https://s.example/")
        assert exc.value.status == status

    def test_proxy_failure_reraises_primary(self) -> None:
        def proxy_fail(
            url: str, *, transport: str, validated_hosts: Any = None
        ) -> bytes:
            del url, transport, validated_hosts
            raise FetchError("https://r.jina.ai/", 401, {}, b"auth")

        with (
            patch.object(fallback, "fetch", _fetch_raising(403)),
            patch.object(fallback, "fetch_reader_proxy", proxy_fail),
            pytest.raises(FetchError) as exc,
        ):
            fallback.fetch_with_reader_fallback("https://s.example/")
        # The primary (403) is surfaced, not the proxy's 401.
        assert exc.value.status == 403


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
