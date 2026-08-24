"""Unit tests for the reader-proxy fallback ladder."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from wesearch.fetch import PolicyParams
from wesearch.fetch.providers import fallback
from wesearch.types.errors import CloudflareChallengeError, FetchError


def _fetch_raising(status: int) -> Any:
    def _fake(url: str, *, request: Any) -> tuple[bytes, None]:
        del url, request
        raise FetchError("https://s.example/", status, {}, b"blocked")

    return _fake


def _fetch_raising_challenge() -> Any:
    def _fake(url: str, *, request: Any) -> tuple[bytes, None]:
        del url, request
        raise CloudflareChallengeError(
            url="https://s.example/", status=403, headers={}, body=b"blocked"
        )

    return _fake


class TestFallback:
    def test_primary_success_skips_proxy(self) -> None:
        def ok(url: str, *, request: Any) -> tuple[bytes, None]:
            del url, request
            return b"<html>ok</html>", None

        with patch.object(fallback, "fetch", ok):
            body, via = fallback.fetch_with_reader_fallback(
                "https://s.example/", policy=PolicyParams()
            )
        assert body == b"<html>ok</html>"
        assert via is False

    def test_bot_wall_falls_through_to_proxy(self) -> None:
        def proxy_ok(url: str, *, policy: Any = None) -> bytes:
            del url, policy
            return b"# md"

        with (
            patch.object(fallback, "fetch", _fetch_raising_challenge()),
            patch.object(fallback, "fetch_reader_proxy", proxy_ok),
        ):
            body, via = fallback.fetch_with_reader_fallback(
                "https://s.example/", policy=PolicyParams()
            )
        assert body == b"# md"
        assert via is True

    def test_a_rate_limit_falls_through_to_proxy(self) -> None:
        # A 429 is keyed to the EGRESS, and the proxy fetches from its own, so
        # this rung genuinely can clear it even though it is not a challenge.
        def proxy_ok(url: str, *, policy: Any = None) -> bytes:
            del url, policy
            return b"# md"

        with (
            patch.object(fallback, "fetch", _fetch_raising(429)),
            patch.object(fallback, "fetch_reader_proxy", proxy_ok),
        ):
            _body, via = fallback.fetch_with_reader_fallback(
                "https://s.example/", policy=PolicyParams()
            )
        assert via is True

    @pytest.mark.parametrize("status", [403, 503])
    def test_an_origin_error_does_not_reach_the_third_party(self, status: int) -> None:
        """A status alone is not a bot wall, so it must not egress the URL.

        ``challenge.py`` already proved these statuses insufficient: Cloudflare
        fronts ordinary origin 403s and 503s, so an expired API token looks
        exactly like a wall by status. Gating on status here sends that URL to a
        third party -- the hop this module documents as opt-in -- for an error
        no change of egress can clear.
        """
        called: list[str] = []

        def proxy(url: str, *, policy: Any = None) -> bytes:
            del policy
            called.append(url)
            return b"# md"

        with (
            patch.object(fallback, "fetch", _fetch_raising(status)),
            patch.object(fallback, "fetch_reader_proxy", proxy),
            pytest.raises(FetchError) as exc,
        ):
            fallback.fetch_with_reader_fallback(
                "https://s.example/", policy=PolicyParams()
            )
        assert exc.value.status == status
        assert called == []

    @pytest.mark.parametrize("status", [404, 500, 401])
    def test_non_bot_wall_raises_immediately(self, status: int) -> None:
        with (
            patch.object(fallback, "fetch", _fetch_raising(status)),
            pytest.raises(FetchError) as exc,
        ):
            fallback.fetch_with_reader_fallback(
                "https://s.example/", policy=PolicyParams()
            )
        assert exc.value.status == status

    def test_proxy_failure_reraises_primary(self) -> None:
        def proxy_fail(url: str, *, policy: Any = None) -> bytes:
            del url, policy
            raise FetchError("https://r.jina.ai/", 401, {}, b"auth")

        with (
            patch.object(fallback, "fetch", _fetch_raising(403)),
            patch.object(fallback, "fetch_reader_proxy", proxy_fail),
            pytest.raises(FetchError) as exc,
        ):
            fallback.fetch_with_reader_fallback(
                "https://s.example/", policy=PolicyParams()
            )
        # The primary (403) is surfaced, not the proxy's 401.
        assert exc.value.status == 403


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
