"""Unit tests for the reader-proxy provider."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from wesearch.fetch import PolicyParams
from wesearch.fetch.providers import reader_proxy
from wesearch.types.errors import FetchError


class TestThirdPartyConsent:
    @pytest.fixture(autouse=True)
    def _no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear the ambient key so consent reflects only the case under test.
        monkeypatch.delenv("JINA_AI_API_KEY", raising=False)

    def test_refuses_without_consent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WESEARCH_ALLOW_THIRD_PARTY_RENDER", raising=False)
        with pytest.raises(FetchError, match="third-party egress"):
            reader_proxy.fetch_reader_proxy("https://x.com/a", policy=PolicyParams())

    @pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
    def test_allows_on_truthy(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WESEARCH_ALLOW_THIRD_PARTY_RENDER", value)
        assert reader_proxy.third_party_render_allowed()

    @pytest.mark.parametrize("value", ["0", "false", "", "no"])
    def test_refuses_on_falsy(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WESEARCH_ALLOW_THIRD_PARTY_RENDER", value)
        assert not reader_proxy.third_party_render_allowed()

    def test_api_key_implies_consent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A configured key is consent on its own -- no separate flag needed.
        monkeypatch.delenv("WESEARCH_ALLOW_THIRD_PARTY_RENDER", raising=False)
        monkeypatch.setenv("JINA_AI_API_KEY", "jina_secret")
        assert reader_proxy.third_party_render_allowed()


class TestFetch:
    @pytest.fixture(autouse=True)
    def _consent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WESEARCH_ALLOW_THIRD_PARTY_RENDER", "1")

    def test_wraps_target_in_proxy_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("JINA_AI_API_KEY", raising=False)
        seen: dict[str, Any] = {}

        def fake_fetch(url: str, *, request: Any) -> tuple[bytes, None]:
            seen["url"] = url
            seen["headers"] = request.content.headers
            return b"# rendered", None

        with patch.object(reader_proxy, "fetch", fake_fetch):
            body = reader_proxy.fetch_reader_proxy(
                "https://x.com/user", policy=PolicyParams()
            )
        assert body == b"# rendered"
        assert seen["url"] == "https://r.jina.ai/https://x.com/user"
        assert seen["headers"] is None

    def test_injects_bearer_when_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("JINA_AI_API_KEY", "jina_secret")
        seen: dict[str, Any] = {}

        def fake_fetch(url: str, *, request: Any) -> tuple[bytes, None]:
            del url
            seen["headers"] = request.content.headers
            return b"ok", None

        with patch.object(reader_proxy, "fetch", fake_fetch):
            reader_proxy.fetch_reader_proxy("https://x.com/a", policy=PolicyParams())
        assert seen["headers"] == {"Authorization": "Bearer jina_secret"}

    def test_soft_fail_sentinel_raises(self) -> None:
        soft = b"Title\n\nWarning: Target URL returned error 404 while fetching"

        def fake_fetch(url: str, *, request: Any) -> tuple[bytes, None]:
            del url, request
            return soft, None

        with (
            patch.object(reader_proxy, "fetch", fake_fetch),
            pytest.raises(FetchError) as exc,
        ):
            reader_proxy.fetch_reader_proxy("https://x.com/a", policy=PolicyParams())
        assert exc.value.status == 502


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
