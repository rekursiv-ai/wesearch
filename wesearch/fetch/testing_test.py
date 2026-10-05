"""Tests for fetch transport test doubles."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import Mock, patch

from curl_cffi import requests as cc_requests

from wesearch.fetch import testing
from wesearch.fetch.common import decompress
from wesearch.fetch.testing import (
    StubCookies,
    StubSession,
    lower_headers,
    zstd_compress,
)


if TYPE_CHECKING:
    import pytest


def test_lower_headers_normalizes_keys_and_values() -> None:
    assert lower_headers({"headers": {"X-Test": "yes", "ACCEPT": "*/*"}}) == {
        "x-test": "yes",
        "accept": "*/*",
    }


def test_lower_headers_uses_a_typed_read() -> None:
    headers = object()
    with patch.object(testing, "convert", return_value={"X": "Y"}) as convert:
        assert lower_headers({"headers": headers}) == {"x": "Y"}
    convert.assert_called_once_with(headers, dict[str, str])


def test_stub_cookies_replace_same_name_and_keep_other_names() -> None:
    cookies = StubCookies()
    cookies.set("a", "1")
    cookies.set("b", "2")
    cookies.set("a", "3")
    assert [(cookie.name, cookie.value) for cookie in cookies.jar] == [
        ("b", "2"),
        ("a", "3"),
    ]


def test_zstd_compress_round_trips_both_frame_modes() -> None:
    data = b"payload" * 100
    regular = zstd_compress(data)
    streaming = zstd_compress(data, streaming=True)
    assert decompress(regular, "zstd") == data
    assert decompress(streaming, "zstd") == data
    assert regular != streaming


def test_zstd_compress_uses_wheel_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(testing, "zstd", None)
    data = b"wheel payload"
    assert decompress(zstd_compress(data), "zstd") == data
    assert decompress(zstd_compress(data, streaming=True), "zstd") == data


def test_zstd_compress_selects_stdlib_frame_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compressor = Mock()
    compressor.compress.return_value = b"regular"
    compressor.flush.return_value = b"flush"
    stdlib = Mock()
    stdlib.compress.return_value = b"one-shot"
    stdlib.ZstdCompressor.return_value = compressor
    monkeypatch.setattr(testing, "zstd", stdlib)

    assert zstd_compress(b"data") == b"one-shot"
    assert zstd_compress(b"data", streaming=True) == b"regularflush"
    stdlib.compress.assert_called_once_with(b"data")
    compressor.compress.assert_called_once_with(b"data")
    compressor.flush.assert_called_once_with()


def test_zstd_compress_selects_wheel_frame_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compressor = Mock()
    compressor.compress.return_value = b"regular"
    compressor.flush.return_value = b"flush"
    writer = Mock()
    writer.__enter__ = Mock(return_value=writer)
    writer.__exit__ = Mock(return_value=None)
    compressor.stream_writer.return_value = writer
    wheel = Mock()
    wheel.ZstdCompressor.return_value = compressor
    monkeypatch.setattr(testing, "zstd", None)
    monkeypatch.setattr(testing, "zstandard", wheel)

    assert zstd_compress(b"data") == b"regular"
    zstd_compress(b"data", streaming=True)
    compressor.stream_writer.assert_called_once()
    assert compressor.stream_writer.call_args.kwargs == {"closefd": False}


def test_stub_cookies_preserves_cookie_attributes() -> None:
    cookies = StubCookies()
    cookies.set("default", "value")
    cookies.set("custom", "value", domain="example.com", path="/x", secure=True)
    assert (cookies.jar[0].domain, cookies.jar[0].path, cookies.jar[0].secure) == (
        "",
        "/",
        False,
    )
    assert (cookies.jar[1].domain, cookies.jar[1].path, cookies.jar[1].secure) == (
        "example.com",
        "/x",
        True,
    )


def test_stub_cookies_keeps_entries_without_names() -> None:
    cookies = StubCookies()
    unnamed = cast(testing.StubCookie, object())
    cookies.jar.append(unnamed)
    cookies.set("name", "value")
    assert cookies.jar[0] is unnamed
    assert [(cookie.name, cookie.value) for cookie in cookies.jar[1:]] == [
        ("name", "value"),
    ]


def test_stub_session_delegates_to_module_request() -> None:
    session = StubSession()
    assert session.cookies.jar == []
    response = Mock(spec=cc_requests.Response)
    with patch.object(cc_requests, "request", return_value=response) as request:
        assert (
            StubSession().request(
                "GET",
                "https://example.com",
                headers={"X-Test": "yes"},
            )
            is response
        )
    request.assert_called_once_with(
        "GET",
        "https://example.com",
        headers={"X-Test": "yes"},
    )


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
