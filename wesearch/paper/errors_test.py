"""Tests for the shared FetchError -> PaperError translator."""

from __future__ import annotations

from wesearch.paper.errors import (
    BackendError,
    NotFoundError,
    RateLimitError,
    translate_http_error,
)
from wesearch.types.errors import FetchError


def _err(status: int, body: bytes = b"x") -> FetchError:
    return FetchError("https://api.example/x", status, {}, body)


class TestTranslateHttpError:
    def test_429_is_rate_limit_with_backend_message(self) -> None:
        out = translate_http_error(
            _err(429),
            backend="Semantic Scholar",
            rate_limit_message="set the key",
        )
        assert isinstance(out, RateLimitError)
        assert str(out) == "set the key"

    def test_429_uses_generic_message_when_empty(self) -> None:
        out = translate_http_error(_err(429), backend="OpenAlex")
        assert str(out) == "OpenAlex rate limit hit; retry shortly."

    def test_404_is_not_found_when_policy_on(self) -> None:
        out = translate_http_error(_err(404), backend="Semantic Scholar")
        assert isinstance(out, NotFoundError)
        assert str(out) == "Semantic Scholar: not found."

    def test_404_is_backend_error_when_policy_off(self) -> None:
        # OpenAlex signals real not-found semantically (200 + empty results); an
        # HTTP 404 there is a bad endpoint, not a missing entity -> BackendError.
        out = translate_http_error(
            _err(404),
            backend="OpenAlex",
            treat_404_as_missing=False,
        )
        assert isinstance(out, BackendError)
        assert out.status == 404

    def test_other_status_is_backend_error_with_status_and_body(self) -> None:
        out = translate_http_error(_err(500, b"boom"), backend="OpenAlex")
        assert isinstance(out, BackendError)
        assert out.status == 500
        assert str(out) == "OpenAlex HTTP 500: boom"

    def test_backend_error_body_is_capped_and_decoded(self) -> None:
        body = b"a" * 200 + b"b" + b"\xff"
        out = translate_http_error(_err(500, body), backend="OpenAlex")
        assert str(out) == "OpenAlex HTTP 500: " + "a" * 200

    def test_backend_error_decodes_invalid_bytes_with_replacement(self) -> None:
        out = translate_http_error(_err(500, b"bad\xff"), backend="OpenAlex")
        assert str(out) == "OpenAlex HTTP 500: bad\ufffd"

    def test_backend_error_default_status_is_zero(self) -> None:
        assert BackendError("x").status == 0

    def test_default_policy_maps_404_to_not_found(self) -> None:
        assert isinstance(translate_http_error(_err(404), backend="B"), NotFoundError)


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
