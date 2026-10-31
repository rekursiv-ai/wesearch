"""Tests for ``wesearch.types.errors``."""

from __future__ import annotations

from wesearch.types.errors import (
    BotDetectionError,
    CloudflareChallengeError,
    FetchError,
    GoogleJavascriptRequiredError,
    GoogleSorryError,
    PuzzleChallengeError,
)


class TestFetchErrorConstruction:
    def test_http_error_stores_context_and_formats_message(self) -> None:
        err = FetchError(
            "https://x.com/doc",
            404,
            {"content-type": "text/html"},
            b"missing",
        )
        assert err.url == "https://x.com/doc"
        assert err.status == 404
        assert err.headers == {"content-type": "text/html"}
        assert err.body == b"missing"
        assert str(err) == "HTTP 404: https://x.com/doc"

    def test_a_message_never_carries_credentials(self) -> None:
        # The ``url`` attribute is the caller's own input; the message reaches
        # logs, tool output, and reports.
        assert str(FetchError("https://u:secret@x.com/d", 404, {}, b"")) == (
            "HTTP 404: https://x.com/d"
        )
        assert "secret" not in str(FetchError("https://u:secret@x.com/", 0, {}, b""))
        blocked = PuzzleChallengeError("walled", url="https://u:secret@x.com/")
        assert "secret" not in str(blocked)
        assert "secret" not in PuzzleChallengeError.explain("https://u:secret@x.com/")

    def test_connection_error_decodes_and_strips_body(self) -> None:
        err = FetchError("https://x.com/doc", 0, {}, b"  refused\n")
        assert str(err) == "connection failed: https://x.com/doc: refused"

    def test_connection_error_replaces_invalid_utf8(self) -> None:
        err = FetchError("https://x.com/doc", 0, {}, b"bad\xff")
        assert str(err) == "connection failed: https://x.com/doc: bad�"

    def test_connection_error_without_body_uses_default_reason(self) -> None:
        assert str(FetchError("https://x.com/doc", 0, {}, b"  ")) == (
            "connection failed: https://x.com/doc: connection failed"
        )


class TestHierarchy:
    def test_leaves_subclass_root(self) -> None:
        assert issubclass(PuzzleChallengeError, BotDetectionError)
        assert issubclass(CloudflareChallengeError, BotDetectionError)
        assert issubclass(GoogleSorryError, BotDetectionError)
        assert issubclass(GoogleJavascriptRequiredError, BotDetectionError)

    def test_leaves_are_distinct(self) -> None:
        assert not issubclass(PuzzleChallengeError, CloudflareChallengeError)
        assert not issubclass(CloudflareChallengeError, GoogleSorryError)
        assert not issubclass(GoogleJavascriptRequiredError, GoogleSorryError)
        assert not issubclass(GoogleSorryError, GoogleJavascriptRequiredError)

    def test_bot_detection_is_fetch_error(self) -> None:
        assert issubclass(BotDetectionError, FetchError)


class TestBotDetectionErrorConstruction:
    def test_http_context_by_keyword(self) -> None:
        err = CloudflareChallengeError(
            url="https://x.com/doc",
            status=403,
            headers={"server": "cloudflare", "cf-ray": "a1"},
            body=b"<title>Just a moment...</title>",
        )
        assert err.url == "https://x.com/doc"
        assert err.status == 403
        assert err.headers == {"server": "cloudflare", "cf-ray": "a1"}
        assert err.body == b"<title>Just a moment...</title>"
        assert "zendriver https://x.com/doc" in str(err)
        assert "unlocks once its page loads" in str(err)

    def test_reason_only(self) -> None:
        err = PuzzleChallengeError("DuckDuckGo returned a challenge form.")
        assert str(err) == "DuckDuckGo returned a challenge form."
        assert err.status == 0
        assert err.headers == {}
        assert err.body == b""

    def test_reasonless_uses_guidance(self) -> None:
        assert str(PuzzleChallengeError()) == PuzzleChallengeError.guidance


class TestGuidance:
    def test_each_class_has_distinct_guidance(self) -> None:
        guidances = {
            PuzzleChallengeError.guidance,
            CloudflareChallengeError.guidance,
            GoogleSorryError.guidance,
            GoogleJavascriptRequiredError.guidance,
            BotDetectionError.guidance,
        }
        assert len(guidances) == 5
        assert "captcha" in PuzzleChallengeError.guidance.lower()
        assert "cloudflare" in CloudflareChallengeError.guidance.lower()
        assert "google" in GoogleSorryError.guidance.lower()
        assert "javascript" in GoogleJavascriptRequiredError.guidance.lower()

    def test_recovery_quotes_url_and_names_cooldown(self) -> None:
        assert BotDetectionError.recovery("https://x.com/a path") == (
            "Run `zendriver 'https://x.com/a path'` and solve the challenge; "
            "the site unlocks once its page loads."
        )

    def test_explain_includes_url_and_guidance(self) -> None:
        message = CloudflareChallengeError.explain("https://x.com/doc")
        assert message == (
            "Fetch blocked: https://x.com/doc -- "
            f"{CloudflareChallengeError.guidance} "
            "Run `zendriver https://x.com/doc` and solve the challenge; "
            "the site unlocks once its page loads."
        )


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
