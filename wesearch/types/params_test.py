"""Tests for request parameter values."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Literal

import pytest

from wesearch.types import params
from wesearch.types.params import (
    NO_BODY,
    ContentParams,
    PolicyParams,
    RequestParams,
    RetryParams,
)


def test_no_body_is_distinct_and_repr_is_stable() -> None:
    assert repr(NO_BODY) == "NO_BODY"
    assert ContentParams(json=None).has_body
    assert not ContentParams().has_body


def test_content_rejects_mutually_exclusive_bodies() -> None:
    with pytest.raises(
        ValueError,
        match=r"^'data' and 'json' are mutually exclusive\.$",
    ):
        ContentParams(data={"a": "1"}, json={"b": 2})


def test_content_data_and_json_each_mark_a_body() -> None:
    assert ContentParams(data={"a": "1"}).has_body
    assert ContentParams(json={"b": 2}).has_body


def test_retry_rejects_negative_retries() -> None:
    with pytest.raises(ValueError, match=r"^'retries' must be >= 0, got -1\.$"):
        RetryParams(retries=-1)


def test_retry_rejects_nonpositive_or_nonfinite_timeout() -> None:
    for timeout_sec, shown in ((0, "0"), (float("nan"), "nan"), (float("inf"), "inf")):
        with pytest.raises(
            ValueError,
            match=rf"^'timeout_sec' must be a finite number > 0, got {shown}\.$",
        ):
            RetryParams(timeout_sec=timeout_sec)


def test_retry_rejects_nonpositive_or_nonfinite_connect_timeout() -> None:
    for timeout_sec, shown in ((0, "0"), (float("nan"), "nan"), (float("inf"), "inf")):
        with pytest.raises(
            ValueError,
            match=rf"^'connect_timeout_sec' must be a finite number > 0, got {shown}\.$",
        ):
            RetryParams(connect_timeout_sec=timeout_sec)


def test_retry_rejects_negative_redirects() -> None:
    with pytest.raises(ValueError, match=r"^'max_redirects' must be >= 0, got -1\.$"):
        RetryParams(max_redirects=-1)


def test_retry_accepts_exact_boundaries() -> None:
    assert RetryParams(timeout_sec=0.1, max_redirects=0).max_redirects == 0
    assert RetryParams(connect_timeout_sec=0.1).connect_timeout_sec == 0.1


@pytest.mark.parametrize(
    ("header", "expected"),
    [("-1", 0.0), ("0", 0.0), ("30", 30.0), ("31", 30.0), ("nan", None)],
)
def test_retry_after_delta_is_clamped(header: str, expected: float | None) -> None:
    retry = RetryParams()
    if expected is None:
        assert retry.backoff_delay(0, {"retry-after": header}) >= 1.0
    else:
        assert retry.backoff_delay(0, {"retry-after": header}) == expected


def test_retry_after_http_date_is_clamped() -> None:
    future = (datetime.now(UTC) + timedelta(seconds=60)).strftime(
        "%a, %d %b %Y %H:%M:%S GMT",
    )
    assert RetryParams().backoff_delay(0, {"retry-after": future}) <= 30


def test_retry_after_http_date_uses_current_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2025, 1, 1, tzinfo=UTC)

    class Clock:
        @classmethod
        def now(cls, tz: object) -> datetime:
            assert tz is UTC
            return now

    monkeypatch.setattr(params, "datetime", Clock)
    retry = RetryParams()
    assert retry.backoff_delay(0, {"retry-after": format_datetime(now)}) == 0.0
    assert (
        retry.backoff_delay(
            0,
            {"retry-after": format_datetime(now + timedelta(seconds=31))},
        )
        == 30.0
    )


def test_retry_after_malformed_value_uses_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def jitter(low: float, high: float) -> float:
        return low + high

    monkeypatch.setattr("wesearch.types.params.random.uniform", jitter)
    assert RetryParams().backoff_delay(4, {"retry-after": "not-a-date"}) == 24.0
    assert RetryParams().backoff_delay(6, {}) == 45.0


def test_policy_field_default_and_missing_name() -> None:
    assert PolicyParams.field_default("transport") == "auto"
    assert PolicyParams.field_default("extractor") == "html2text"
    assert PolicyParams.field_default("trust") == "untrusted"
    with pytest.raises(KeyError) as exc:
        PolicyParams.field_default("missing")
    assert exc.value.args == ("PolicyParams has no field 'missing'.",)


def test_request_defaults_are_independent() -> None:
    first = RequestParams()
    second = RequestParams()
    assert first == second
    assert first.content is not second.content
    assert first.retry is not second.retry


@pytest.mark.parametrize(
    ("transport", "content", "message"),
    [
        (
            "zendriver",
            ContentParams(method="POST"),
            "The zendriver backend supports only GET requests.",
        ),
        (
            "curl-then-zendriver",
            ContentParams(json=None),
            "The curl-then-zendriver backend cannot send a request body.",
        ),
        (
            "zendriver",
            ContentParams(raw_headers=True),
            "The zendriver transport cannot honor 'raw_headers'.",
        ),
    ],
)
def test_request_rejects_browser_unsupported_content(
    transport: Literal["zendriver", "curl-then-zendriver"],
    content: ContentParams,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=f"^{message}$"):
        RequestParams(content=content, policy=PolicyParams(transport=transport))


def test_request_allows_non_browser_transport_content() -> None:
    request = RequestParams(
        content=ContentParams(method="POST", json={"ok": True}, raw_headers=True),
        policy=PolicyParams(transport="curl"),
    )
    assert request.content.method == "POST"


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
