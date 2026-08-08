"""Tests for wesearch.types.params."""

from __future__ import annotations

from typing import get_args

import dataclasses

import pytest

from wesearch.types.params import (
    Content,
    Observe,
    Policy,
    RequestParams,
    Retry,
    Transport,
    Trust,
)


class TestGrouping:
    """``RequestParams`` composes four groups, one per concern.

    The flat 15-field shape let a transport dispatch on a request field, which
    is how ``validated_hosts`` became simultaneously a security policy, a
    transport selector, and a session-pool disabler. Grouping makes that
    category error unrepresentable: ``Content`` holds nothing a transport can
    route on, and ``Policy`` holds nothing a server ever sees.
    """

    def test_defaults_compose(self) -> None:
        params = RequestParams()
        assert params.content == Content()
        assert params.retry == Retry()
        assert params.observe == Observe()
        assert params.policy == Policy()

    @pytest.mark.parametrize(
        ("group", "attribute"),
        [
            (Content(), "method"),
            (Retry(), "retries"),
            (Observe(), "on_redirect"),
            (Policy(), "trust"),
        ],
    )
    def test_groups_are_frozen(self, group: object, attribute: str) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(group, attribute, "mutated")

    def test_policy_defaults_to_untrusted(self) -> None:
        # Safe by default: a caller who says nothing gets SSRF validation. The
        # old contract defaulted to unvalidated, so every caller had to opt in
        # and the one that did (sagent) lost the browser transport for it.
        assert Policy().trust == "untrusted"
        assert Policy().transport == "auto"

    def test_content_holds_no_transport_knob(self) -> None:
        fields = {f.name for f in dataclasses.fields(Content)}
        assert "transport" not in fields
        assert "trust" not in fields
        assert "validated_hosts" not in fields

    def test_request_params_exposes_no_flat_transport_fields(self) -> None:
        # The whole point of the grouping: policy is reachable only via
        # ``params.policy``, so a provider forwards one opaque object rather
        # than restating a security decision at each of 29 construction sites.
        fields = {f.name for f in dataclasses.fields(RequestParams)}
        assert fields == {"content", "retry", "observe", "policy"}


class TestValidation:
    """Contradictions are rejected at construction, never mid-fetch.

    The browser guards for method/body/raw-headers already lived in
    ``__post_init__``; the ``validated_hosts`` guard did not, so a browser
    request that could never succeed still constructed cleanly and failed five
    frames deep. Every browser constraint is checked in one place now.
    """

    def test_body_rejected_on_browser_transport(self) -> None:
        with pytest.raises(ValueError, match="cannot send a request body"):
            RequestParams(
                content=Content(data={"a": "1"}),
                policy=Policy(transport="zendriver"),
            )

    def test_non_get_rejected_on_browser_transport(self) -> None:
        with pytest.raises(ValueError, match="only GET"):
            RequestParams(
                content=Content(method="POST"),
                policy=Policy(transport="zendriver"),
            )

    def test_raw_headers_rejected_on_browser_transport(self) -> None:
        with pytest.raises(ValueError, match="raw_headers"):
            RequestParams(
                content=Content(raw_headers=True),
                policy=Policy(transport="curl-then-zendriver"),
            )

    def test_untrusted_browser_request_is_valid(self) -> None:
        # The regression this whole redesign exists to kill: a browser
        # transport under the DEFAULT trust must construct and run. Chrome owns
        # its DNS, so ``untrusted`` validates the host and declines to pin --
        # it does not reject the request.
        params = RequestParams(policy=Policy(transport="zendriver"))
        assert params.policy.trust == "untrusted"

    def test_negative_retries_rejected(self) -> None:
        with pytest.raises(ValueError, match="retries"):
            RequestParams(retry=Retry(retries=-1))

    def test_nonpositive_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="timeout_sec"):
            RequestParams(retry=Retry(timeout_sec=0))

    def test_data_and_json_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            RequestParams(content=Content(data={"a": "1"}, json={"b": 2}))


class TestTrustVocabulary:
    """``Trust`` names where a URL came from, not what DNS to do about it."""

    def test_two_levels(self) -> None:
        assert set(get_args(Trust)) == {"untrusted", "internal"}

    def test_transport_vocabulary_unchanged(self) -> None:
        # A ``type`` alias would break ``get_args`` at the sagent tool schemas
        # (web_fetch.py, web_search.py both enumerate this into JSON Schema).
        assert set(get_args(Transport)) == {
            "auto",
            "curl",
            "curl-then-zendriver",
            "zendriver",
            "stdlib",
        }


if __name__ == "__main__":
    from wesearch.lib.testing import test_main

    test_main(__file__)
