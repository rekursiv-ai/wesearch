"""Tests for fetch parameter schemas."""

from __future__ import annotations

from wesearch.fetch.custom_types import FetchBodyParamsSchema, FetchParamsSchema


def test_fetch_schema_declares_url_and_policy_defaults() -> None:
    assert FetchParamsSchema.url.required is True
    assert FetchParamsSchema.url.annotation is str
    assert FetchParamsSchema.transport.annotation is not None
    assert FetchParamsSchema.extractor.annotation is not None


def test_body_schema_adds_method_json_and_form_fields() -> None:
    assert FetchBodyParamsSchema.method.default == "GET"
    assert FetchBodyParamsSchema.method.required is False
    assert FetchBodyParamsSchema.json.annotation is object
    assert FetchBodyParamsSchema.form.annotation is dict
    assert FetchBodyParamsSchema.form.schema_extra == {
        "additionalProperties": {"type": "string"},
    }


if __name__ == "__main__":
    from wesearch.lib.testing.main import test_main

    test_main(__file__)
