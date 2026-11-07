"""Tests for fetch parameter schemas."""

from __future__ import annotations

from wesearch.browser import BrowserElement, BrowserResult, BrowserTab
from wesearch.mcp.types import (
    FetchBodyParamsSchema,
    FetchParamsSchema,
    browser_data,
)


def test_browser_data_carries_the_inspection_but_not_the_screenshot() -> None:
    tab = BrowserTab(tab_id="T1", url="https://example.com/", title="Example")
    other = BrowserTab(tab_id="T2", url="https://example.org/", title="Other")
    result = BrowserResult(
        tab=tab,
        tabs=(tab, other),
        text="Hello",
        elements=(
            BrowserElement(
                selector="#go",
                tag="button",
                role="button",
                label="Go",
                kind="",
            ),
        ),
        truncated=True,
        screenshot=b"\x89PNG",
        mode="headless",
        restarted=True,
    )

    assert browser_data(result) == {
        "tab": {"tab_id": "T1", "url": "https://example.com/", "title": "Example"},
        "tabs": [
            {"tab_id": "T1", "url": "https://example.com/", "title": "Example"},
            {"tab_id": "T2", "url": "https://example.org/", "title": "Other"},
        ],
        "text": "Hello",
        "elements": [
            {
                "selector": "#go",
                "tag": "button",
                "role": "button",
                "label": "Go",
                "kind": "",
            },
        ],
        "truncated": True,
        "mode": "headless",
        "restarted": True,
    }


def test_browser_data_without_a_tab_names_none() -> None:
    data = browser_data(BrowserResult())

    assert data["tab"] is None
    assert data["tabs"] == []
    assert data["mode"] is None


def test_fetch_schema_declares_url_and_policy_defaults() -> None:
    assert FetchParamsSchema.url.required is True
    assert FetchParamsSchema.url.annotation is str
    assert FetchParamsSchema.transport.default == "auto"
    assert FetchParamsSchema.transport.required is False
    assert FetchParamsSchema.extractor.default == "html2text"
    assert FetchParamsSchema.extractor.required is False


def test_body_schema_adds_method_json_and_form_fields() -> None:
    assert FetchBodyParamsSchema.method.default == "GET"
    assert FetchBodyParamsSchema.method.required is False
    assert FetchBodyParamsSchema.json.annotation is object
    assert FetchBodyParamsSchema.form.annotation is dict
    assert FetchBodyParamsSchema.form.schema_extra == {
        "additionalProperties": {"type": "string"},
    }


if __name__ == "__main__":
    from wesearch.lib.testing.runner import test_main

    test_main(__file__)
