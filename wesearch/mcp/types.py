"""Parameters of the fetch and search tools, for the MCP server and sagent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from wesearch.browser import BrowserAction, BrowserMode, BrowserResult, BrowserTab
from wesearch.mcp.schema import Field, Schema
from wesearch.search.searxng import category_gloss
from wesearch.search.types import (
    DEFAULT_SEARCH_CHAIN,
    SearchBackends,
    SearxngCategory,
)
from wesearch.types.params import Extractor, PolicyParams, Transport


if TYPE_CHECKING:
    from treekle import PlainTree


FetchMethod = Literal["GET", "POST"]


__all__ = [
    "BrowserParamsSchema",
    "FetchBodyParamsSchema",
    "FetchMethod",
    "FetchParamsSchema",
    "SearchParamsSchema",
    "browser_data",
]


class BrowserParamsSchema(Schema):
    """What persistent-browser tools accept, without session ownership."""

    action = Field[BrowserAction](
        annotation=BrowserAction,
        required=True,
        description="List tabs, open, navigate, inspect, click, fill, press, screenshot, close, or report/change Chrome mode.",
    )
    tab_id = Field[str](
        annotation=str,
        default="",
        description="Stable tab id from tabs/open; required except for tabs/open/mode.",
    )
    url = Field[str](
        annotation=str,
        default="",
        description="Public HTTP(S) destination for open/navigate.",
    )
    selector = Field[str](
        annotation=str,
        default="",
        description="CSS selector from inspection for click/fill/press.",
    )
    text = Field[str](
        annotation=str,
        default="",
        description="Replacement field contents for fill; may be empty or sensitive.",
    )
    key = Field[str](
        annotation=str,
        default="",
        description="Enter, Tab, Escape, Backspace, Delete, or an Arrow key for press.",
    )
    mode = Field[BrowserMode](
        annotation=BrowserMode,
        description="For action='mode': headed or headless; omit to report actual mode. Rejected for other actions.",
    )
    restart = Field[bool](
        annotation=bool,
        default=False,
        description="For action='mode' only: permit restarting shared Chrome to change mode. Obtain user consent first; interrupts Chrome, invalidates tab ids, restores tabs best-effort, and preserves the profile.",
    )
    max_chars = Field[int](
        annotation=int,
        default=20_000,
        description="Maximum inspected text length; must be positive.",
    )
    timeout_sec = Field[float](
        annotation=float,
        default=30,
        description="Entire action deadline in seconds, including attachment and loading.",
    )


def browser_data(result: BrowserResult) -> dict[str, PlainTree]:
    """Return browser metadata without serializing screenshot bytes.

    Args:
      result: The library's bounded inspection, tabs, or screenshot.

    Returns:
      data: JSON-compatible inspection metadata; images travel separately.

    """
    return {
        "tab": _browser_tab(result.tab) if result.tab is not None else None,
        "tabs": [_browser_tab(tab) for tab in result.tabs],
        "text": result.text,
        "elements": [
            {
                "selector": element.selector,
                "tag": element.tag,
                "role": element.role,
                "label": element.label,
                "kind": element.kind,
            }
            for element in result.elements
        ],
        "truncated": result.truncated,
        "mode": result.mode,
        "restarted": result.restarted,
    }


class FetchParamsSchema(Schema):
    """What the fetch tools accept."""

    url = Field[str](
        annotation=str,
        required=True,
        description=(
            "The URL to fetch. Fully qualified; `http://` is upgraded to HTTPS."
        ),
    )
    transport = Field[Transport](
        annotation=Transport,
        default=PolicyParams.field_default("transport", Transport),
        description=(
            "Retrieval path. 'auto' tries curl and escalates to Zendriver when a "
            "site bot-blocks it, routing straight to Zendriver for domains already "
            "learned to require it. Set it explicitly to stress a path or isolate "
            "a transport failure."
        ),
    )
    extractor = Field[Extractor](
        annotation=Extractor,
        default=PolicyParams.field_default("extractor", Extractor),
        description=(
            "How the page becomes text. 'html2text' (default) converts every text "
            "node to Markdown and loses nothing. 'trafilatura' returns only the "
            "scored article body: far smaller, and correct ONLY when the page's "
            "substance is one contiguous prose body -- an encyclopedia article, a "
            "spec document, a long-form post. On a page whose content is many "
            "small fragments it silently drops most of it (a Q&A thread keeps one "
            "answer of dozens; a profile timeline returns almost nothing; a "
            "dictionary entry loses its pronunciation), and nothing signals the "
            "loss. 'raw' returns the HTML source untouched. 'markdownify' "
            "converts the document's elements, keeping nested lists and tables a "
            "text walk flattens."
        ),
    )


class FetchBodyParamsSchema(FetchParamsSchema):
    """What the fetch tools that can POST accept."""

    method = Field[FetchMethod](
        annotation=FetchMethod,
        default="GET",
        description=(
            "HTTP method. Defaults to GET. Use POST to call JSON or form APIs."
        ),
    )
    json = Field[object](
        annotation=object,
        description=(
            "JSON-serializable body for POST requests. Sets Content-Type: "
            "application/json. Mutually exclusive with 'form'."
        ),
    )
    form = Field[dict[str, str]](
        annotation=dict,
        schema_extra={"additionalProperties": {"type": "string"}},
        description=(
            "Form fields for POST requests. Sets Content-Type: "
            "application/x-www-form-urlencoded. Mutually exclusive with 'json'."
        ),
    )


class SearchParamsSchema(Schema):
    """What the search tools accept."""

    query = Field[str](
        annotation=str,
        required=True,
        description="Search query string.",
    )
    backend = Field[SearchBackends](
        annotation=SearchBackends,
        # No default: an unnamed backend is RESOLVED by ``search`` -- a
        # non-general category selects SearXNG, anything else walks the
        # fallback chain. Naming one here would preempt that.
        description=(
            "Search backend. Omit to let a category choose, else try "
            f"{', '.join(DEFAULT_SEARCH_CHAIN)} in order, skipping any that is "
            "unconfigured, blocked, or empty."
        ),
    )
    categories = Field[SearxngCategory](
        annotation=SearxngCategory,
        default="general",
        description=(
            "SearXNG result category (tab). A non-default value selects the "
            "SearXNG backend when none is named, and is rejected alongside an "
            f"explicit non-SearXNG one.\n{category_gloss()}"
        ),
    )
    transport = Field[Transport](
        annotation=Transport,
        default=PolicyParams.field_default("transport", Transport),
        description=(
            "Retrieval path. 'auto' tries curl and escalates to Zendriver when "
            "a site bot-blocks it. Set an explicit transport to stress a path."
        ),
    )


def _browser_tab(tab: BrowserTab) -> dict[str, PlainTree]:
    return {"tab_id": tab.tab_id, "url": tab.url, "title": tab.title}
