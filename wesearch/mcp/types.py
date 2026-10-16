"""Parameters of the fetch and search tools, for the MCP server and sagent."""

from __future__ import annotations

from typing import Literal

from wesearch.mcp.schema import Field, Schema
from wesearch.search.searxng import category_gloss
from wesearch.search.types import (
    DEFAULT_SEARCH_CHAIN,
    SearchBackends,
    SearxngCategory,
)
from wesearch.types.params import Extractor, PolicyParams, Transport


FetchMethod = Literal["GET", "POST"]


__all__ = [
    "FetchBodyParamsSchema",
    "FetchMethod",
    "FetchParamsSchema",
    "SearchParamsSchema",
]


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
