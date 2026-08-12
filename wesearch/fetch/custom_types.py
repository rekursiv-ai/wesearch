"""The vocabulary and parameter descriptions every web-fetch surface exposes.

Declared here, in the library both adapters import, so the sagent tool's
schema, that tool's directive validation, and the MCP server's signature are
three renderings of one description rather than three copies of it.
"""

from __future__ import annotations

from typing import Literal

from wesearch.types.params import Extractor, PolicyParams, Transport
from wesearch.types.schema import Field, Schema


# The methods a fetch surface accepts. Declared here beside the param that
# constrains it, so the tool's guard and its schema enum cannot disagree.
HttpMethod = Literal["GET", "POST"]


__all__ = ["FetchBodyParamsSchema", "FetchParamsSchema", "HttpMethod"]


class FetchParamsSchema(Schema):
    """What every fetch surface accepts, including the GET-only MCP tool."""

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
    """Adds the request-body params a GET-only surface deliberately omits."""

    method = Field[HttpMethod](
        annotation=HttpMethod,
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
        # A free-form object of string fields, so the schema pins the VALUE
        # type rather than enumerating keys.
        schema_extra={"additionalProperties": {"type": "string"}},
        description=(
            "Form fields for POST requests. Sets Content-Type: "
            "application/x-www-form-urlencoded. Mutually exclusive with 'json'."
        ),
    )
