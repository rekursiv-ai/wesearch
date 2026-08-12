"""The knobs every web-search surface exposes.

The search counterpart to :mod:`wesearch.fetch.spec`, and here for the
same reason: the sagent tool's schema, that tool's directive validation, and
the MCP server's signature are three renderings of one description rather than
three copies of it.
"""

from __future__ import annotations

from wesearch.search.custom_types import DEFAULT_SEARCH_BACKEND, SearchBackends
from wesearch.search.searxng import SearxngCategory, category_gloss
from wesearch.types.params import Policy, Transport
from wesearch.types.spec import Param, ParamSet


__all__ = ["SearchParams"]


class SearchParams(ParamSet):
    """What every search surface accepts."""

    query = Param[str](
        annotation=str, required=True, description="Search query string."
    )
    backend = Param[SearchBackends](
        annotation=SearchBackends,
        # No default: an unnamed backend is RESOLVED by ``search`` -- a
        # non-general category selects SearXNG, anything else takes the
        # build's default. Naming one here would preempt that.
        description=(
            "Search backend. Omit to let a category choose, else this build's"
            f' default ("{DEFAULT_SEARCH_BACKEND}").'
        ),
    )
    categories = Param[SearxngCategory](
        annotation=SearxngCategory,
        default="general",
        description=(
            "SearXNG result category (tab). A non-default value selects the "
            "SearXNG backend when none is named, and is rejected alongside an "
            f"explicit non-SearXNG one.\n{category_gloss()}"
        ),
    )
    transport = Param[Transport](
        annotation=Transport,
        default=Policy.field_default("transport", Transport),
        description=(
            "Retrieval path. 'auto' tries curl and escalates to Zendriver when "
            "a site bot-blocks it. Set an explicit transport to stress a path."
        ),
    )
