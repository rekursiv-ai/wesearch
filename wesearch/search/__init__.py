"""Web search across pluggable backends.

The search analogue of :mod:`wesearch.fetch`: one synchronous
backend-agnostic entry point, :func:`wesearch.search.search.search`, over
SearXNG, DuckDuckGo, and Google.

Import each name from the submodule that defines it (this package's ``__init__``
re-exports nothing), exactly as :mod:`wesearch.fetch.providers` and
:mod:`wesearch.paper.providers` are used.
"""

from __future__ import annotations
