"""Web utilities: HTTP fetch, HTML scraping, and search backends.

Import each name from the submodule that defines it (this ``__init__``
re-exports nothing, matching :mod:`wesearch.paper`):

- :mod:`.fetch` -- ``fetch`` (the sole HTTP egress; transparently backed by a
  persistent per-``(egress_ip, domain)`` cookie + User-Agent profile).
- :mod:`.profile` -- ``Profile``, ``ProfileStore`` (the cross-process jar the
  ``fetch`` orchestrator drives).
- :mod:`.errors` -- ``FetchError``, ``BotDetectionError`` + its subclasses.
- :mod:`.scrape` -- ``get_element_content``.
- :mod:`.search` -- ``search`` + ``SearchResult`` and the result types.
- :mod:`.fetch` also exposes ``egress_ip`` / ``last_known_egress_ip``
  (public-egress-IP lookup, memoized) alongside ``fetch``.
- :mod:`.paper` -- scholarly-paper lookup subpackage.

Usage::

    from wesearch.fetch import fetch
    from wesearch.errors import FetchError, BotDetectionError
    from wesearch.scrape import get_element_content
    from wesearch.search import search, SearchResult
"""

from __future__ import annotations
