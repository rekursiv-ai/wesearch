"""Site- and service-specific web fetch providers for :mod:`wesearch`.

Each module wraps one awkward target (a render proxy, a site whose useful
content hides behind JS or an API) behind a plain ``fetch_*`` function returning
bytes. The generic transport, anti-bot ladder, cookie/UA profile, and SSRF
policy live in :mod:`wesearch.fetch`; a provider adds only the per-target
URL shaping and response handling that transport cannot know.

Import each name from the submodule that defines it (this package's ``__init__``
re-exports nothing), exactly as :mod:`wesearch.paper.providers` is used.
"""

from __future__ import annotations
