"""Site- and service-specific web fetch providers.

Each module wraps one awkward target (a render proxy, a site whose useful
content hides behind JS or an API) behind a plain ``fetch_*`` function returning
bytes. The generic transport, anti-bot ladder, cookie/UA profile, and SSRF
policy live in :mod:`wesearch.fetch`; a provider adds only the per-target
URL shaping and response handling that transport cannot know.

Sited under ``fetch`` rather than at the package root because every provider
here is a fetch strategy -- each one's imports reduce to ``fetch`` plus
``types`` -- mirroring :mod:`wesearch.paper.providers`, which sits under
the layer IT serves.

Import each name from the submodule that defines it (this package's ``__init__``
re-exports nothing), exactly as :mod:`wesearch.paper.providers` is used.
"""

from __future__ import annotations
