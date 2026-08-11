"""HTML-to-text extractors: how a fetched page becomes text.

One module per extractor, selected by the ``extractor`` field of
:class:`wesearch.types.params.Policy` and satisfying
:class:`wesearch.types.extractor.Extract` -- the same shape the transports
use.

They are not ranked versions of one algorithm. ``html2text`` converts every text
node to Markdown; ``trafilatura`` scores blocks and returns only what it judges
to be the article. The second is right for a news page and wrong for a
dictionary entry, a Q&A thread, or a profile timeline, where it returns
plausible-looking output with the substance missing.

Deliberately not a facade, matching ``fetch.transport``: importing one extractor
must not pull in the other's dependency.
"""

from __future__ import annotations
