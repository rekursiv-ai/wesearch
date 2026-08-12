"""The extractor protocol: how a fetched HTML page becomes text."""

from __future__ import annotations

from typing import Protocol


__all__ = ["Extract"]


class Extract(Protocol):
    """Turn an HTML document into text.

    One implementation per module under :mod:`wesearch.fetch.extractor`,
    selected by the ``extractor`` field of
    :class:`wesearch.types.params.PolicyParams` -- the same shape the transports
    use.
    """

    def __call__(self, html: str, *, url: str = "") -> str:
        """Return the document's text.

        Args:
          html: The page source.
          url: The page's own URL, for resolving relative links to absolute
            ones. Part of the protocol even though not every implementation
            needs it: an extractor that emits ``](/login)`` hands a reader a
            link it cannot follow, and the caller always knows the URL.

        Returns:
          text: The extracted text; empty when the document yields none.

        """
        ...
