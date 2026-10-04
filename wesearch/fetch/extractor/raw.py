"""The identity extractor: the page source, untouched."""

from __future__ import annotations


__all__ = ["extract_raw"]


# pragma: no mutate start -- ``url`` is discarded, so its default is inert.
def extract_raw(html: str, *, url: str = "") -> str:
    """Return the document unchanged.

    Args:
      html: The page source.
      url: Unused; present because :class:`wesearch.types.extractor.Extract`
        declares it.

    Returns:
      text: ``html``, verbatim.

    """
    # pragma: no mutate end
    del url
    return html
