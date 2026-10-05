"""The identity extractor: the page source, untouched."""

from __future__ import annotations


__all__ = ["extract_raw"]


def extract_raw(html: str, *, url: str = "") -> str:
    """Return the document unchanged.

    Args:
      html: The page source.
      url: Unused; present because :class:`wesearch.types.extractor.Extract`
        declares it.

    Returns:
      text: ``html``, verbatim.

    """
    del url
    return html
