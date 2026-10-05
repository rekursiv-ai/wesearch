"""Markdown conversion of the whole document, via the ``markdownify`` library."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import markdownify
else:
    from wrapt import lazy_import

    markdownify = lazy_import("markdownify")


__all__ = ["extract_markdownify"]


def extract_markdownify(html: str, *, url: str = "") -> str:
    """Convert an HTML document to Markdown, converting every element.

    Preserves more of a page's structure than the other extractors -- nested
    lists and tables survive where a text-node walk flattens them -- at the cost
    of size, because it converts the document rather than reading its text.

    Args:
      html: The page source.
      url: Unused. ``markdownify`` has no base-URL option, so relative links
        stay relative here where the other extractors resolve them absolute.
        Accepted because :class:`wesearch.types.extractor.Extract` declares
        it.

    Returns:
      text: Markdown text; empty when the document has no content.

    """
    del url
    # Called bare, deliberately. ``strip=["script", "style"]`` looks like the
    # way to drop a page's CSS and JS and does the opposite: ``strip`` removes a
    # tag's MARKUP and keeps its text, while the default conversion already
    # discards both elements wholesale. Passing it turned one dictionary entry
    # into 567_708 characters, 516_564 of them a single CSS rule; bare, the same
    # page yields 19_650.
    return markdownify.markdownify(html)
