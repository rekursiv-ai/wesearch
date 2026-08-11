"""Markdown conversion of every text node, via the ``html2text`` library."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import html2text
else:
    from wrapt import lazy_import

    html2text = lazy_import("html2text")


__all__ = ["extract_html2text"]


def extract_html2text(html: str, *, url: str = "") -> str:
    """Convert an HTML document to Markdown, keeping every text node.

    Args:
      html: The page source.
      url: The page's own URL, used to resolve relative links to absolute ones.

    Returns:
      text: Markdown text; empty when the document has no body.

    """
    converter = html2text.HTML2Text(baseurl=url)
    # Images carry no text a reader can use, and their base64 data URLs can
    # outweigh a page's prose.
    converter.ignore_images = True
    # Unwrapped: the default rewraps at 78 columns, which splits sentences
    # mid-phrase and breaks any downstream search for a quoted string.
    converter.body_width = 0
    return converter.handle(html)
