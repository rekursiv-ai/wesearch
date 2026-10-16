"""Turn a fetched HTML page into text."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never


if TYPE_CHECKING:
    import html2text
    import markdownify
    import trafilatura

    from wesearch.types.params import Extractor
else:
    from wrapt import lazy_import

    # Lazy despite 10ms: every caller of ``wesearch.web`` imports this
    # module, and most never extract.
    html2text = lazy_import("html2text")  # 10ms.
    markdownify = lazy_import("markdownify")  # 42ms.
    trafilatura = lazy_import("trafilatura")  # 90ms.


__all__ = ["html_to_text"]


def html_to_text(
    page: str,
    *,
    extractor: Extractor = "html2text",
    url: str = "",
) -> str:
    """Return ``page`` as text, empty when the extractor finds none.

    Args:
      page: HTML source.
      extractor: ``html2text`` converts every text node to Markdown.
        ``markdownify`` converts every element, keeping nested lists and tables,
        at about twice the size. ``trafilatura`` keeps only what it scores as
        the article, which loses content on pages that are not articles.
        ``raw`` returns ``page`` unchanged.
      url: The page's URL; ``html2text`` and ``trafilatura`` make relative links
        absolute with it.

    Returns:
      text: The extracted text.

    Raises:
      AssertionError: When ``extractor`` names no extractor.

    """
    if extractor == "html2text":
        converter = html2text.HTML2Text(baseurl=url)
        converter.ignore_images = True
        converter.body_width = 0
        return converter.handle(page)
    if extractor == "markdownify":
        return markdownify.markdownify(page)
    if extractor == "trafilatura":
        return (
            trafilatura.extract(
                page,
                url=url or None,
                output_format="markdown",
                include_links=True,
                include_tables=True,
                with_metadata=True,
            )
            or ""
        )
    if extractor == "raw":
        return page
    assert_never(extractor)
